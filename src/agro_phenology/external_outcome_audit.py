"""Read-only availability audit of homogeneous external observation outcomes.

This module trains no model and scores no biological or notification outcome.
It counts first registrations in control tables, retaining censored/unknown
entries and explaining overlap of measurement-specific technical trial IDs.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .external_validation_collection import check as check_collection
from .external_validation_io import digest_file, utc_now, write_json


SOURCES = ("zenodo_6520484", "zenodo_6520937")
DEFAULT_COLLECTION = Path("data/external_validation/20261110_external_v2")
CATEGORIES = ("eligible", "positive_at_entry", "right_censored", "unknown_at_entry",
              "short_warning", "outside_season_entry")
GROUPS = ["source_id", "trial_id", "measurement", "unit", "treatment"]
DEFINITION = {
    "version": "1.0.0",
    "purpose": "outcome_availability_only_no_model_scoring_or_biological_validation",
    "sources": list(SOURCES), "population": "untreated_control_true_percent_units_only",
    "unit_exclusion": "source_units_unverified_classes_not_assumed_percent",
    "series_key": GROUPS,
    "visit_key": GROUPS + ["observation_date"],
    "positive": "any_reported_valid_percent_value_greater_than_zero",
    "zero": "every_reported_value_present_valid_and_zero",
    "unknown": "no_confirmed_positive_and_incomplete_or_invalid_reported_values",
    "replicates": "not_invented_conflicting_same_day_rows_aggregated_conservatively",
    "entry": "strictly_first_source_assessment_never_later_convenient_zero",
    "first_issue_assumption": "entry_date_plus_one_calendar_day",
    "last_entry_day": "October_30_for_at_least_one_issue_day",
    "season_end": "October_31",
    "event": "first_positive_registration_no_later_than_season_end",
    "eligible": "zero_at_entry_and_event_at_least_four_days_after_entry",
    "censoring": "last_confirmed_zero_or_positive_assessment_by_season_end",
    "non_visits": "unknown_not_absence",
    "interpretation": "technical_trial_ids_not_author_stated_independent_trials",
    "measurement_semantics": "preserved_as_is_no_German_English_or_organ_alias_merging",
}


def _prepare(frame: pd.DataFrame, source_id: str) -> pd.DataFrame:
    required = {*GROUPS, "site_id", "observation_date", "value", "value_domain_valid", "untreated_control"}
    if not required.issubset(frame):
        raise ValueError(f"Missing audit columns: {sorted(required-set(frame))}")
    if frame.source_id.isna().any() or set(frame.source_id) != {source_id} or source_id not in SOURCES:
        raise ValueError("Exactly one of the two declared datasets is required per input table")
    if frame.untreated_control.isna().any() or not frame.untreated_control.isin([True, False]).all():
        raise ValueError("Unknown control flag")
    controls = frame.loc[frame.untreated_control.eq(True)].copy()
    if controls[GROUPS + ["site_id"]].isna().any().any():
        raise ValueError("Unknown observation identity or measurement unit")
    controls["observation_date"] = pd.to_datetime(controls.observation_date, errors="raise")
    if controls.observation_date.isna().any() or controls.observation_date.dt.tz is not None:
        raise ValueError("Source assessment dates must be known local calendar dates")
    if not controls.observation_date.eq(controls.observation_date.dt.normalize()).all():
        raise ValueError("Source assessment time must not be silently rounded to a date")
    controls["value"] = pd.to_numeric(controls.value, errors="raise").astype(float)
    return controls


def audit_tables(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Count separate measurements; no field dates or IDs enter public summaries."""
    visits, series, excluded, source_base = [], [], [], []
    for source_id in sorted(frames):
        controls = _prepare(frames[source_id], source_id)
        percent = controls[controls.unit.eq("percent")]
        unsupported = controls[~controls.unit.eq("percent")]
        for (measurement, unit), group in unsupported.groupby(["measurement", "unit"], sort=True):
            excluded.append(dict(source_id=source_id, measurement=measurement, unit=unit,
                                 rows=len(group), technical_trial_ids=group.trial_id.nunique(),
                                 sites=group.site_id.nunique(), reason="measurement_unit_unverified_not_assumed_percent"))
        source_base.append(dict(source_id=source_id, control_rows=len(controls),
                                technical_control_trial_ids=controls.trial_id.nunique(), control_sites=controls.site_id.nunique(),
                                percent_control_rows=len(percent), unit_excluded_rows=len(unsupported)))
        for identity, group in percent.groupby(GROUPS, sort=True):
            if group.site_id.nunique() != 1 or group.observation_date.dt.year.nunique() != 1:
                raise ValueError("One technical trial-series must have one site and one calendar year")
            entry_year = int(group.observation_date.dt.year.iloc[0])
            end = pd.Timestamp(year=entry_year, month=10, day=31)
            first_day = group.observation_date.min()
            local_visits = []
            for day, values in group.groupby("observation_date", sort=True):
                valid = (values.value.notna() & np.isfinite(values.value) & values.value.between(0, 100)
                         & values.value_domain_valid.astype("boolean").fillna(False).astype(bool))
                positive = bool((valid & values.value.gt(0)).any())
                zero = bool(valid.all() and values.value.eq(0).all())
                status = "positive" if positive else "zero" if zero else "unknown"
                visit = dict(zip(GROUPS, identity))
                visit.update(site_id=group.site_id.iloc[0], season=entry_year, observation_date=day,
                             status=status, reported_values=len(values), valid_values=int(valid.sum()),
                             unknown_or_invalid_values=int((~valid).sum()),
                             duplicate_group=len(values) > 1,
                             conflicting_group=len(values) > 1 and values.value.nunique(dropna=False) > 1,
                             mixed_zero_positive=bool((valid & values.value.eq(0)).any() and positive),
                             before_or_on_season_end=bool(day <= end))
                local_visits.append(visit)
                visits.append(visit)
            first_status = local_visits[0]["status"]
            eligible_visits = [v for v in local_visits if v["before_or_on_season_end"]]
            positives = [v["observation_date"] for v in eligible_visits if v["status"] == "positive"]
            event = min(positives) if positives else pd.NaT
            zeros = [v["observation_date"] for v in eligible_visits if v["status"] == "zero"
                     and (pd.isna(event) or v["observation_date"] < event)]
            last_zero = max(zeros) if zeros else pd.NaT
            measured = [v["observation_date"] for v in eligible_visits if v["status"] != "unknown"]
            last_assessment = max(measured) if measured else pd.NaT
            included = bool(first_status == "zero" and first_day <= end-pd.Timedelta(days=1))
            warnable = bool(included and pd.notna(event) and pd.notna(last_zero) and (event-first_day).days >= 4)
            category = ("outside_season_entry" if first_day > end-pd.Timedelta(days=1)
                        else "positive_at_entry" if first_status == "positive"
                        else "unknown_at_entry" if first_status == "unknown"
                        else "right_censored" if pd.isna(event)
                        else "eligible" if warnable else "short_warning")
            record = dict(zip(GROUPS, identity))
            record.update(site_id=group.site_id.iloc[0], season=entry_year,
                          entry_date=first_day, entry_status=first_status,
                          first_recorded_event_date=event, last_complete_zero_date=last_zero,
                          last_assessment_date=last_assessment, category=category,
                          cohort_included=included, eligible=warnable,
                          source_rows=len(group), assessment_dates=len(local_visits),
                          registration_gap_days=(event-last_zero).days if pd.notna(event) and pd.notna(last_zero) else np.nan,
                          duplicate_groups=sum(v["duplicate_group"] for v in local_visits),
                          conflicting_groups=sum(v["conflicting_group"] for v in local_visits),
                          mixed_zero_positive_groups=sum(v["mixed_zero_positive"] for v in local_visits))
            series.append(record)
    if not series:
        raise ValueError("No homogeneous control-percent observation series")
    visit_frame, registry = pd.DataFrame(visits), pd.DataFrame(series)
    measure_rows, annual_rows, overlaps, sources = [], [], [], []
    for (source_id, measurement, unit), group in registry.groupby(["source_id", "measurement", "unit"], sort=True):
        eligible = group[group.eligible]
        row = dict(source_id=source_id, measurement=measurement, unit=unit, series=len(group),
                   source_rows=int(group.source_rows.sum()), assessment_dates=int(group.assessment_dates.sum()),
                   sites=group.site_id.nunique(), years=",".join(str(y) for y in sorted(group.season.unique())),
                   eligible_sites=eligible.site_id.nunique(),
                   eligible_counts_by_year=json.dumps({str(y): len(g) for y, g in eligible.groupby("season")}, sort_keys=True),
                   registration_gap_min_days=float(eligible.registration_gap_days.min()) if len(eligible) else np.nan,
                   registration_gap_median_days=float(eligible.registration_gap_days.median()) if len(eligible) else np.nan,
                   registration_gap_max_days=float(eligible.registration_gap_days.max()) if len(eligible) else np.nan,
                   duplicate_groups=int(group.duplicate_groups.sum()), conflicting_groups=int(group.conflicting_groups.sum()),
                   mixed_zero_positive_groups=int(group.mixed_zero_positive_groups.sum()))
        row.update({name: int(group.category.eq(name).sum()) for name in CATEGORIES})
        measure_rows.append(row)
        for year, annual in group.groupby("season", sort=True):
            annual_rows.append(dict(source_id=source_id, measurement=measurement, unit=unit, season=int(year),
                                    series=len(annual), sites=annual.site_id.nunique(),
                                    eligible_sites=annual.loc[annual.eligible, "site_id"].nunique(),
                                    **{name: int(annual.category.eq(name).sum()) for name in CATEGORIES}))
    for source_id, group in registry.groupby("source_id", sort=True):
        eligible = group[group.eligible]
        measures = sorted(eligible.measurement.unique())
        for index, left in enumerate(measures):
            left_trials = set(eligible.loc[eligible.measurement.eq(left), "trial_id"])
            for right in measures[index+1:]:
                overlap = len(left_trials & set(eligible.loc[eligible.measurement.eq(right), "trial_id"]))
                if overlap:
                    overlaps.append(dict(source_id=source_id, measurement_a=left, measurement_b=right,
                                         shared_eligible_technical_trial_ids=overlap))
        base = next(row for row in source_base if row["source_id"] == source_id)
        sources.append(dict(base, percent_measurement_series=len(group), eligible_measurement_series=len(eligible),
                            unique_eligible_technical_trial_ids=eligible.trial_id.nunique(), eligible_sites=eligible.site_id.nunique(),
                            duplicate_groups=int(group.duplicate_groups.sum()), conflicting_groups=int(group.conflicting_groups.sum()),
                            mixed_zero_positive_groups=int(group.mixed_zero_positive_groups.sum()),
                            **{name: int(group.category.eq(name).sum()) for name in CATEGORIES}))
    return dict(series=registry, visits=visit_frame, measure_summary=pd.DataFrame(measure_rows),
                annual_summary=pd.DataFrame(annual_rows), source_summary=pd.DataFrame(sources),
                overlaps=pd.DataFrame(overlaps, columns=["source_id", "measurement_a", "measurement_b", "shared_eligible_technical_trial_ids"]),
                excluded_units=pd.DataFrame(excluded, columns=["source_id", "measurement", "unit", "rows", "technical_trial_ids", "sites", "reason"]))


def _compute(collection: Path) -> dict[str, pd.DataFrame]:
    return audit_tables({source: pd.read_parquet(collection/f"normalized/{source}/observations.parquet") for source in SOURCES})


def _report(tables: dict[str, pd.DataFrame], root: Path) -> str:
    source_text = "; ".join(f"{r.source_id}: {r.technical_control_trial_ids} technical trial_id"
                            for r in tables["source_summary"].itertuples(index=False))
    lines = ["# Доступность первых регистраций в яблочных и виноградных контрольных таблицах", "",
             "10 ноября 2026. Это воспроизводимый аудит схемы и наблюдаемости исходов. Здесь нет модели, обучения, score, оценки биологического развития, точности или качества предупреждений. Все итоговые числа рассчитаны по реальным сохранённым таблицам; синтетика используется только в тестах.", "",
             "Источники Niels Holst: [Zenodo 6520484](https://zenodo.org/records/6520484) и [Zenodo 6520937](https://zenodo.org/records/6520937), лицензии CC BY 4.0. Полная атрибуция сохранена в исходных metadata и хешах входов. Значения датасетов не смешиваются. Публичный отчёт содержит только агрегаты; source IDs отдельных полей, координаты и даты отдельных осмотров остаются в локальных Parquet.", "",
             "## Алгоритм", "",
             "Только явно контрольные варианты и подтверждённые процентные единицы. Группа — technical trial_id × measurement × unit × treatment. Названия measurement сохраняются без слияния немецких/английских кодов или разных органов. Повторные строки одной даты не объявляются новыми делянками: любое валидное >0 подтверждает положительную регистрацию; ноль требует всех сообщённых значений, их допустимости и равенства нулю. Остальное неизвестно. Частичный набор измерений не объявляется полным осмотром всего поля.", "",
             "Вход — строго первый исходный осмотр. Положительный при входе ряд левоцензурирован; неизвестный вход не заменяется удобным последующим нулём. Первая положительная регистрация рассматривается до фиксированного 31 октября; первый потенциальный выпуск — entry+1, поэтому окно предупреждения 3–10 дней требует event≥entry+4. Более короткие события отмечаются отдельно. Последний информативный осмотр задаёт цензурирование; дни без визита неизвестны. Полный нулевой выборочный осмотр не доказывает отсутствия объекта во всём поле.", "",
             f"В проверенных контролях: {source_text}. Технические trial_id не приравниваются к независимым авторским опытам: названия публикаций содержат 89 и 36, соответственно. Нумерация, измерения и их пересечения сохраняются явно.", "",
             "## По каждой однородной мере", "", "| Набор | Measurement | Рядов | Положительный вход | Цензурированы | Unknown вход | Короткие события | Eligible | Мест с eligible |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["measure_summary"].itertuples(index=False):
        lines.append(f"| {row.source_id} | {row.measurement} | {row.series} | {row.positive_at_entry} | {row.right_censored} | {row.unknown_at_entry} | {row.short_warning} | {row.eligible} | {row.eligible_sites} |")
    lines += ["", "Eligible здесь означает только наличие наблюдаемого перехода и возможности предупреждения. Это не пойманное моделью событие и не число независимых заражений.", "",
              "## Годы и интервалы наблюдений", "", "| Набор | Measurement с eligible | Число по годам | Интервал последний ноль → регистрация, min / median / max дней |", "|---|---|---|---:|"]
    for row in tables["measure_summary"].query("eligible > 0").itertuples(index=False):
        lines.append(f"| {row.source_id} | {row.measurement} | {row.eligible_counts_by_year} | {row.registration_gap_min_days:g} / {row.registration_gap_median_days:g} / {row.registration_gap_max_days:g} |")
    lines += ["", "Суммы разных measurement не являются независимой общей событийной популяцией:", "", "| Набор | Measure-specific eligible | Уникальных technical trial_id с ними | Мест | Конфликтующих одинаковых ключей даты | Из них смесь 0 и >0 |", "|---|---:|---:|---:|---:|---:|"]
    for row in tables["source_summary"].itertuples(index=False):
        lines.append(f"| {row.source_id} | {row.eligible_measurement_series} | {row.unique_eligible_technical_trial_ids} | {row.eligible_sites} | {row.conflicting_groups} | {row.mixed_zero_positive_groups} |")
    lines += ["", "Все конфликтующие смеси 0 и >0 признаны positive. Ни усреднение, ни выбор удобного нулевого дубля не увеличивали число переходов. В исходных процентных контролях текущих версий отсутствуют неизвестные/недопустимые значения; ветка unknown проверяется синтетическими тестами.", "",
              "| Набор | Пара measurement | Общих eligible technical trial_id |", "|---|---|---:|"]
    for row in tables["overlaps"].itertuples(index=False):
        lines.append(f"| {row.source_id} | {row.measurement_a} / {row.measurement_b} | {row.shared_eligible_technical_trial_ids} |")
    lines += ["", "## Исключённые неподтверждённые единицы", "", "| Набор | Measurement | Единица | Контрольных строк | Technical trial_id |", "|---|---|---|---:|---:|"]
    for row in tables["excluded_units"].itertuples(index=False):
        lines.append(f"| {row.source_id} | {row.measurement} | {row.unit} | {row.rows} | {row.technical_trial_ids} |")
    lines += ["", "CLASSES не переименовывается в процентную частоту или интенсивность. Источники/органы/методы не объединяются ради большего знаменателя.", "",
              "## Что остаётся для временного исследования", "",
              "Следующая таблица — диагностическое разложение доступности, не предложенный подобранный split: первые три наблюдавшихся года каждой меры условно оставлены начальной историей, показаны все более поздние годы. Если их нет, нет и внешних событий. Положительные при входе и повторные осмотры не увеличивают знаменатель.", "",
              "| Набор | Measurement | Начальные годы | Eligible в истории | Последующие годы | Eligible далее | Правое цензурирование далее | Мест с eligible далее |", "|---|---|---|---:|---|---:|---:|---:|"]
    for identity, group in tables["series"].groupby(["source_id", "measurement"], sort=True):
        if not group.eligible.any():
            continue
        years = sorted(group.season.unique())
        warmup = years[:3]
        history = group[group.season.isin(warmup)]
        later = group[group.season.gt(warmup[-1])]
        later_eligible = later[later.eligible]
        lines.append(f"| {identity[0]} | {identity[1]} | {','.join(map(str,warmup))} | {int(history.eligible.sum())} | {','.join(map(str,years[3:])) or 'нет'} | {int(later.eligible.sum())} | {int(later.category.eq('right_censored').sum())} | {later_eligible.site_id.nunique()} |")
    lines += ["", "У яблони крупнейшая отдельная мера даёт лишь четыре пригодных регистрации, а после начальной истории больших временных знаменателей не возникает. У виноградной LEAF_P%INF возможен небольшой отдельный статистический пилот, но редкие осмотры и интервалы до 100 дней не доказывают точности биологической даты. Фенология и увлажнение листа этим аудитом не проверяются и не восстанавливаются.", "",
              "Ни Hutton, ни картофельные таблицы, ни механизмы развития патогенов этот модуль не рассчитывает и не аудирует.", "",
              "## Артефакты и воспроизведение", "",
              "Локальные series.parquet/visits.parquet сохраняют причины отбора и ключи для проверки. measure_summary, annual_summary, source_summary, overlaps и excluded_units — отдельные агрегированные таблицы. manifest.json содержит хеши всех входов, кода и результатов; audit_definition.json фиксирует алгоритм. Старые коллекции и результаты не обновляются.", "", "```bash",
              f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_outcome_audit verify --output-dir {root}",
              "PYTHONPATH=src .venv/bin/python -m agro_phenology.external_outcome_audit run --output-dir results/external_validation/outcome_audit/reproduction_NEW",
              "PYTHONPATH=src .venv/bin/python -m pytest tests/test_external_outcome_audit.py -q", "```", "",
              "Verify проверяет файлы/входы и заново рассчитывает все семь таблиц, не обращаясь к сети. Повторный run требует нового каталога.", ""]
    return "\n".join(lines)


def run(collection: Path, root: Path) -> dict:
    if root.exists():
        raise FileExistsError("Outcome audit run is immutable; choose a new directory")
    checked = check_collection(collection)
    if checked["status"] != "passed":
        raise ValueError(f"Collection integrity audit failed: {checked}")
    inputs = [Path(__file__), collection/"manifest.json"]
    for source in SOURCES:
        inputs.extend([collection/f"normalized/{source}/observations.parquet", collection/f"raw/{source}/metadata.json"])
    hashes = {str(p.resolve()): digest_file(p) for p in inputs}
    root.mkdir(parents=True)
    write_json(root/"input_audit.json", checked)
    write_json(root/"audit_definition.json", DEFINITION)
    snapshot = root/"source_snapshot"/Path(__file__).name
    snapshot.parent.mkdir()
    shutil.copyfile(Path(__file__), snapshot)
    tables = _compute(collection)
    for name, table in tables.items():
        table.to_parquet(root/f"{name}.parquet", index=False)
        if name not in {"series", "visits"}:
            table.to_csv(root/f"{name}.csv", index=False)
    (root/"REPORT_RU.md").write_text(_report(tables, root), encoding="utf-8")
    if any(digest_file(Path(p)) != sha for p, sha in hashes.items()):
        raise RuntimeError("Inputs changed during audit; output is not sealed")
    inventory = {str(p.relative_to(root)): digest_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root/"manifest.json", dict(status="complete", purpose=DEFINITION["purpose"], created_at=utc_now(),
                                        collection_dir=str(collection.resolve()), input_hashes=hashes, file_inventory=inventory,
                                        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__))
    return dict(status="complete", output_dir=str(root), source_summary=tables["source_summary"].to_dict("records"))


def verify(root: Path) -> dict:
    manifest = json.loads((root/"manifest.json").read_text())
    failures = [p for p, sha in manifest["file_inventory"].items() if not (root/p).is_file() or digest_file(root/p) != sha]
    inputs = [p for p, sha in manifest["input_hashes"].items() if not Path(p).is_file() or digest_file(Path(p)) != sha]
    checked = []
    if not failures and not inputs:
        tables = _compute(Path(manifest["collection_dir"]))
        for name, table in tables.items():
            try:
                pd.testing.assert_frame_equal(table, pd.read_parquet(root/f"{name}.parquet"), check_exact=True)
                checked.append(name)
            except AssertionError:
                failures.append("recompute:"+name)
        if json.loads((root/"audit_definition.json").read_text()) != DEFINITION:
            failures.append("recompute:audit_definition")
    return dict(status="passed" if not failures and not inputs and manifest["status"] == "complete" else "failed",
                checked_files=len(manifest["file_inventory"]), checked_inputs=len(manifest["input_hashes"]),
                recomputed_tables=checked, mismatches=failures, input_mismatches=inputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.collection_dir, args.output_dir) if args.command == "run" else verify(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
