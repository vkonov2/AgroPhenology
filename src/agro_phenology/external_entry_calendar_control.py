"""Post-hoc fixed entry-duration control against frozen native v2 models.

This supplement does not refit AFT or the original DOY calendar. It asks whether
the observed timing advantage remains beyond knowing when monitoring started.
It was specified after inspecting v2, so is not an untouched independent test.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .early_warning_models import simulate_policy
from .external_event_aft import _prediction_entry, _training_rows, feature_frame
from .external_native_validation import (AFT, CALENDAR, POLICY, SCOPES, _unit_metrics,
                                         aggregate_metrics, paired_uncertainty)
from .external_validation_io import digest_file, utc_now, write_json


CONTRACT = Path("docs/research/external_validation/entry_calendar_contract.json")
ENTRY = "calendar_entry_duration"
MIN_GROUP_EVENTS = 3
BASE_TABLES = ["event_registry", "decisions", "predictions", "notifications", "per_series_metrics", "metrics"]


def fit_entry_calendar(registry: pd.DataFrame, outer_year: int, use_station: bool = True) -> dict:
    train, metadata = _training_rows(registry, outer_year)
    observed = train[train._observed].copy()
    result = dict(metadata, model_kind=ENTRY, use_station=use_station, min_group_events=MIN_GROUP_EVENTS,
                  global_median_duration=int(max(1, np.floor(observed._duration.median()+.5))), groups={})
    if use_station:
        feature_frame(train, use_station=True)
        for cluster, group in observed.groupby(observed.cluster.astype(str), sort=True):
            if len(group) >= MIN_GROUP_EVENTS:
                result["groups"][cluster] = dict(events=len(group), median_duration=int(max(1, np.floor(group._duration.median()+.5))))
    return result


def predict_entry_calendar(calendar: dict, rows: pd.DataFrame) -> pd.Series:
    entry = _prediction_entry(rows, calendar)
    if calendar["use_station"]:
        feature_frame(rows, use_station=True)
        durations = [calendar["groups"].get(str(cluster), {}).get("median_duration", calendar["global_median_duration"])
                     for cluster in rows.cluster]
    else:
        durations = [calendar["global_median_duration"]] * len(rows)
    return pd.Series([day+pd.Timedelta(days=int(duration)) for day, duration in zip(entry, durations)],
                     index=rows.index, dtype="datetime64[ns]", name="predicted_date")


def base_audit(base: Path) -> dict:
    manifest = json.loads((base/"manifest.json").read_text())
    files = [name for name, sha in manifest["file_inventory"].items()
             if not (base/name).is_file() or digest_file(base/name) != sha]
    inputs = [name for name, sha in manifest["input_hashes"].items()
              if not Path(name).is_file() or digest_file(Path(name)) != sha]
    missing = [name for name in BASE_TABLES if not (base/f"{name}.parquet").is_file()]
    return dict(status="passed" if manifest["status"] == "complete" and not files and not inputs and not missing else "failed",
                checked_files=len(manifest["file_inventory"]), checked_inputs=len(manifest["input_hashes"]),
                mismatches=files, input_mismatches=inputs, missing_tables=missing)


def _compute(base: Path) -> dict[str, pd.DataFrame]:
    manifest = json.loads((base/"manifest.json").read_text())
    target, zero_mode = manifest["target"], manifest["zero_mode"]
    registry = pd.read_parquet(base/"event_registry.parquet")
    decisions = pd.read_parquet(base/"decisions.parquet")
    frozen_units = pd.read_parquet(base/"per_series_metrics.parquet")
    frozen_metrics = pd.read_parquet(base/"metrics.parquet")
    pd.testing.assert_frame_equal(aggregate_metrics(frozen_units, target, zero_mode), frozen_metrics, check_exact=True)
    outer = registry[registry.field_season.isin(decisions.field_season)].copy()
    if set(outer.field_season) != set(frozen_units.field_season):
        raise ValueError("Frozen decision and metric populations differ")
    if set(frozen_units.model_code) != {AFT, CALENDAR}:
        raise ValueError("Expected frozen AFT and DOY calendar models")
    # Inspect the saved policy, without rerunning or modifying either system.
    policy = pd.read_parquet(base/"notifications.parquet", columns=[
        "policy_threshold", "policy_active_days", "policy_cooldown_days", "policy_version"]).drop_duplicates()
    if len(policy) != 1 or policy.iloc[0].to_dict() != dict(policy_threshold=POLICY.threshold,
            policy_active_days=POLICY.active_days, policy_cooldown_days=POLICY.cooldown_days, policy_version=POLICY.version):
        raise ValueError("Frozen notification policy does not match the declared unchanged P0")
    history, predicted = [], []
    for year in sorted(outer.season.unique()):
        train = registry[registry.cohort_included & registry.season.lt(year)].copy()
        test = outer[outer.season.eq(year)].copy()
        calendar = fit_entry_calendar(train, int(year), use_station=target != "cpb_wisconsin")
        history.append(dict(season=int(year), training_years=",".join(map(str, calendar["training_years"])),
                            training_max_year=calendar["training_max_year"], observed_events=calendar["observed_events"],
                            global_median_duration=calendar["global_median_duration"], calendar_json=json.dumps(calendar, sort_keys=True)))
        rows = test[["field_season", "season"]].copy()
        rows["model_code"], rows["predicted_date"] = ENTRY, predict_entry_calendar(calendar, test)
        predicted.append(rows)
    predictions = pd.concat(predicted, ignore_index=True)
    dates = decisions.field_season.map(predictions.set_index("field_season").predicted_date)
    score = (dates-decisions.issue_date).dt.days.between(3, 10).astype(float).where(dates.notna())
    replay = simulate_policy(decisions, score, POLICY, evaluation_scope="full_service_calendar", evaluation_mask=decisions.service_active)
    replay["model_code"], replay["target"], replay["zero_mode"] = ENTRY, target, zero_mode
    replay["source_id"] = manifest.get("source_id", registry.source_id.iloc[0])
    replay["predicted_date"] = dates
    replay["issued_at_semantics"] = decisions.issued_at_semantics
    unit_blocks = [frozen_units]
    for scope in SCOPES:
        rows = _unit_metrics(replay, outer, predictions, scope)
        rows["model_code"] = ENTRY
        unit_blocks.append(rows)
    units = pd.concat(unit_blocks, ignore_index=True)
    metrics = aggregate_metrics(units, target, zero_mode)
    # The full frozen histories remain in base; only the new control's replay
    # is duplicated here. Their hashes and original per-series metrics are pinned.
    gained = []
    for scope in SCOPES:
        observed = units[units.evaluation_scope.eq(scope) & units.warnable_first_event]
        pair = observed.pivot(index="field_season", columns="model_code", values="timely_hit")
        pair["category"] = np.select([pair[ENTRY] & pair[AFT], pair[ENTRY], pair[AFT]],
                                      ["both", "entry_only", "aft_only"], default="neither")
        pair = pair[[ENTRY, AFT, "category"]].reset_index()
        pair["evaluation_scope"] = scope
        gained.append(pair)
    tables = dict(predictions=predictions, notifications=replay, per_series_metrics=units,
                  metrics=metrics, gained_lost_vs_AFT=pd.concat(gained, ignore_index=True), training_history=pd.DataFrame(history))
    for frame in tables.values():
        frame.columns.name = None
    return tables


def uncertainty(units: pd.DataFrame, target: str) -> dict:
    comparison = units[units.model_code.isin([ENTRY, AFT])].copy()
    # This temporary adapter invokes the frozen paired estimator; explicit
    # metadata prevents confusing its reference with the original DOY calendar.
    comparison.loc[comparison.model_code.eq(ENTRY), "model_code"] = CALENDAR
    return dict(candidate_model=AFT, reference_model=ENTRY,
                interpretation="post_hoc_diagnostic_not_new_independent_test",
                results=paired_uncertainty(comparison, target))


def _inputs(base: Path) -> list[Path]:
    code = Path(__file__).parent
    return [CONTRACT, Path(__file__), code/"external_event_aft.py", code/"external_native_validation.py",
            code/"early_warning_models.py", base/"manifest.json", *[base/f"{name}.parquet" for name in BASE_TABLES]]


def report_text(tables: dict, manifest: dict, ci: dict, base: Path, root: Path) -> str:
    main = tables["metrics"][tables["metrics"].fold_id.eq("all_outer_years")]
    lines = ["# Дополнительный календарный контроль от даты начала наблюдений", "",
             f"Target: {manifest['target']}; zero_mode: {manifest['zero_mode']}. Отдельный диагностический опыт поверх замороженного v2, без объединения разных видов и датасетов.", "",
             "Контроль добавлен ПОСЛЕ просмотра результатов v2: проверяется объяснение, что AFT использует известную дату начала учётов. Это post-hoc диагностика, а не новый независимый тест и не заранее предусмотренное доказательство преимущества.", "",
             "Новый календарь: дата первого нулевого учёта + медиана времени до первой регистрации по строго прошлым годам; локальная медиана при ≥3 наблюдённых событиях, иначе общая. Wisconsin использует общую медиану без выдуманной географии. Короткие события входят в обучение медианы; правые цензурирования не получают вымышленных дат событий. Округление половины вверх, минимум день, без обрезания 31 октября.", "",
             "AFT и старый календарь по дню года не переобучались: используются их сохранённые прогнозы, per-series метрики и полный журнал из base-run. Политика P0, окно 3–10 дней, события, даты решения и бюджеты не менялись. Новый контроль имеет отдельную историю уведомлений.", "",
             "| Scope | Метод | Своевременно | Покрытие | Сообщения | Тревожные дни / дни | Бюджет сообщений / тревог превышен |", "|---|---|---:|---:|---:|---:|---|"]
    for r in main.itertuples(index=False):
        lines.append(f"| {r.evaluation_scope} | {r.model_code} | {r.timely_hits}/{r.warnable_events} | {r.timely_coverage:.1%} | {r.messages} | {r.active_alarm_days}/{r.field_days} | {r.message_budget_exceeded} / {r.alarm_budget_exceeded} |")
    lines += ["", "`observed_followup` ограничивает нагрузку последним измеренным визитом; `full_service_calendar` включает неизвестный хвост до 31 октября. Неизвестное продолжение не становится отрицательным исходом. Оценивается первая регистрация, не биологическое появление или повреждение культуры. MAE в таблицах относится только к warnable-событиям.", "",
              "## Парное сравнение AFT с новым контролем", ""]
    for scope, blocks in ci["results"].items():
        counts = tables["gained_lost_vs_AFT"].loc[lambda x: x.evaluation_scope.eq(scope), "category"].value_counts()
        lines.append(f"- {scope}: " + "; ".join(f"{name} {int(counts.get(name,0))}" for name in ["both", "entry_only", "aft_only", "neither"]) + ".")
        for key, block in blocks.items():
            bounds = block["coverage_difference"]
            if bounds["p025"] is not None:
                lines.append(f"- {scope}, {key}, {block['clusters']} блоков: разность покрытия AFT−ENTRY {100*bounds['p025']:+.1f}…{100*bounds['p975']:+.1f} п.п.")
    lines += ["", "Неравное фактическое число сообщений нельзя выдавать за сравнение при равной нагрузке. Бюджеты прежние: 2 сообщения/30 полевых дней и 50% тревожных дней; внешние превышения сохранены. Этот контроль не показывает пользу погоды и не доказывает независимость полевых серий.", "",
              "## Воспроизведение", "", f"Замороженный base-run: `{base}`. Его файлы не менялись; полные исходные журналы остаются там. Новый каталог хранит только новый replay, три сравниваемые метрики, парные исходы, историю медиан и хеши. Код и первичный контракт v2 не изменялись.", "", "```bash",
              f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_entry_calendar_control verify --output-dir {root}",
              f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_entry_calendar_control run --base-run {base} --output-dir results/external_validation/entry_control_REPRO_NEW", "```", "",
              "Verify проверяет исходный frozen bundle, все входы/выходы дополнения и пересчитывает контроль, replay, все таблицы и bootstrap. Исходные идентификаторы и сведения отдельных полей остаются локально; реальных уведомлений нет.", ""]
    return "\n".join(lines)


def run(base: Path, root: Path) -> dict:
    if root.exists():
        raise FileExistsError("Supplement is immutable; choose a new output directory")
    audit = base_audit(base)
    if audit["status"] != "passed":
        raise ValueError(f"Frozen base failed integrity audit: {audit}")
    paths = _inputs(base)
    hashes = {str(path.resolve()): digest_file(path) for path in paths}
    base_manifest = json.loads((base/"manifest.json").read_text())
    root.mkdir(parents=True)
    shutil.copyfile(CONTRACT, root/"contract.json")
    write_json(root/"base_audit.json", audit)
    for path in paths[:5]:
        dest = root/"source_snapshot"/path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
    tables = _compute(base)
    for name, frame in tables.items():
        frame.to_parquet(root/f"{name}.parquet", index=False)
        if name in {"metrics", "training_history", "gained_lost_vs_AFT"}:
            frame.to_csv(root/f"{name}.csv", index=False)
    ci = uncertainty(tables["per_series_metrics"], base_manifest["target"])
    write_json(root/"uncertainty.json", ci)
    (root/"REPORT_RU.md").write_text(report_text(tables, base_manifest, ci, base, root), encoding="utf-8")
    if any(digest_file(Path(path)) != sha for path, sha in hashes.items()):
        raise RuntimeError("Inputs changed during supplement; not sealed")
    inventory = {str(path.relative_to(root)): digest_file(path) for path in sorted(root.rglob("*")) if path.is_file()}
    write_json(root/"manifest.json", dict(status="complete", created_at=utc_now(), base_run=str(base.resolve()),
              target=base_manifest["target"], zero_mode=base_manifest["zero_mode"],
              input_hashes=hashes, file_inventory=inventory, interpretation="post_hoc_diagnostic"))
    main = tables["metrics"]
    result = dict(status="complete", output_dir=str(root), target=base_manifest["target"], zero_mode=base_manifest["zero_mode"],
                  metrics=main[main.fold_id.eq("all_outer_years")].to_dict("records"))
    del tables
    gc.collect()
    return result


def verify(root: Path) -> dict:
    manifest = json.loads((root/"manifest.json").read_text())
    files = [name for name, sha in manifest["file_inventory"].items() if not (root/name).is_file() or digest_file(root/name) != sha]
    inputs = [name for name, sha in manifest["input_hashes"].items() if not Path(name).is_file() or digest_file(Path(name)) != sha]
    base = Path(manifest["base_run"])
    audit = base_audit(base)
    checked = []
    if not files and not inputs and audit["status"] == "passed":
        tables = _compute(base)
        for name, frame in tables.items():
            try:
                pd.testing.assert_frame_equal(frame, pd.read_parquet(root/f"{name}.parquet"), check_exact=True)
                checked.append(name)
            except AssertionError:
                files.append("recompute:"+name)
        if uncertainty(tables["per_series_metrics"], manifest["target"]) != json.loads((root/"uncertainty.json").read_text()):
            files.append("recompute:uncertainty")
        del tables
        gc.collect()
    return dict(status="passed" if not files and not inputs and audit["status"] == "passed" and manifest["status"] == "complete" else "failed",
                checked_files=len(manifest["file_inventory"]), checked_inputs=len(manifest["input_hashes"]),
                recomputed_tables=checked, base_audit=audit, mismatches=files, input_mismatches=inputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run" and args.base_run is None:
        parser.error("run requires --base-run")
    result = run(args.base_run, args.output_dir) if args.command == "run" else verify(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
