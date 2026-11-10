"""Fixed temporal first-registration experiments; metrics never pool endpoints."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .early_warning_models import Policy, simulate_policy, wilson_interval
from .external_event_aft import (aft_training_metadata, fit_aft, fit_calendar,
                                 predict_aft_dates, predict_calendar_dates)
from .external_native_cohorts import TARGET_SPECS, build_decisions, build_registry, load_visits
from .external_validation_collection import check as check_collection
from .external_validation_io import digest_file, utc_now, write_json

CONTRACT = Path("docs/research/external_validation/native_event_contract.json")
COLLECTION = Path("data/external_validation/20261110_external_v2")
CALENDAR = "calendar_past_station_median"
AFT = "CatBoost_AFT_entry"
MODELS = [CALENDAR, AFT]
POLICY = Policy(.5, active_days=7, cooldown_days=15, version="native_fixed_P0_v1")
SCOPES = ["observed_followup", "full_service_calendar"]


def target_config(target: str) -> dict:
    return json.loads(CONTRACT.read_text())["targets"][target]


def _unit_metrics(states: pd.DataFrame, registry: pd.DataFrame, predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Aggregate once per series, preserving the existing sequential policy."""
    mask = states.evaluation_field_day if scope == "observed_followup" else states.service_active
    scored = states.loc[mask].copy()
    scored["computable"] = scored.score.notna()
    agg = scored.groupby("field_season").agg(field_days=("issue_date", "size"), messages=("message_issued", "sum"),
              active_alarm_days=("alarm_active", "sum"), computable_days=("computable", "sum"),
              suppressed_repeats=("suppressed_repeat", "sum"))
    rows = registry[["source_id", "target", "zero_mode", "field_season", "season", "cluster", "warnable_first_event", "first_recorded_event_date"]].set_index("field_season").copy()
    rows = rows.join(agg).fillna({c: 0 for c in agg.columns})
    for column in agg:
        rows[column] = rows[column].astype(int)
    known_event_dates = registry.set_index("field_season").first_recorded_event_date
    messages = scored[scored.message_issued].copy()
    messages["lead"] = (messages.field_season.map(known_event_dates)-messages.issue_date).dt.days
    timely = messages[messages.lead.between(3, 10)].groupby("field_season").size()
    rows["timely_hit"] = rows.index.to_series().map(timely).fillna(0).gt(0) & rows.warnable_first_event
    for name, condition in [("too_early_messages", messages.lead.gt(10)), ("late_messages", messages.lead.lt(3))]:
        counts = messages[condition].groupby("field_season").size()
        rows[name] = rows.index.to_series().map(counts).fillna(0).astype(int)
    opportunity = states[states.service_active & states.days_to_first_recorded_event.between(3, 10)]
    available = opportunity.assign(computable=opportunity.score.notna()).groupby("field_season").computable.any()
    rows["computable_event"] = rows.index.to_series().map(available).astype("boolean").fillna(False).astype(bool) & rows.warnable_first_event
    dates = predictions.set_index("field_season").predicted_date
    rows["predicted_date"] = rows.index.to_series().map(dates)
    rows["absolute_date_error_days"] = (rows.predicted_date-rows.first_recorded_event_date).dt.days.abs().where(rows.warnable_first_event)
    rows["evaluation_scope"] = scope
    return rows.reset_index()


def aggregate_metrics(units: pd.DataFrame, target: str, zero_mode: str) -> pd.DataFrame:
    identity = {"target": target, "source_id": TARGET_SPECS[target]["source_id"], "zero_mode": zero_mode}
    if any(key not in units or units[key].isna().any() or set(units[key]) != {value} for key, value in identity.items()):
        raise ValueError("Do not pool targets, datasets or zero interpretations")
    records = []
    for (model, scope), group in units.groupby(["model_code", "evaluation_scope"], sort=True):
        for year in [None, *sorted(group.season.unique())]:
            part = group if year is None else group[group.season.eq(year)]
            events, hits = int(part.warnable_first_event.sum()), int(part.timely_hit.sum())
            days, messages, alarms = (int(part[c].sum()) for c in ["field_days", "messages", "active_alarm_days"])
            lower, upper = wilson_interval(hits, events)
            rate = 30*messages/days if days else np.nan
            fraction = alarms/days if days else np.nan
            records.append(dict(target=target, source_id=TARGET_SPECS[target]["source_id"],
                organism=TARGET_SPECS[target]["organism"], zero_mode=zero_mode, model_code=model,
                evaluation_scope=scope, fold_id="all_outer_years" if year is None else str(year),
                series=len(part), clusters=part.cluster.nunique(), first_events=int(part.first_recorded_event_date.notna().sum()),
                warnable_events=events, timely_hits=hits, timely_coverage=hits/events if events else np.nan,
                descriptive_wilson_low=lower, descriptive_wilson_high=upper,
                field_days=days, messages=messages, active_alarm_days=alarms,
                messages_per_30_field_days=rate, alarm_fraction=fraction,
                computable_days=int(part.computable_days.sum()), computability=part.computable_days.sum()/days if days else np.nan,
                computable_events=int(part.computable_event.sum()),
                suppressed_repeats=int(part.suppressed_repeats.sum()),
                date_mae_days=float(part.absolute_date_error_days.mean()),
                message_budget_exceeded=bool(rate > 2), alarm_budget_exceeded=bool(fraction > .5)))
    return pd.DataFrame(records)


def paired_uncertainty(units: pd.DataFrame, target: str) -> dict:
    output = {}
    for scope in SCOPES:
        base = units[units.model_code.eq(CALENDAR) & units.evaluation_scope.eq(scope)].set_index("field_season")
        aft = units[units.model_code.eq(AFT) & units.evaluation_scope.eq(scope)].set_index("field_season")
        if base.index.has_duplicates or aft.index.has_duplicates or set(base.index) != set(aft.index):
            raise ValueError("Paired populations differ")
        aft = aft.reindex(base.index)
        common = ["season", "cluster", "warnable_first_event", "field_days"]
        if not base[common].equals(aft[common]):
            raise ValueError("Paired event definitions or exposure differ")
        blocks = {}
        for key in (["season"] if target == "cpb_wisconsin" else ["cluster", "season"]):
            grouped = []
            for name in base[key].unique():
                mask = base[key].eq(name)
                grouped.append([base.loc[mask, "warnable_first_event"].sum(), base.loc[mask, "field_days"].sum(),
                    aft.loc[mask, "timely_hit"].sum()-base.loc[mask, "timely_hit"].sum(),
                    aft.loc[mask, "messages"].sum()-base.loc[mask, "messages"].sum(),
                    aft.loc[mask, "active_alarm_days"].sum()-base.loc[mask, "active_alarm_days"].sum()])
            values = np.array(grouped, dtype=float)
            rng = np.random.default_rng(20261110)
            selected = rng.integers(0, len(values), size=(2000, len(values)))
            draws = values[selected].sum(axis=1)
            valid_event, valid_days = draws[:, 0] > 0, draws[:, 1] > 0
            stats = {"coverage_difference": draws[valid_event, 2]/draws[valid_event, 0],
                     "messages_per_30_days_difference": 30*draws[valid_days, 3]/draws[valid_days, 1],
                     "alarm_fraction_difference": draws[valid_days, 4]/draws[valid_days, 1]}
            blocks[key] = dict(clusters=len(values), replicates=2000, **{name: {
                "valid_replicates": len(arr), "p025": float(np.quantile(arr, .025)) if len(arr) else None,
                "p975": float(np.quantile(arr, .975)) if len(arr) else None} for name, arr in stats.items()})
        output[scope] = blocks
    return output


def _compute(collection: Path, target: str, zero_mode: str, model_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    config = target_config(target)
    visits = load_visits(collection, target, zero_mode)
    registry = build_registry(visits)
    outer = registry[registry.cohort_included & registry.season.isin(config["outer_years"])].copy()
    predicted, history = [], []
    use_station = target != "cpb_wisconsin"
    if model_dir is not None:
        model_dir.mkdir()
    for year in config["outer_years"]:
        train = registry[registry.cohort_included & registry.season.lt(year)].copy()
        test = outer[outer.season.eq(year)].copy()
        if test.empty:
            continue
        if train.season.nunique() < 3:
            raise ValueError("Insufficient predeclared past years; do not silently drop an outer year")
        model = fit_aft(train, outer_year=year, use_station=use_station)
        calendar = fit_calendar(train, outer_year=year, use_station=use_station)
        metadata = aft_training_metadata(model)
        history.append(dict(season=year, training_years=",".join(map(str, sorted(train.season.unique()))),
                            training_max_year=int(train.season.max()), training_series=len(train),
                            metadata_json=json.dumps(metadata, sort_keys=True), calendar_json=json.dumps(calendar, sort_keys=True)))
        if model_dir is not None:
            model.save_model(str(model_dir/f"{year}.cbm"))
            write_json(model_dir/f"{year}_calendar.json", calendar)
        for code, dates in [(CALENDAR, predict_calendar_dates(calendar, test)), (AFT, predict_aft_dates(model, test))]:
            part = test[["field_season", "season"]].copy()
            part["model_code"], part["predicted_date"] = code, dates
            predicted.append(part)
    predictions = pd.concat(predicted, ignore_index=True)
    decisions = build_decisions(outer)
    states, units = [], []
    for code in MODELS:
        point = predictions[predictions.model_code.eq(code)]
        dates = decisions.field_season.map(point.set_index("field_season").predicted_date)
        score = (dates-decisions.issue_date).dt.days.between(3, 10).astype(float).where(dates.notna())
        replay = simulate_policy(decisions, score, POLICY, evaluation_scope="full_service_calendar", evaluation_mask=decisions.service_active)
        replay["model_code"], replay["target"], replay["zero_mode"] = code, target, zero_mode
        replay["source_id"] = TARGET_SPECS[target]["source_id"]
        replay["predicted_date"] = dates
        replay["issued_at_semantics"] = decisions.issued_at_semantics
        states.append(replay)
        for scope in SCOPES:
            part = _unit_metrics(replay, outer, point, scope)
            part["model_code"] = code
            units.append(part)
    per_unit = pd.concat(units, ignore_index=True)
    metrics = aggregate_metrics(per_unit, target, zero_mode)
    eligible = per_unit[per_unit.evaluation_scope.eq("observed_followup") & per_unit.warnable_first_event]
    pair = eligible.pivot(index="field_season", columns="model_code", values="timely_hit")
    pair["category"] = np.select([pair[CALENDAR] & pair[AFT], pair[CALENDAR], pair[AFT]],
                                  ["both", "calendar_only", "aft_only"], default="neither")
    # Parquet does not preserve the pivot's column-axis name. Normalize it
    # before sealing and recomputing, without changing any event or value.
    pair.columns.name = None
    return dict(visits=visits, event_registry=registry, predictions=predictions, decisions=decisions,
                notifications=pd.concat(states, ignore_index=True), per_series_metrics=per_unit,
                metrics=metrics, gained_lost=pair.reset_index(), training_history=pd.DataFrame(history))


def _input_paths(collection: Path, target: str) -> list[Path]:
    code = Path(__file__).parent
    paths = [CONTRACT, Path(__file__), code/"external_native_cohorts.py", code/"external_event_aft.py",
             code/"early_warning_models.py", collection/"manifest.json",
             collection/f"normalized/{TARGET_SPECS[target]['source_id']}/observations.parquet"]
    return paths


def report_text(tables: dict, target: str, zero_mode: str, ci: dict, root: Path) -> str:
    config = target_config(target)
    registry = tables["event_registry"]
    outer = registry[registry.cohort_included & registry.season.isin(config["outer_years"])]
    metrics = tables["metrics"]
    main = metrics[metrics.fold_id.eq("all_outer_years")]
    lines = [f"# {TARGET_SPECS[target]['organism']}: модель времени первой регистрации", "",
        f"Набор: {TARGET_SPECS[target]['source_id']}. Интерпретация нулей: `{zero_mode}`. Раздельный ретроспективный опыт, без объединения с другими наборами/видами. Константы и правила зафиксированы до первого обучения и оценки.", "",
        "## Результаты", "", "| Scope | Метод | Своевременно | Покрытие | Сообщения | Тревожные дни / полевые дни | Вычислимость |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in main.itertuples(index=False):
        lines.append(f"| {r.evaluation_scope} | {r.model_code} | {r.timely_hits}/{r.warnable_events} | {r.timely_coverage:.1%} | {r.messages} | {r.active_alarm_days}/{r.field_days} | {r.computability:.1%} |")
    counts = tables["gained_lost"].category.value_counts().to_dict()
    lines += ["", "Одни и те же пригодные события: " + "; ".join(f"{k}: {counts.get(k,0)}" for k in ["both", "calendar_only", "aft_only", "neither"]) + ".", "",
        "Своевременно означает сообщение за 3–10 календарных дней до первой положительной регистрации. Это не точность истинной даты появления, не оценка повреждения растения и не суточная accuracy. Score — индикатор попадания предсказанной даты в +3…+10, не калиброванная вероятность.", "",
        "## Модель и проверка времени", "",
        "Это НОВАЯ статистическая модель CatBoost SurvivalAft, не воспроизведение коэффициентов AHDB и не проверка погодной формулы UW. Оценивается медиана времени до первой регистрации от наблюдённого нулевого входа. Обнаруженный исход задаёт точечное время; правое цензурирование — нижнюю границу до последнего измеренного визита с неизвестным продолжением. Короткие события остаются в обучении и нагрузке, но не добавляются к знаменателю своевременных предупреждений.", "",
        "[CatBoost SurvivalAft](https://catboost.ai/docs/en/concepts/loss-functions-regression): depth3, iterations200, learning_rate0.03, l2_leaf_reg10, Normal scale1, bootstrap No, seed20261110. RawFormulaVal интерпретируется через exp как медиана длительности; округление половины вверх, минимум один день. Предсказание не обрезается концом сезона. Optuna, Ranker и подбор политик не применяются.", "",
        "Признаки известны при подключении: день года первого нулевого учёта и категория известной станции/округлённого места. Для анонимного Wisconsin география не выдумывается: вместо неё используются наблюдённые при входе яйца/имаго и день года. Сезонные погодные агрегаты, cum_dd, ошибочные Day/Week, будущая фенофаза, дата следующего визита и будущие учёты не являются признаками. Это НЕ демонстрация пользы погоды сверх календаря.", "",
        "Сильный календарный контроль: медиана дня года первой регистрации по строго более ранним годам, локальная при наличии не менее трёх прошлых событий, иначе общая. Для Wisconsin только общая: устойчивое географическое соответствие анонимных участков не доказано.", "",
        f"Начальная история: {config['warmup_years']}. Последовательные внешние годы: {config['outer_years']}. Каждый внешний год обучается только на прошлом; исходы текущего/будущих лет не участвуют в fit. Уже доступные/изученные годы не названы новым независимым тестом программы. Параметры фиксированы, внутренний подбор отсутствует.", "",
        f"## Наблюдаемость\n\nВсего рядов в реестре: {len(registry)}. Основной сервисный состав внешних лет: {len(outer)}. Положительный/неизвестный первый исходный визит не заменяется удобным поздним нулём. Причины включения/исключения сохранены для всех рядов.", "",
        "Два пола KBS объединяются внутри одного сбора. Любой измеренный >0 подтверждает положительную регистрацию; ноль требует полных измерений. У Wisconsin повторные/конфликтующие сообщения объединены лишь в статус присутствия: любой >0, полный ноль, иначе неизвестно. Их не усредняли и не объявляли отдельными событиями. Учтены все личиночные возраста, не только L1.", "",
        "Первый выпуск — следующий день после исходного нулевого учёта. Сетка продолжается до 31 октября независимо от будущих визитов, положительная запись прекращает сервис со следующего дня. Время08:00 условно местное: часовые пояса 46 американских станций не восстановлены, UTC-метки не выдуманы. Прогноз от входа статичен; последующие отрицательные визиты не используются для скрытого переобучения.", "",
        "`observed_followup`: нагрузка до положительной регистрации/последнего измеренного визита. `full_service_calendar`: при правом цензурировании сервис продолжает неизвестный хвост до31 октября, все дополнительные сообщения и тревожные дни посчитаны. Этот хвост не превращён в отрицательные биологические исходы. Нагрузка до первого события не равна числу сообщений за полный сезон с повторными вспышками.", "",
        "KBS содержит явные исходные нули, но нет журнала исправности ловушек. 284 из6550 дат нулевые по всем таксонам; основное reported принимает их как записи источника. Отдельный заранее заданный сценарий all_taxa_zero_unknown маскирует их неизвестными, без выбора более удачного результата. Нельзя считать разные сценарии независимыми датасетами.", "",
        "## Неопределённость и бюджеты", "",
        "Парный bootstrap по станциям/местам и отдельно по годам (2000 повторов). Для Wisconsin только годовые блоки: независимость анонимных полей неизвестна. Ресемплированный блок без событий не получает вымышленный recall0. Wilson — описательный и не заменяет кластерную оценку.", ""]
    for scope, blocks in ci.items():
        for key, item in blocks.items():
            interval = item["coverage_difference"]
            if interval["p025"] is not None:
                lines.append(f"- {scope}, {key}, {item['clusters']} блоков: разность покрытия AFT−календарь {100*interval['p025']:+.1f}…{100*interval['p975']:+.1f} п.п.")
    lines += ["", "Бюджеты:2 сообщения/30 полевых дней и50% тревожных дней. Политика P0 неизменна: порог0,5, active7, cooldown15. Равные допустимые бюджеты не означают одинакового числа сообщений. Годовые превышения ниже сохранены, пороги после результатов не меняются.", "",
        "| Год | Scope | Метод | Своевременно | Сообщения | Превышение сообщений / тревожных дней |", "|---|---|---|---:|---:|---|"]
    for r in metrics[~metrics.fold_id.eq("all_outer_years")].itertuples(index=False):
        lines.append(f"| {r.fold_id} | {r.evaluation_scope} | {r.model_code} | {r.timely_hits}/{r.warnable_events} | {r.messages} | {r.message_budget_exceeded} / {r.alarm_budget_exceeded} |")
    lines += ["", "## Воспроизведение", "",
        "Модели .cbm, календарные состояния, исходные наблюдения/идентификаторы, прогнозы отдельных станций и полей остаются локально. KBS требует отдельного разрешения для публикации core data; этот агрегированный отчёт не предоставляет такого разрешения. Исходники, лицензии и атрибуция сохранены в каталоге коллекции. Обучение не является валидацией опубликованной зимней погодной модели AHDB.", "", "```bash",
        f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_native_validation verify --output-dir {root}",
        f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_native_validation run --target {target} --zero-mode {zero_mode} --output-dir results/external_validation/{target}_REPRO_NEW",
        "```", "",
        "Проверка хеширует входы/выходы, заново обучает фиксированные модели на прошлых годах, пересчитывает таблицы/политику/bootstrap и отдельно сверяет прогнозы загруженных сохранённых моделей. Результаты старых циклов не меняются, реальные уведомления не отправляются.", ""]
    return "\n".join(lines)


def run(collection: Path, target: str, zero_mode: str, root: Path) -> dict:
    if root.exists():
        raise FileExistsError("Choose a new immutable run directory")
    audit = check_collection(collection)
    if audit["status"] != "passed":
        raise ValueError("Collection failed integrity check")
    paths = _input_paths(collection, target)
    hashes = {str(p.resolve()): digest_file(p) for p in paths}
    root.mkdir(parents=True)
    shutil.copyfile(CONTRACT, root/"contract.json")
    write_json(root/"input_audit.json", audit)
    for p in paths[:5]:
        dest = root/"source_snapshot"/p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, dest)
    tables = _compute(collection, target, zero_mode, root/"models")
    for name, frame in tables.items():
        frame.to_parquet(root/f"{name}.parquet", index=False)
        if name in {"metrics", "training_history", "gained_lost"}:
            frame.to_csv(root/f"{name}.csv", index=False)
    ci = paired_uncertainty(tables["per_series_metrics"], target)
    write_json(root/"uncertainty.json", ci)
    (root/"REPORT_RU.md").write_text(report_text(tables, target, zero_mode, ci, root), encoding="utf-8")
    if any(digest_file(Path(p)) != h for p, h in hashes.items()):
        raise RuntimeError("Inputs changed; run was not sealed")
    inventory = {str(p.relative_to(root)): digest_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root/"manifest.json", dict(status="complete", created_at=utc_now(), target=target, zero_mode=zero_mode,
        collection_dir=str(collection.resolve()), input_hashes=hashes, file_inventory=inventory,
        python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__))
    main = tables["metrics"]
    result = dict(status="complete", target=target, zero_mode=zero_mode, output_dir=str(root),
                  metrics=main[main.fold_id.eq("all_outer_years")].to_dict("records"))
    del tables
    gc.collect()
    return result


def verify(root: Path) -> dict:
    m = json.loads((root/"manifest.json").read_text())
    failures = [p for p, h in m["file_inventory"].items() if not (root/p).is_file() or digest_file(root/p) != h]
    inputs = [p for p, h in m["input_hashes"].items() if not Path(p).is_file() or digest_file(Path(p)) != h]
    checked, loaded = [], 0
    if not failures and not inputs:
        tables = _compute(Path(m["collection_dir"]), m["target"], m["zero_mode"])
        for name, frame in tables.items():
            try:
                pd.testing.assert_frame_equal(frame, pd.read_parquet(root/f"{name}.parquet"), check_exact=True)
                checked.append(name)
            except AssertionError:
                failures.append("recompute:"+name)
        if paired_uncertainty(tables["per_series_metrics"], m["target"]) != json.loads((root/"uncertainty.json").read_text()):
            failures.append("recompute:uncertainty")
        registry = tables["event_registry"]
        for file in sorted((root/"models").glob("*.cbm")):
            year = int(file.stem)
            rows = registry[registry.cohort_included & registry.season.eq(year)]
            dates = predict_aft_dates(CatBoostRegressor().load_model(str(file)), rows)
            expected = tables["predictions"]
            expected = expected[expected.model_code.eq(AFT) & expected.season.eq(year)].set_index("field_season").predicted_date
            values = rows.field_season.map(expected)
            if not dates.reset_index(drop=True).equals(values.reset_index(drop=True)):
                failures.append("saved_model:"+file.name)
            loaded += 1
    return dict(status="passed" if not failures and not inputs and m["status"] == "complete" else "failed",
                checked_files=len(m["file_inventory"]), checked_inputs=len(m["input_hashes"]),
                recomputed_tables=checked, checked_saved_models=loaded, mismatches=failures, input_mismatches=inputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--collection-dir", type=Path, default=COLLECTION)
    parser.add_argument("--target", choices=sorted(TARGET_SPECS))
    parser.add_argument("--zero-mode", choices=["reported", "all_taxa_zero_unknown"], default="reported")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run" and not args.target:
        parser.error("run requires --target")
    result = run(args.collection_dir, args.target, args.zero_mode, args.output_dir) if args.command == "run" else verify(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
