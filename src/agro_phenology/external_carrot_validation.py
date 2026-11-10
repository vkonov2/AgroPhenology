"""Retrospective Luke carrot-fly validation with a strict past-only calendar."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .early_warning_models import Policy, burden_metrics, event_metrics, simulate_policy
from .external_carrot_cohort import SOURCE_ID, build_decisions, build_registry
from .external_carrot_model import build_carrot_predictions
from .external_validation_collection import check as check_collection
from .external_validation_io import digest_file, utc_now, write_json
from .external_validation_weather import check as check_weather


CONTRACT = Path("docs/research/external_validation/carrot_spring_contract.json")
COLLECTION = Path("data/external_validation/20261110_external_v2")
WEATHER = Path("data/external_validation/20261110_carrot_spring_era5_v1")
OUTER_YEARS = list(range(2012, 2020))
CALENDAR = "calendar_expanding_median"
LUKE = "Luke_DD260_persistence7_lag5"
LAG7 = "Luke_DD260_persistence7_lag7"
MODELS = [CALENDAR, LUKE, LAG7]
POLICY = Policy(.5, active_days=7, cooldown_days=15, version="external_carrot_fixed_P0_v1")


def calendar_history(registry: pd.DataFrame) -> pd.DataFrame:
    """Every external year sees strictly earlier spring records; no fitting search."""
    rows = []
    for year in OUTER_YEARS:
        train = registry[registry.cohort_included & registry.warnable_first_event & registry.season.lt(year)]
        if train.season.nunique() < 3:
            raise ValueError("Calendar needs the three predeclared warmup years")
        doy = train.first_recorded_event_date.dt.dayofyear
        median_doy = int(np.floor(float(doy.median()) + .5))
        rows.append(dict(season=year, model_code=CALENDAR,
                         training_years=",".join(str(y) for y in sorted(train.season.unique())),
                         training_max_year=int(train.season.max()), training_events=len(train),
                         median_doy=median_doy, predicted_date=pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=median_doy-1)))
    return pd.DataFrame(rows)


def interval_distance(predicted, lower_zero, first_positive) -> float:
    if any(pd.isna(d) for d in (predicted, lower_zero, first_positive)):
        return np.nan
    return float(max((lower_zero + pd.Timedelta(days=1) - predicted).days,
                     (predicted-first_positive).days, 0))


def summarize_states(states: pd.DataFrame, registry: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, hits, per_series = [], [], []
    for model in MODELS:
        all_states = states[states.model_code.eq(model)].copy()
        if scope == "paired_computable_days":
            all_states["evaluation_scope_day"] &= all_states.common_weather_complete
            # Full sequential replay is preserved. Only count issued messages
            # on the common dates for this conditional diagnostic view.
            all_states["message_issued"] &= all_states.common_weather_complete
            all_states.loc[~all_states.common_weather_complete, "score"] = np.nan
        for year in [None, *OUTER_YEARS]:
            frame = all_states if year is None else all_states[all_states.season.eq(year)]
            events = registry if year is None else registry[registry.season.eq(year)]
            fold = "all_2012_2019" if year is None else str(year)
            event, event_hits = event_metrics(frame, events, model, fold, slice_name="spring_cohort", evaluation_scope=scope)
            burden = burden_metrics(frame, model, fold, slice_name="spring_cohort", evaluation_scope=scope)
            record = dict(event, **{k: v for k, v in burden.items() if k not in event})
            record.update(source_id=SOURCE_ID, organism="Psila rosae", outcome="first_registered_spring_adult_capture",
                          series=len(events), right_censored=int(events.first_recorded_event_date.isna().sum()),
                          spatial_groups=events.spatial_group.nunique(),
                          message_budget_exceeded=bool(burden["messages_per_30_field_days"] > 2),
                          alarm_budget_exceeded=bool(burden["active_alarm_fraction"] > .5))
            metrics.append(record)
            if year is None:
                hits.extend(event_hits)
        for row in registry.itertuples(index=False):
            sub = all_states[all_states.field_season.eq(row.field_season)]
            b = burden_metrics(sub, model, "series", evaluation_scope=scope)
            messages = sub[sub.message_issued]
            leads = (row.first_recorded_event_date - messages.issue_date).dt.days
            per_series.append(dict(b, field_season=row.field_season, season=row.season,
                                   spatial_group=row.spatial_group, eligible_event=bool(row.warnable_first_event),
                                   timely_hit=bool(row.warnable_first_event and leads.between(3, 10).any())))
    return pd.DataFrame(metrics), pd.DataFrame(hits), pd.DataFrame(per_series)


def _compute(collection: Path, weather: Path) -> dict[str, pd.DataFrame]:
    observations = pd.read_parquet(collection / f"normalized/{SOURCE_ID}/observations.parquet")
    registry = build_registry(observations)
    calendar = calendar_history(registry)
    outer = registry[registry.cohort_included & registry.season.isin(OUTER_YEARS)].copy()
    decisions = build_decisions(outer)
    mapping = pd.read_parquet(weather / "site_request_mapping.parquet")
    mapping = mapping[mapping.source_id.eq(SOURCE_ID)]
    manifest = json.loads((weather / "weather_manifest.json").read_text())
    blocks = []
    cache = {}
    for row in outer.itertuples(index=False):
        frame = decisions[decisions.field_season.eq(row.field_season)].copy()
        match = mapping[mapping.site_id.eq(row.site_id) & mapping.year.eq(row.season)]
        if len(match) != 1:
            raise ValueError("Expected exactly one weather mapping per monitoring series")
        rid = match.request_id.iloc[0]
        request = manifest["request_results"].get(rid, {})
        if request.get("status") != "complete":
            raise ValueError(f"Weather is unavailable for required request {rid}")
        if rid not in cache:
            cache[rid] = pd.read_parquet(weather / request["hourly_path"])
        weather_blocks = []
        for model, lag in [(LUKE, 5), (LAG7, 7)]:
            pred = build_carrot_predictions(cache[rid], frame.issue_date, row.season, lag_days=lag)
            pred.index = frame.index
            block = frame.copy()
            for name in pred.columns.difference(["issue_date", "issued_at"]):
                block[name] = pred[name]
            block["model_code"] = model
            block["request_id"] = rid
            weather_blocks.append(block)
        common = weather_blocks[0].score.notna() & weather_blocks[1].score.notna()
        for block in weather_blocks:
            block["common_weather_complete"] = common
        cal = weather_blocks[0].copy()
        cal["model_code"] = CALENDAR
        cal["predicted_event_date"] = calendar.loc[calendar.season.eq(row.season), "predicted_date"].iloc[0]
        cal["score"] = (cal.predicted_event_date - cal.issue_date).dt.days.between(3, 10).astype(float)
        cal["score_status"] = "computed"
        cal[["known_cumulative_dd", "persistence_daily_dd"]] = np.nan
        cal["feature_cutoff_date"] = pd.NaT
        blocks.extend([cal, *weather_blocks])
    predictions = pd.concat(blocks, ignore_index=True)
    states = []
    for model in MODELS:
        frame = predictions[predictions.model_code.eq(model)].reset_index(drop=True)
        state = simulate_policy(frame, frame.score, POLICY, evaluation_mask=frame.evaluation_field_day)
        state["source_id"], state["model_code"] = SOURCE_ID, model
        state["spatial_group"] = frame.spatial_group
        states.append(state)
    notifications = pd.concat(states, ignore_index=True)
    metrics, hits, series = [], [], []
    for scope in ["service_calendar", "paired_computable_days"]:
        m, h, s = summarize_states(notifications, outer, scope)
        metrics.append(m)
        hits.append(h)
        series.append(s)
    errors, miss_reasons = [], []
    for model in MODELS:
        for row in outer[outer.warnable_first_event].itertuples(index=False):
            pred = predictions[predictions.model_code.eq(model) & predictions.field_season.eq(row.field_season)].sort_values("issue_date")
            state = notifications[notifications.model_code.eq(model) & notifications.field_season.eq(row.field_season)]
            initial = pred.iloc[0]
            predicted = initial.predicted_event_date
            errors.append(dict(model_code=model, field_season=row.field_season, season=row.season,
                               spatial_group=row.spatial_group, issued_at=initial.issued_at, predicted_date=predicted,
                               first_registration=row.first_recorded_event_date, last_sampled_zero=row.last_complete_zero_date,
                               registration_error_days=(predicted-row.first_recorded_event_date).days if pd.notna(predicted) else np.nan,
                               distance_to_observation_interval_days=interval_distance(predicted, row.last_complete_zero_date, row.first_recorded_event_date)))
            window = state[state.days_to_first_recorded_event.between(3, 10) & state.service_active]
            timely = bool(window.message_issued.any())
            messages = state[state.message_issued]
            message_leads = (row.first_recorded_event_date - messages.issue_date).dt.days
            reason = ("timely_hit" if timely else "no_computable_actionable_day" if not window.score.notna().any()
                      else "policy_suppressed_in_actionable_window" if window.suppressed_repeat.any()
                      else "score_below_threshold_in_actionable_window")
            miss_reasons.append(dict(model_code=model, field_season=row.field_season, season=row.season,
                                     spatial_group=row.spatial_group, timely_hit=timely, reason=reason,
                                     computable_actionable_days=int(window.score.notna().sum()),
                                     above_threshold_actionable_days=int(window.score.ge(.5).sum()),
                                     suppressed_actionable_days=int(window.suppressed_repeat.sum()),
                                     too_early_messages=int(message_leads.gt(10).sum()), late_messages=int(message_leads.lt(3).sum()),
                                     initial_prediction_before_entry=bool(pd.notna(predicted) and predicted < row.entry_date+pd.Timedelta(days=1))))
    return dict(event_registry=registry, calendar_history=calendar, decisions=decisions, predictions=predictions,
                notifications=notifications, metrics=pd.concat(metrics, ignore_index=True),
                event_hits=pd.concat(hits, ignore_index=True), per_series_metrics=pd.concat(series, ignore_index=True),
                initial_date_errors=pd.DataFrame(errors), miss_reasons=pd.DataFrame(miss_reasons))


def uncertainty(series: pd.DataFrame) -> dict:
    data = series[series.evaluation_scope.eq("service_calendar")]
    baseline = data[data.model_code.eq(CALENDAR)].set_index("field_season")
    result = {}
    for model in [LUKE, LAG7]:
        other = data[data.model_code.eq(model)].set_index("field_season").reindex(baseline.index)
        comparisons = {}
        for group_col in ["spatial_group", "season"]:
            groups = baseline[group_col].unique()
            rng = np.random.default_rng(20261110)
            samples = rng.integers(0, len(groups), size=(2000, len(groups)))
            totals = []
            for group in groups:
                mask = baseline[group_col].eq(group)
                totals.append([baseline.loc[mask, "eligible_event"].sum(), baseline.loc[mask, "field_days"].sum(),
                               other.loc[mask, "timely_hit"].sum()-baseline.loc[mask, "timely_hit"].sum(),
                               other.loc[mask, "messages"].sum()-baseline.loc[mask, "messages"].sum(),
                               other.loc[mask, "active_alarm_days"].sum()-baseline.loc[mask, "active_alarm_days"].sum()])
            totals = np.array(totals, dtype=float)
            draws = totals[samples].sum(axis=1)
            # A resample can contain only right-censored spatial groups.
            # Its coverage denominator is unknown/zero, never zero recall.
            valid_events = draws[:, 0] > 0
            stats = {"coverage_difference": draws[valid_events, 2]/draws[valid_events, 0],
                     "messages_per_30_days_difference": 30*draws[:, 3]/draws[:, 1],
                     "alarm_fraction_difference": draws[:, 4]/draws[:, 1]}
            comparisons[group_col] = {"clusters": len(groups), "replicates": 2000,
                                     **{key: {"valid_replicates": len(value),
                                              "p025": float(np.quantile(value, .025)) if len(value) else None,
                                              "p975": float(np.quantile(value, .975)) if len(value) else None} for key, value in stats.items()}}
        result[model+"_minus_calendar"] = comparisons
    return result


def _input_paths(collection: Path, weather: Path) -> list[Path]:
    paths = [CONTRACT, Path(__file__), Path(__file__).with_name("external_carrot_model.py"),
             Path(__file__).with_name("external_carrot_cohort.py"), Path(__file__).with_name("early_warning_models.py"),
             collection/"manifest.json", collection/f"normalized/{SOURCE_ID}/observations.parquet",
             collection/f"raw/{SOURCE_ID}/metadata.json", weather/"weather_manifest.json", weather/"weather_plan.json",
             weather/"site_request_mapping.parquet"]
    manifest = json.loads((weather/"weather_manifest.json").read_text())
    paths.extend(weather/v["hourly_path"] for v in manifest["request_results"].values() if v.get("status") == "complete")
    return paths


def _report(tables: dict, root: Path, ci: dict) -> str:
    all_registry = tables["event_registry"]
    registry = all_registry[all_registry.cohort_included & all_registry.season.isin(OUTER_YEARS)]
    metrics = tables["metrics"]
    overall = metrics[metrics.fold_id.eq("all_2012_2019") & metrics.evaluation_scope.eq("service_calendar")]
    lines = ["# Морковная муха: внешняя проверка опубликованной температурной модели", "",
             "10 ноября 2026. Один датасет, фиксированные параметры, без Optuna и нового перебора политик. Все показатели рассчитаны по реальным ловушечным наблюдениям. Сырые данные и сведения об отдельных полях остаются локально.", "",
             "## Результат", "", "| Метод | Своевременные события | Покрытие | Сообщения | Тревожные дни | Вычислимость | Сообщений / 30 дней |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in overall.itertuples(index=False):
        lines.append(f"| {r.model_code} | {r.timely_hits}/{r.events_with_warning_opportunity} | {r.timely_recall:.1%} | {r.messages} | {r.active_alarm_days}/{r.field_days} | {r.computable_fraction:.1%} | {r.messages_per_30_field_days:.2f} |")
    a = overall[overall.model_code.eq(CALENDAR)].iloc[0]
    b = overall[overall.model_code.eq(LUKE)].iloc[0]
    lines += ["", f"Основное сравнение: температурная модель {int(b.timely_hits)}/{int(b.events_with_warning_opportunity)}, календарь {int(a.timely_hits)}/{int(a.events_with_warning_opportunity)}. Разность покрытия {(b.timely_recall-a.timely_recall):+.1%}; разность сообщений {int(b.messages-a.messages):+d}, тревожных дней {int(b.active_alarm_days-a.active_alarm_days):+d}. Это ретроспективный результат на уже доступном наборе, не новый независимый тест всей программы исследований.", "",
              "## Что именно предсказывается", "",
              "[Luke / IPM Decisions](https://www.ipmdecisions.net/media/ev1e2wz2/ipm_factsheet-carrot-rust-fly_v0001_print.pdf) описывает начало весеннего лёта морковной мухи: 260 °C·дней выше 5 °C с 1 марта. Реализована адаптация по средней суточной температуре воздуха: max(mean(T за полные местные сутки)−5,0). Верхний порог не придуман. Тождество закрытому эталонному коду Luke не заявляется. Источник сообщает разработку в Финляндии и предполагаемую применимость в Дании; переносимость здесь проверяется, а не считается доказанной.", "",
              "Измеряемый исход — первая положительная регистрация взрослой мухи в ловушке, прокси начала наблюдаемого лёта. Это не истинный момент выхода из куколки, не яйцекладка и не повреждение корнеплода. Культура в каждой строке источника не указана; она не выдумывается. Нормировка дробных уловов по ловушкам/дням не установлена, поэтому используются только явный ноль и значение >0.", "",
              "Score равен 1, когда предсказанная дата находится в issue+3…issue+10. Это индикатор временного окна, не калиброванная вероятность заражения. Сообщение считается своевременным, если оно отправлено за 3–10 дней до первой регистрации. Суточная accuracy и доказанная доля ложных тревог не рассчитываются: отсутствие визита не равно отсутствию вредителя.", "",
              "## Данные и отбор до оценки качества", "",
              "Источник: [Niels Holst (2022), Zenodo 6351794](https://zenodo.org/records/6351794), Aarhus University; наблюдения собраны SEGES Innovation, Denmark, CC BY 4.0. Проверка выполнена отдельно от совки, DaKS и VAAD. Нормализованные значения/даты/FieldId сверены с исходным TXT; пропусков и дубликатов в рассматриваемом источнике не найдено.", "",
              f"Всего {len(all_registry)} рядов морковной мухи. Первый осмотр должен быть не позднее 31 мая и иметь измеренный ноль. Нельзя искать удобный последующий ноль после положительного входа. Отобрано 65 рядов: 59 первых весенних регистраций и 6 правых цензурирований. Остальные 214 начинаются позднее и сохранены с причиной исключения. Конец весеннего периода заранее фиксирован 30 июня; летние поколения не выданы за первое весеннее.", "",
              f"2009–2011 — начальная история для календаря: пять событий. Основное сравнение 2012–2019: {len(registry)} рядов, {int(registry.warnable_first_event.sum())} пригодных событий и {int(registry.first_recorded_event_date.isna().sum())} правых цензурирований. Каждому году календарь предсказывает медианный день года регистрации по строго более ранним годам этой же весенней когорты, с округлением половины вверх. История календаря сохранена; внешние исходы текущего и будущих лет не доступны его обучению. Параметры Luke, порог сообщений и политика не обучаются.", "",
              "Физическая независимость ограничена: в основном сравнении пять пространственных групп по округлённым исходным координатам, 29 погодных ячеек-лет. Крупнейшая группа даёт 37 рядов и 32 события. Это не 54 независимых участка; разные FieldId одного района и года могут иметь одинаковую погоду и общий процесс наблюдения.", "",
              "## Время, погода и цензурирование", "",
              "Используется отдельно сохранённая ERA5 (Copernicus / ECMWF) через [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api). Выпуск ежедневно в 08:00 Europe/Copenhagen со следующего дня после первого нулевого осмотра до 30 июня, независимо от даты будущего визита. Запись положительного улова становится доступна на следующий день и прекращает дальнейшие сообщения. Для оценки наблюдаемого сервиса последний измеренный осмотр ограничивает знаменатель; дни после него неизвестны, а не отрицательны. В журнале полная фиксированная сетка сохранена.", "",
              "В основном варианте доступны только полные сутки до issue−5. Пятисуточная задержка — предположение, не восстановленный архив публикаций ERA5. Все часы после cutoff исключаются до обработки температур. Недавний недоступный промежуток и будущее оцениваются одним средним суточным темпом суммы тепла по семи последним последовательным полностью доступным дням. Фактически состоявшаяся будущая погода не используется. Заранее предусмотрена чувствительность к ещё двум дням задержки (lag7); её результат не использован для выбора лучшего варианта.", "",
              "Недостающий час/день вызывает отказ от прогноза, а не заполнение. Нулевой прогнозный темп вызывает отказ, если тепловой порог ещё не достигнут. При достигнутом пороге возвращаются прошлая дата и score0. Полный нулевой выборочный осмотр не доказывает отсутствия мух во всём поле; расстояние до интервала осмотров — вспомогательный ориентир, не точность даты биологического появления.", "",
              "## По годам", "", "| Год | Метод | Своевременно | Сообщения | Тревожные дни |", "|---|---|---:|---:|---:|"]
    annual = metrics[metrics.evaluation_scope.eq("service_calendar") & ~metrics.fold_id.eq("all_2012_2019")]
    for r in annual.sort_values(["fold_id", "model_code"]).itertuples(index=False):
        lines.append(f"| {r.fold_id} | {r.model_code} | {r.timely_hits}/{r.events_with_warning_opportunity} | {r.messages} | {r.active_alarm_days}/{r.field_days} |")
    lines += ["", "## Цена предупреждений и неопределённость", "",
              "Общая неизменная политика P0: порог0,5, cooldown15 дней, тревожное состояние7 дней после отправки. У каждой системы отдельная история. Считаются все сообщения, а не только полезные; слишком раннее сообщение может подавить последующее. Бюджеты прежние:2 сообщения/30 полевых дней и50% тревожных дней. Равные допустимые бюджеты не означают равную фактическую нагрузку. Порог после внешних результатов не меняется. Короткая экспозиция до первой регистрации не характеризует нагрузку полного дачного сезона.", "", "| Метод | Превышен бюджет сообщений | Превышен бюджет тревожных дней |", "|---|---|---|"]
    for r in overall.itertuples(index=False):
        lines.append(f"| {r.model_code} | {'да' if r.message_budget_exceeded else 'нет'} | {'да' if r.alarm_budget_exceeded else 'нет'} |")
    comp = ci[LUKE+"_minus_calendar"]["spatial_group"]
    bounds = comp["coverage_difference"]
    lines += ["", f"Парный кластерный bootstrap (2 000 повторов, {comp['clusters']} пространственных групп) даёт описательный интервал разности покрытия Luke−календарь: {bounds['p025']:+.1%}…{bounds['p975']:+.1%}. Пять групп и доминирование одной делают оценку нестабильной. Дополнительно сохранён bootstrap по восьми годам. Повторные строки не используются как независимые наблюдения; Wilson в машинной таблице лишь описательный и не заменяет кластерную оценку.", "",
              "Сохранены полный сервис и условная общая маска вычислимых дат. Общая маска ограничивает только подсчёт: полная последовательная история не пересобирается и не сжимает время/cooldown. Сообщения на несопоставимых датах не считаются в условном покрытии; состояние от них сохраняется. Правое цензурирование не вырезает ряды из нагрузки до последнего наблюдения.", "",
              "## Ограничения и дальнейший приоритет", "",
              "Сопоставляется регистрация в ловушке с моделью начала лёта: несовпадение может отражать и переносимость модели, и интервал осмотров/постановку ловушек. Ведомостей об укрытии растений, обработках и точных датах установки нет. Округлённые координаты и ERA5 не описывают локальную температуру почвы. Погодные задержки условны; новые районы и будущие годы ещё не проверены. Никакого подбора для победы над календарём не выполнялось.", "",
              "Следующий приоритет — независимые районы с ранними нулевыми ловушечными учётами и точным временем установки/проверки ловушек. До выбора новой модели проверить разбор промахов и причин слишком ранних решений, а не перебирать пороги на этих пяти группах.", "",
              "## Артефакты и воспроизведение", "",
              "Локально сохранены все причины отбора, ежедневные оценки, последовательный журнал, история календаря, события, нагрузка по годам, ошибки дат и причины промахов. `manifest.json` фиксирует входы, код, контракт и результаты; `verify` пересчитывает все таблицы. `contract.json` и `source_snapshot/` фиксируют выбранные настройки. Никакие сообщения не отправлялись.", "", "```bash",
              f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_carrot_validation verify --output-dir {root}",
              "PYTHONPATH=src .venv/bin/python -m agro_phenology.external_carrot_validation run --output-dir results/external_validation/zenodo_6351794/carrot_spring_REPRO_NEW",
              "PYTHONPATH=src .venv/bin/python -m pytest tests/test_external_carrot_*.py -q", "```", ""]
    return "\n".join(lines)


def run(collection: Path, weather: Path, root: Path) -> dict:
    if root.exists():
        raise FileExistsError("Run directories are immutable; choose a new directory")
    audit = {"collection": check_collection(collection), "weather": check_weather(weather)}
    if any(x["status"] != "passed" for x in audit.values()):
        raise ValueError(f"Inputs failed integrity check: {audit}")
    paths = _input_paths(collection, weather)
    hashes = {str(p.resolve()): digest_file(p) for p in paths}
    root.mkdir(parents=True)
    shutil.copyfile(CONTRACT, root/"contract.json")
    write_json(root/"input_audit.json", audit)
    for p in paths[:5]:
        dest = root/"source_snapshot"/p.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, dest)
    tables = _compute(collection, weather)
    for name, frame in tables.items():
        frame.to_parquet(root/f"{name}.parquet", index=False)
        if name in {"metrics", "calendar_history", "per_series_metrics", "initial_date_errors", "miss_reasons", "event_hits"}:
            frame.to_csv(root/f"{name}.csv", index=False)
    ci = uncertainty(tables["per_series_metrics"])
    write_json(root/"uncertainty.json", ci)
    (root/"REPORT_RU.md").write_text(_report(tables, root, ci), encoding="utf-8")
    if any(digest_file(Path(p)) != sha for p, sha in hashes.items()):
        raise RuntimeError("Input changed while evaluating; run is not sealed")
    inventory = {str(p.relative_to(root)): digest_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root/"manifest.json", dict(status="complete", created_at=utc_now(), dataset=SOURCE_ID,
               collection_dir=str(collection.resolve()), weather_dir=str(weather.resolve()), input_hashes=hashes,
               file_inventory=inventory, python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__))
    main = tables["metrics"]
    main = main[main.fold_id.eq("all_2012_2019") & main.evaluation_scope.eq("service_calendar")]
    return dict(status="complete", output_dir=str(root), metrics=main[["model_code", "timely_hits", "events_with_warning_opportunity", "messages", "active_alarm_days", "field_days", "computable_fraction"]].to_dict("records"))


def verify(root: Path) -> dict:
    manifest = json.loads((root/"manifest.json").read_text())
    failures = [p for p, h in manifest["file_inventory"].items() if not (root/p).is_file() or digest_file(root/p) != h]
    inputs = [p for p, h in manifest["input_hashes"].items() if not Path(p).is_file() or digest_file(Path(p)) != h]
    checked = []
    if not failures and not inputs:
        tables = _compute(Path(manifest["collection_dir"]), Path(manifest["weather_dir"]))
        for name, frame in tables.items():
            try:
                pd.testing.assert_frame_equal(frame, pd.read_parquet(root/f"{name}.parquet"), check_exact=True)
                checked.append(name)
            except AssertionError:
                failures.append("recompute:"+name)
        expected_ci = uncertainty(tables["per_series_metrics"])
        if expected_ci != json.loads((root/"uncertainty.json").read_text()):
            failures.append("recompute:uncertainty")
    return dict(status="passed" if not failures and not inputs and manifest["status"] == "complete" else "failed",
                checked_files=len(manifest["file_inventory"]), checked_inputs=len(manifest["input_hashes"]),
                recomputed_tables=checked, mismatches=failures, input_mismatches=inputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--collection-dir", type=Path, default=COLLECTION)
    parser.add_argument("--weather-dir", type=Path, default=WEATHER)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.collection_dir, args.weather_dir, args.output_dir) if args.command == "run" else verify(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
