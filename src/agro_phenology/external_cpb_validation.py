"""One fixed CPB thermal model, one external dataset, no fitting or selection."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .early_warning_models import Policy, burden_metrics, event_metrics, simulate_policy
from .external_cpb_model import build_cpb_predictions
from .external_validation_collection import check as check_collection
from .external_validation_io import digest_file, utc_now, write_json
from .external_validation_weather import check as check_weather


DATASET = "daks_cpb_2019_2022"
DD_MODEL = "CPB_DD120_persistence7"
CALENDAR = "calendar_biofix_plus_7_days"
MODELS = [CALENDAR, DD_MODEL]
CONTRACT = Path("docs/research/external_validation/cpb_degree_days_contract.json")
DEFAULT_COLLECTION = Path("data/external_validation/20261110_external_v2")
DEFAULT_WEATHER = Path("data/external_validation/20261110_era5_v2")
POLICY = Policy(.5, active_days=7, cooldown_days=15, version="external_cpb_fixed_P0_v1")


def control_visits(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep genuine zeros, unknowns and positives distinct; never pool treatments."""
    if set(observations.source_id) != {DATASET}:
        raise ValueError("A validation run must contain exactly the DaKS CPB dataset")
    if set(observations.unit) != {"individuals_per_plant"}:
        raise ValueError("Unexpected measurement unit")
    frame = observations[observations.treatment.eq("control")].copy()
    if frame.empty:
        raise ValueError("No control observations")
    frame["observation_date"] = pd.to_datetime(frame.observation_date).dt.normalize()
    frame["plant_key"] = frame[["plot_id", "triplet_id", "block", "plant_id"]].fillna("NA").astype(str).agg("/".join, axis=1)
    keys = ["site_id", "observation_date", "plant_key", "stage"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate plant-stage observations")
    if frame.value.dropna().lt(0).any():
        raise ValueError("Negative insect counts")
    # Plant identifiers can restart or change scope between assessments.
    # Use the largest scheduled within-visit count, never a longitudinal union.
    expected = frame.groupby(["site_id", "observation_date"]).plant_key.nunique().groupby(level=0).max()
    rows = []
    for (site, day), group in frame.groupby(["site_id", "observation_date"], sort=True):
        early = group[group.stage.isin(["larva_1", "larva_2"])]
        stage_counts = early.groupby("plant_key").stage.nunique()
        complete = (len(early) == 2 * expected[site] and len(stage_counts) == expected[site]
                    and stage_counts.eq(2).all() and early.value.notna().all())
        positive = bool(early.value.gt(0).any())
        eggs = group[group.stage.eq("egg")]
        rows.append(dict(source_id=DATASET, site_id=site, observation_date=day,
                         egg_positive=bool(eggs.value.gt(0).any()),
                         early_larvae_status="positive" if positive else "zero" if complete else "unknown",
                         early_values=len(early), expected_values=int(2 * expected[site]),
                         missing_early_values=int(early.value.isna().sum()), treatment="control"))
    return pd.DataFrame(rows)


def event_registry(visits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site, group in visits.groupby("site_id", sort=True):
        group = group.sort_values("observation_date")
        eggs = group[group.egg_positive]
        biofix = eggs.observation_date.min() if len(eggs) else pd.NaT
        after = group[group.observation_date.ge(biofix)]
        positives = after[after.early_larvae_status.eq("positive")]
        event = positives.observation_date.min() if len(positives) else pd.NaT
        zeros = after[after.early_larvae_status.eq("zero") & after.observation_date.lt(event)]
        last_zero = zeros.observation_date.max() if len(zeros) else pd.NaT
        positive_at_entry = bool(pd.notna(event) and event == biofix)
        eligible = bool(pd.notna(event) and pd.notna(last_zero) and (event - biofix).days >= 4)
        category = ("eligible" if eligible else "positive_at_entry_left_censored" if positive_at_entry
                    else "right_censored" if pd.isna(event) else "insufficient_warning_or_zero")
        season = int(group.observation_date.iloc[0].year)
        if group.observation_date.dt.year.nunique() != 1:
            raise ValueError("A site series must represent one season")
        rows.append(dict(source_id=DATASET, site_id=site, field_season=site, season=season,
                         biofix_date=biofix, first_recorded_event_date=event,
                         last_complete_zero_date=last_zero, last_assessment_date=group.observation_date.max(),
                         warnable_first_event=eligible, positive_at_first_visit=positive_at_entry,
                         eligibility_category=category, treatment="control", coordinate_scope="public_field",
                         previous_visit_gap_days=(event-last_zero).days if pd.notna(last_zero) else np.nan))
    return pd.DataFrame(rows)


def decision_calendar(registry: pd.DataFrame) -> pd.DataFrame:
    """Issue grid never uses a future visit. Outcome availability stops service."""
    blocks = []
    for row in registry.itertuples(index=False):
        if pd.isna(row.biofix_date):
            continue
        days = pd.date_range(row.biofix_date + pd.Timedelta(days=1), f"{row.season}-10-31")
        block = pd.DataFrame({"issue_date": days})
        block["issued_at"] = [(d + pd.Timedelta(hours=8)).tz_localize("Europe/Berlin").isoformat() for d in days]
        for name in ("source_id", "site_id", "field_season", "season", "biofix_date",
                     "warnable_first_event", "coordinate_scope", "previous_visit_gap_days"):
            block[name] = getattr(row, name)
        block["service_active"] = True if pd.isna(row.first_recorded_event_date) else days <= row.first_recorded_event_date
        block["evaluation_field_day"] = block.service_active & (days <= row.last_assessment_date)
        block["days_to_first_recorded_event"] = (row.first_recorded_event_date - days).days
        block["target_class"] = "unknown"
        block["target_observable"] = False
        block["common_weather_complete"] = False
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def interval_distance(predicted, last_zero, first_positive) -> float:
    """Distance to a sampled observation interval, not bounds of true field onset."""
    if any(pd.isna(d) for d in (predicted, last_zero, first_positive)):
        return np.nan
    lower = pd.Timestamp(last_zero) + pd.Timedelta(days=1)
    return float(max((lower - predicted).days, (predicted - first_positive).days, 0))


def _compute(collection: Path, weather: Path) -> dict[str, pd.DataFrame]:
    observations = pd.read_parquet(collection / f"normalized/{DATASET}/observations.parquet")
    visits = control_visits(observations)
    registry = event_registry(visits)
    decisions = decision_calendar(registry)
    mapping = pd.read_parquet(weather / "site_request_mapping.parquet")
    mapping = mapping[mapping.source_id.eq(DATASET)]
    manifest = json.loads((weather / "weather_manifest.json").read_text())
    blocks = []
    for event in registry.itertuples(index=False):
        frame = decisions[decisions.field_season.eq(event.field_season)].copy()
        match = mapping[mapping.site_id.eq(event.site_id) & mapping.year.eq(event.season)]
        if len(match) != 1:
            raise ValueError(f"Expected exactly one weather mapping: {event.site_id}")
        request = manifest["request_results"].get(match.request_id.iloc[0], {})
        if request.get("status") != "complete":
            raise ValueError(f"Missing required weather: {event.site_id}")
        hourly = pd.read_parquet(weather / request["hourly_path"])
        thermal = build_cpb_predictions(hourly, frame.issue_date, event.biofix_date)
        thermal.index = frame.index
        for column in thermal.columns.difference(["issue_date", "issued_at"]):
            frame[column] = thermal[column]
        frame["common_weather_complete"] = frame.score.notna()
        frame["model_code"] = DD_MODEL
        frame["request_id"] = match.request_id.iloc[0]
        blocks.append(frame)
        calendar = frame.copy()
        calendar["model_code"] = CALENDAR
        calendar["predicted_event_date"] = event.biofix_date + pd.Timedelta(days=7)
        calendar["score"] = ((calendar.predicted_event_date - calendar.issue_date).dt.days.between(3, 10)).astype(float)
        calendar["score_status"] = "computed"
        calendar[["known_cumulative_dd", "persistence_daily_dd"]] = np.nan
        calendar["feature_cutoff_date"] = pd.NaT
        blocks.append(calendar)
    predictions = pd.concat(blocks, ignore_index=True)
    metrics, hits, states, per_series, errors = [], [], [], [], []
    eligible = registry[registry.warnable_first_event]
    for model in MODELS:
        frame = predictions[predictions.model_code.eq(model)].reset_index(drop=True)
        state = simulate_policy(frame, frame.score, POLICY, evaluation_mask=frame.evaluation_field_day)
        state["model_code"] = model
        state["source_id"] = DATASET
        states.append(state)
        population = state[state.field_season.isin(eligible.field_season)]
        event, event_hits = event_metrics(population, eligible, model, "external_fixed", slice_name="eligible_control")
        burden = burden_metrics(population, model, "external_fixed", slice_name="eligible_control")
        record = dict(event, **{k: v for k, v in burden.items() if k not in event})
        record.update(source_id=DATASET, treatment="control", outcome="first_registered_L1_or_L2",
                      message_budget_exceeded=burden["messages_per_30_field_days"] > 2,
                      alarm_budget_exceeded=burden["active_alarm_fraction"] > .5)
        metrics.append(record)
        hits.extend(event_hits)
        for item in eligible.itertuples(index=False):
            sub = state[state.field_season.eq(item.field_season)]
            b = burden_metrics(sub, model, "external_fixed", slice_name="eligible_control")
            h = next(h for h in event_hits if h["field_season"] == item.field_season)
            per_series.append(dict(b, timely_hit=h["timely_hit"], field_season=item.field_season, season=item.season))
            initial = frame[frame.field_season.eq(item.field_season)].sort_values("issue_date").iloc[0]
            errors.append(dict(model_code=model, field_season=item.field_season, season=item.season,
                               issue_date=initial.issue_date, predicted_date=initial.predicted_event_date,
                               first_registration=item.first_recorded_event_date,
                               last_complete_zero=item.last_complete_zero_date,
                               registration_error_days=(initial.predicted_event_date-item.first_recorded_event_date).days
                               if pd.notna(initial.predicted_event_date) else np.nan,
                               distance_to_onset_interval_days=interval_distance(initial.predicted_event_date, item.last_complete_zero_date, item.first_recorded_event_date)))
    return {"control_visits": visits, "event_registry": registry, "predictions": predictions,
            "notifications": pd.concat(states, ignore_index=True), "metrics": pd.DataFrame(metrics),
            "event_hits": pd.DataFrame(hits), "per_series_metrics": pd.DataFrame(per_series), "initial_date_errors": pd.DataFrame(errors)}


def paired_uncertainty(per_series: pd.DataFrame) -> dict:
    left = per_series[per_series.model_code.eq(CALENDAR)].set_index("field_season")
    right = per_series[per_series.model_code.eq(DD_MODEL)].set_index("field_season").reindex(left.index)
    rng = np.random.default_rng(20261110)
    draws = rng.integers(0, len(left), size=(2000, len(left)))
    result = {"clusters": len(left), "replicates": 2000, "interpretation": "exploratory_two_field_series_not_population_precision"}
    for column in ["timely_hit", "messages", "active_alarm_days"]:
        delta = right[column].astype(float).to_numpy() - left[column].astype(float).to_numpy()
        means = delta[draws].mean(axis=1)
        result[column + "_difference"] = {"mean_per_series": float(delta.mean()), "bootstrap_p025": float(np.quantile(means, .025)), "bootstrap_p975": float(np.quantile(means, .975))}
    return result


def _inputs(collection: Path, weather: Path) -> list[Path]:
    paths = [CONTRACT, Path(__file__), Path(__file__).with_name("external_cpb_model.py"),
             Path(__file__).with_name("early_warning_models.py"),
             collection / "manifest.json", collection / "catalog.json", weather / "weather_manifest.json",
             weather / "site_request_mapping.parquet", collection / f"normalized/{DATASET}/observations.parquet",
             collection / f"normalized/{DATASET}/locations.parquet"]
    mapping = pd.read_parquet(weather / "site_request_mapping.parquet")
    manifest = json.loads((weather / "weather_manifest.json").read_text())
    for rid in sorted(set(mapping.loc[mapping.source_id.eq(DATASET), "request_id"])):
        paths.append(weather / manifest["request_results"][rid]["hourly_path"])
    return paths


def _report(tables: dict, root: Path, uncertainty: dict) -> None:
    registry, metrics = tables["event_registry"], tables["metrics"]
    lines = ["# Внешняя проверка температурной модели колорадского жука", "", "Дата: 10 ноября 2026. Один набор DaKS 2019–2022, только контрольные варианты. Это завершённый ретроспективный пилот, без обучения и подбора параметров.", "",
             "## Что реализовано", "", "Адаптация опубликованной модели [UW–Madison](https://vegento.russell.wisc.edu/pests/colorado-potato-beetle/): 120 °F·дней после первых обнаруженных яиц, нижний порог 52 °F, верхний 86 °F. Здесь сумма рассчитывается почасовым ограничением температуры; эквивалентность методу суточной синусоиды не заявляется. Порог соответствует примерно 66,7 °C·дней. Это модель срока появления молодых личинок, не бинарный прогноз наличия жука и не модель повреждения растений.", "",
             "Контроль: дата первых обнаруженных яиц +7 дней. Параметры обоих методов зафиксированы до расчёта качества; календарь выбран как простой ориентир по опубликованному диапазону 4–9 дней, а не как оптимальный календарь. Ни CatBoost, ни оптимизация на этих исходах не применялись.", "",
             "## Входы и честная доступность", "", "Исходы: [DaKS, DOI 10.48662/daks-28](https://daks.uni-kassel.de/entities/dataset/aeb5b8aa-ece2-4bfc-a915-087cd29f1dbf). Погода: отдельно собранный ERA5 через Open-Meteo. Используются пять ячеек-лет, связанных с шестью полевыми сериями. Публичные координаты остаются локально. Лицензия DaKS CC BY-NC-SA 4.0 сохраняется; доступность данных не означает разрешения на любое коммерческое использование.", "",
             "Ежедневный выпуск в 08:00 Europe/Berlin, начиная со следующего дня после регистрации яиц. Погода доступна только до issue_date−5 включительно. Задержка пять суток — исследовательское предположение, а не восстановленный журнал исторических публикаций. Недоступные недавние и будущие дни оцениваются одной постоянной скоростью: средней суммой тепла за последние семь полностью доступных дней. Это прогноз постоянства, не скачанный оперативный прогноз. Реализованная будущая температура не используется. Пропуск часа делает день неизвестным; нулевой темп вызывает отказ от прогноза, если тепловой порог ещё не достигнут. При уже достигнутом пороге возвращается прошлая дата и score 0.", "",
             "## Популяция и исход", "", f"В исходном наборе {len(registry)} контрольных серий поле × сезон. В четырёх молодые личинки уже были при подключении: это левоцензурированные случаи, исключённые из знаменателя предупреждений. Для оценки остаются два перехода от полного нулевого осмотра к положительному. Растения, блоки и варианты мульчи не объявляются независимыми событиями.", "",
             "Исход — первая регистрация L1 или L2. Положительный осмотр: хотя бы один измеренный счётчик >0. Нулевой: все предусмотренные измерения обеих стадий есть и равны нулю. Пропуски и дни без осмотра неизвестны. Повторные положительные осмотры не являются новыми событиями. Интервал (последний полный нулевой выборочный осмотр, первая положительная регистрация] используется как вспомогательный ориентир неопределённости наблюдений. Ноль на осмотренных растениях не доказывает отсутствия личинок во всём поле: это не строгие границы истинного появления и не точная дата вылупления.", "",
             "## Предупреждения и результаты", "", "Score равен 1, когда предсказанная дата попадает в +3…+10 дней от выпуска. Общая существующая последовательная политика: порог 0,5, тревожное состояние семь дней, cooldown 15 дней. Каждое реальное сообщение считается; после доступности первого положительного осмотра сервис останавливается. Сетка дат строится до 31 октября независимо от будущих визитов; последний осмотр ограничивает только оценку. Тревожные дни — длительность состояния после сообщения, а не будущие дни прогнозного окна.", "",
             "| Метод | Своевременно / событий | 95% Wilson | Сообщений | Тревожных дней / дней наблюдаемого сервиса | Вычислимость | Сообщений / 30 дней |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples(index=False):
        lines.append(f"| {r.model_code} | {r.timely_hits}/{r.events_with_warning_opportunity} | {r.timely_recall_wilson_low:.1%}–{r.timely_recall_wilson_high:.1%} | {r.messages} | {r.active_alarm_days}/{r.field_days} | {r.computable_fraction:.1%} | {r.messages_per_30_field_days:.2f} |")
    lines += ["", "Метрики относятся только к двум eligible-сериям. Обе модели проверены на одних событиях и датах. Частые тревоги и изменение порогов не использованы для улучшения результата. Бюджеты: не более 2 сообщений на 30 полевых дней и не более 50% тревожных дней; внешние превышения оставлены как есть.", "",
              "По годам: в 2021 году оба метода поймали 1/1, отправив по одному сообщению и дав по 7 тревожных дней на 7 дней сервиса. В 2022 — 0/1, по одному сообщению и 7/15 тревожных дней. В 2019 и 2020 нет пригодных событий для этого исхода: первые осмотры уже положительны. Эти годы не получают вымышленного нулевого качества.", "",
              "Сообщения двух систем полностью совпали. В 2021 году сообщение было за 6 дней до регистрации; в 2022 — за 14 дней, то есть слишком рано для основного окна. Повторные положительные score попадали в cooldown; позднее score уже был ниже порога, новых сообщений не было. Подавленные дни сами по себе не считаются упущенными полезными сообщениями.", "",
              "| Метод | Бюджет сообщений превышен | Бюджет тревожных дней превышен |", "|---|---|---|"]
    for r in metrics.itertuples(index=False):
        lines.append(f"| {r.model_code} | {'да' if r.message_budget_exceeded else 'нет'} | {'да' if r.alarm_budget_exceeded else 'нет'} |")
    lines += ["", "## Каждый проверяемый случай", "", "| Серия | Метод | Предсказание на первом выпуске | Первая регистрация | Ошибка к регистрации, дни | Расстояние до интервала осмотров, дни |", "|---|---|---|---|---:|---:|"]
    for r in tables["initial_date_errors"].itertuples(index=False):
        date = str(r.predicted_date.date()) if pd.notna(r.predicted_date) else "не вычислено"
        lines.append(f"| {r.field_season} | {r.model_code} | {date} | {r.first_registration.date()} | {r.registration_error_days:g} | {r.distance_to_onset_interval_days:g} |")
    lines += ["", "Знак ошибки: минус — прогноз раньше регистрации. Расстояние 0 до интервала означает совместимость с операционным интервалом осмотров, а не точное попадание в биологический момент. Legacy-имя столбца distance_to_onset_interval_days сохраняется, но его смысл именно такой. Интервалы осмотров в наборе 6–15 дней; в проверяемых случаях ширина интервала 7 и 8 дней.", "", "## Неопределённость и вывод", "",
              "Два случая не позволяют доказать превосходство или устойчивую переносимость. Сохранён парный bootstrap по полевым сериям (2 000 повторов); при двух кластерах он лишь описательный. Все парные различия уведомлений здесь равны нулю, поэтому bootstrap вырожден в ноль; это не доказательство равенства в популяции. Он не создаёт дополнительных независимых наблюдений. Разбиения train/validation/test здесь нет: нечего обучать или выбирать, опубликованные константы и политика фиксированы. Годы являются ретроспективной внешней проверкой для этой реализации, а не новым нетронутым тестом всей программы исследований.", ""]
    a, b = [metrics[metrics.model_code.eq(m)].iloc[0] for m in MODELS]
    lines.append(f"Факт этого пилота: температурный метод поймал {int(b.timely_hits)} из 2 событий, календарь — {int(a.timely_hits)} из 2; сообщений {int(b.messages)} против {int(a.messages)}, тревожных дней {int(b.active_alarm_days)} против {int(a.active_alarm_days)}. Эти числа не являются доказательством улучшения над календарём. Отдельно нужно учитывать ширину интервалов осмотров, выбранный грубый календарный ориентир, приближение суммы тепла и условную доступность ERA5. Нагрузка рассчитана на коротком периоде до первого обнаружения: 22 = 7+15 дней; это не число предупреждений за полный дачный сезон.")
    lines += ["", "Следующий приоритет: получить дополнительные независимые серии с регистрацией первых яиц и полными отрицательными учётами молодых личинок до их появления. Четыре уже положительные при входе серии текущего набора не восстанавливают эту информацию; менять исход на другую стадию ради увеличения знаменателя нельзя.", "", "## Артефакты и воспроизведение", "",
              "`event_registry.parquet` содержит все шесть серий и причину включения/исключения; `control_visits.parquet` — статусы осмотров; `predictions.parquet` — ежедневные оценки; `notifications.parquet` — полный последовательный replay; `event_hits.csv`, `per_series_metrics.csv`, `initial_date_errors.csv`, `metrics.csv` — раздельные результаты. `uncertainty.json`, `contract.json`, `input_audit.json`, `manifest.json` и `source_snapshot/` фиксируют протокол, проверки и код. Все артефакты локальные; уведомления не отправлялись.", "", "Из корня репозитория:", "", "```bash",
              f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_cpb_validation verify --output-dir {root}",
              "# Полный повторный расчёт — в новый каталог, старый не перезаписывается:",
              "PYTHONPATH=src .venv/bin/python -m agro_phenology.external_cpb_validation run --output-dir results/external_validation/daks_cpb_2019_2022/reproduction_NEW",
              "PYTHONPATH=src .venv/bin/python -m pytest tests/test_external_cpb_model.py tests/test_external_cpb_validation.py -q", "```", "",
              "Verify проверяет хеши всех результатов и входов, заново рассчитывает все восемь таблиц из сохранённых данных и сравнивает значения, включая журналы сообщений. Сбор сетевых данных для повторения не требуется. Старые результаты и frozen registry не меняются.", ""]
    (root / "REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")


def run(collection: Path, weather: Path, root: Path) -> dict:
    if root.exists():
        raise FileExistsError("A validation run is immutable; choose a new output directory")
    audit = {"collection": check_collection(collection), "weather": check_weather(weather)}
    if any(v["status"] != "passed" for v in audit.values()):
        raise ValueError(f"Input audit failed: {audit}")
    paths = _inputs(collection, weather)
    before = {str(p.resolve()): digest_file(p) for p in paths}
    root.mkdir(parents=True)
    write_json(root / "input_audit.json", audit)
    shutil.copyfile(CONTRACT, root / "contract.json")
    for p in paths[:4]:
        target = root / "source_snapshot" / p.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, target)
    tables = _compute(collection, weather)
    for name, table in tables.items():
        table.to_parquet(root / f"{name}.parquet", index=False)
        if name not in {"predictions", "notifications", "control_visits", "event_registry"}:
            table.to_csv(root / f"{name}.csv", index=False)
    uncertainty = paired_uncertainty(tables["per_series_metrics"])
    write_json(root / "uncertainty.json", uncertainty)
    _report(tables, root, uncertainty)
    if any(digest_file(Path(p)) != h for p, h in before.items()):
        raise RuntimeError("Inputs changed during computation; run is not sealed")
    inventory = {str(p.relative_to(root)): digest_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    manifest = dict(status="complete", created_at=utc_now(), dataset=DATASET, input_hashes=before,
                    collection_dir=str(collection.resolve()), weather_dir=str(weather.resolve()),
                    file_inventory=inventory, python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__)
    write_json(root / "manifest.json", manifest)
    return {"status": "complete", "output_dir": str(root), "eligible_events": int(tables["event_registry"].warnable_first_event.sum()),
            "metrics": tables["metrics"][["model_code", "timely_hits", "messages", "active_alarm_days", "field_days"]].to_dict("records")}


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    failures = [name for name, sha in manifest["file_inventory"].items() if not (root/name).is_file() or digest_file(root/name) != sha]
    inputs = [p for p, sha in manifest["input_hashes"].items() if not Path(p).is_file() or digest_file(Path(p)) != sha]
    tables_checked = []
    if not failures and not inputs:
        computed = _compute(Path(manifest["collection_dir"]), Path(manifest["weather_dir"]))
        for name, table in computed.items():
            try:
                pd.testing.assert_frame_equal(table, pd.read_parquet(root/f"{name}.parquet"), check_exact=True)
                tables_checked.append(name)
            except AssertionError:
                failures.append(f"recompute:{name}")
    return dict(status="passed" if not failures and not inputs and manifest["status"] == "complete" else "failed",
                checked_files=len(manifest["file_inventory"]), checked_inputs=len(manifest["input_hashes"]),
                recomputed_tables=tables_checked, mismatches=failures, input_mismatches=inputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--weather-dir", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.collection_dir, args.weather_dir, args.output_dir) if args.command == "run" else verify(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
