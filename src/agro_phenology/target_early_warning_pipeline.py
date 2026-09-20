"""Reproducible early-warning pipelines for codling moth and apple scab."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd

from .apple_target_labels import (
    APPLE_CROP_CODE,
    APPLE_SCAB_ID,
    CODLING_MOTH_ID,
    deduplicate_apple_visits,
    label_apple_rows,
)
from .early_warning_core import sha256_file
from .early_warning_reporting import aggregate_pooled_metrics, paired_year_bootstrap
from .target_early_warning_core import add_daily_features, build_daily_decisions, build_field_seasons
from .target_early_warning_models import run_experiments


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_IDS = {"codling_moth": CODLING_MOTH_ID, "apple_scab": APPLE_SCAB_ID}
TARGET_NAMES_RU = {"codling_moth": "яблонная плодожорка", "apple_scab": "парша яблони"}
DEFAULT_CONTRACTS = {
    "codling_moth": REPO_ROOT / "docs/research/codling_moth_early_warning/evaluation_contract.json",
    "apple_scab": REPO_ROOT / "docs/research/apple_scab_early_warning/evaluation_contract.json",
}
SOURCE_FILES = (
    "pyproject.toml",
    "requirements-early-warning.txt",
    "src/agro_phenology/__init__.py",
    "src/agro_phenology/apple_target_labels.py",
    "src/agro_phenology/early_warning_core.py",
    "src/agro_phenology/target_early_warning_core.py",
    "src/agro_phenology/target_early_warning_models.py",
    "src/agro_phenology/target_early_warning_pipeline.py",
    "src/agro_phenology/early_warning_models.py",
    "src/agro_phenology/early_warning_reporting.py",
    "tests/test_apple_target_labels.py",
    "tests/test_target_early_warning.py",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _target_id(contract: dict) -> int:
    key = str(contract["target_key"])
    expected = TARGET_IDS.get(key)
    actual = int(contract["host_filter"]["organism_id"])
    if expected is None or actual != expected:
        raise ValueError(f"Incompatible target_key/organism_id: {key}/{actual}")
    if int(contract["host_filter"]["crop_code"]) != APPLE_CROP_CODE:
        raise ValueError("The orchard pipeline requires apple crop_code=41")
    return actual


def prepare_visits(csv_path: Path, contract: dict) -> tuple[pd.DataFrame, dict]:
    """Label and filter the target without exposing omissions as negatives."""
    raw = pd.read_csv(csv_path, low_memory=False)
    target_id = _target_id(contract)
    target_key = str(contract["target_key"])
    labeled = label_apple_rows(raw, target_id)
    labeled["observation_date"] = pd.to_datetime(labeled["observation_date"], errors="coerce")
    labeled["growth_stage_code"] = pd.to_numeric(
        labeled.get("growth_stage_code"), errors="coerce"
    )
    labeled["exclusion_reason"] = ""
    geo_valid = (
        labeled["final_coordinate_class"].isin({"A_direct", "B_new_subtraction"})
        & labeled["final_field_uid"].notna()
        & labeled["final_latitude"].between(55, 59)
        & labeled["final_longitude"].between(20, 29)
    )
    labeled.loc[~geo_valid, "exclusion_reason"] = "no_safe_final_coordinates"
    labeled.loc[labeled["observation_date"].isna(), "exclusion_reason"] = "invalid_date"
    storage = labeled["growth_stage_code"].eq(99) | labeled.get(
        "crop_stage_raw", pd.Series("", index=labeled.index)
    ).fillna("").str.contains(r"noliktav|uzglabāšanas", case=False, regex=True)
    labeled.loc[storage, "exclusion_reason"] = "storage_or_postharvest"
    in_season = labeled["observation_date"].dt.month.between(4, 10)
    labeled.loc[~in_season, "exclusion_reason"] = "outside_orchard_field_season"
    eligible = labeled.loc[labeled["exclusion_reason"].eq("")].copy()
    visits = deduplicate_apple_visits(eligible, target_id)
    visits["calendar_year"] = visits["observation_date"].dt.year.astype(int)
    visits["crop_season"] = visits["calendar_year"]
    visits["field_season"] = (
        visits["final_field_uid"].astype(str)
        + "_"
        + visits["crop_code"].astype(int).astype(str)
        + "_"
        + visits["crop_season"].astype(str)
    )
    visits["adult_trap_positive"] = (
        visits.get("adult_trap_status", pd.Series("unassessed", index=visits.index)).eq("positive")
    )
    visits = visits.sort_values(["field_season", "observation_date", "visit_id"]).reset_index(
        drop=True
    )
    audit = {
        "input_path": str(csv_path),
        "input_sha256": sha256_file(csv_path),
        "source_rows": int(len(raw)),
        "apple_rows": int(len(labeled)),
        "eligible_source_rows": int(len(eligible)),
        "deduplicated_visits": int(len(visits)),
        "fields": int(visits["final_field_uid"].nunique()),
        "field_seasons": int(visits["field_season"].nunique()),
        "label_status_counts": {
            str(k): int(v) for k, v in visits["label_status"].value_counts().items()
        },
        "source_conflicts": int(visits["source_conflict"].fillna(False).sum()),
        "exclusion_counts": {
            str(k): int(v)
            for k, v in labeled.loc[
                labeled["exclusion_reason"].ne(""), "exclusion_reason"
            ].value_counts().items()
        },
        "target_key": target_key,
        "target_id": target_id,
    }
    if target_key == "codling_moth":
        audit["adult_trap_status_counts"] = {
            str(k): int(v) for k, v in visits["adult_trap_status"].value_counts().items()
        }
        audit["adult_trap_source_conflicts"] = int(
            visits["adult_trap_source_conflict"].fillna(False).sum()
        )
    return visits, audit


def _validate_contract(contract: dict) -> None:
    _target_id(contract)
    lead = contract["timeliness_window_days"]
    if (int(lead["minimum"]), int(lead["maximum"])) != (3, 10):
        raise ValueError("This cycle is frozen to the 3-10 day warning window")
    weather = contract["weather_availability"]
    if int(weather["cutoff_days_before_issue"]) != 2:
        raise ValueError("This cycle is frozen to weather through issue_date-2")
    folds = contract["rolling_origin_folds"]
    for fold in folds:
        if int(fold["train_years"][1]) >= int(fold["validation_years"][0]):
            raise ValueError(f"Training overlaps validation in {fold['id']}")
        if int(fold["validation_years"][1]) >= int(fold["test_years"][0]):
            raise ValueError(f"Validation overlaps test in {fold['id']}")
    if contract.get("optuna", {}).get("enabled"):
        raise ValueError("Optuna is excluded from the first orchard cycle")


def _invariants(
    visits: pd.DataFrame,
    seasons: pd.DataFrame,
    decisions: pd.DataFrame,
    mapping: pd.DataFrame,
    contract: dict,
) -> dict:
    checks = {
        "unique_visit_field_date": not visits.duplicated(
            ["final_field_uid", "observation_date"]
        ).any(),
        "unique_field_season": not seasons["field_season"].duplicated().any(),
        "one_first_event_per_field_season": not seasons.loc[
            seasons["first_recorded_event_date"].notna(), "field_season"
        ].duplicated().any(),
        "weather_cutoff_exactly_two_days": (
            (decisions["issue_date"] - decisions["feature_cutoff_nasa_date"]).dt.days.eq(2).all()
        ),
        "weather_cutoff_strictly_before_issue": decisions["feature_cutoff_nasa_date"].lt(
            decisions["issue_date"]
        ).all(),
        "service_stops_only_after_event_available": not (
            decisions["service_active"]
            & decisions["first_recorded_event_date"].notna()
            & decisions["issue_date"].gt(decisions["first_recorded_event_date"])
        ).any(),
        "mapping_complete": decisions["nasa_cell"].notna().all(),
        "mapping_unique": not mapping.duplicated(["final_latitude", "final_longitude"]).any(),
        "candidate_mask_is_weather_only": decisions["candidate_comparison_complete"].equals(
            decisions["common_weather_complete"]
        ),
        "crop_code_41_only": visits["crop_code"].astype(int).eq(41).all(),
        "no_rows_outside_april_october": visits["observation_date"].dt.month.between(4, 10).all(),
    }
    if contract["target_key"] == "codling_moth":
        checks["adult_zero_not_primary_positive"] = not (
            visits["adult_trap_status"].eq("explicit_target_absent")
            & visits["label_status"].eq("positive")
            & visits["prevalence_pct"].isna()
        ).any()
        checks["biofix_never_available_before_next_day"] = (
            decisions.loc[
                decisions["first_adult_trap_positive_available_date"].notna(),
                "first_adult_trap_positive_available_date",
            ]
            > decisions.loc[
                decisions["first_adult_trap_positive_available_date"].notna(),
                "first_adult_trap_positive_date",
            ]
        ).all()
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {"status": "passed" if not failed else "failed", "checks": checks, "failed": failed}


def _save_frames(run_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    parquet = {"predictions", "validation_predictions", "alarm_states", "event_hits"}
    for name, frame in frames.items():
        if name in parquet:
            frame.to_parquet(run_dir / f"{name}.parquet", index=False)
        else:
            frame.to_csv(run_dir / f"{name}.csv", index=False)
    notifications = frames["alarm_states"].loc[
        frames["alarm_states"]["message_issued"] | frames["alarm_states"]["suppressed_repeat"]
    ].copy()
    notifications.to_parquet(run_dir / "notification_log.parquet", index=False)


def _gained_lost(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare event hits on the exact same external event identities."""
    source = frames["event_hits"].loc[
        frames["event_hits"]["season"].between(2020, 2025)
        & frames["event_hits"]["evaluation_scope"].eq("service_calendar")
        & frames["event_hits"]["slice"].eq("A_plus_B")
        & frames["event_hits"]["warnable_event"].astype(bool)
        & frames["event_hits"]["model_code"].isin(
            ["O0", "O1", "O2", "O3", "O4", "calendar_window"]
        )
    ].copy()
    if source.duplicated(["season", "field_season", "model_code"]).any():
        raise ValueError("Duplicate event identities prevent gained/lost analysis")
    wide = source.pivot(
        index=["season", "field_season"], columns="model_code", values="timely_hit"
    ).reset_index()
    events: list[dict] = []
    summaries: list[dict] = []
    for baseline in ("calendar_window", "O0", "O1", "O2", "O4"):
        if baseline not in wide or "O3" not in wide:
            continue
        rows = wide[["season", "field_season", "O3", baseline]].copy()
        rows["candidate"] = "O3"
        rows["baseline"] = baseline
        rows["category"] = np.select(
            [
                rows["O3"].astype(bool) & rows[baseline].astype(bool),
                rows["O3"].astype(bool) & ~rows[baseline].astype(bool),
                ~rows["O3"].astype(bool) & rows[baseline].astype(bool),
            ],
            ["both", "gained_by_O3", "lost_by_O3"],
            default="neither",
        )
        events.extend(rows.to_dict(orient="records"))
        counts = rows["category"].value_counts()
        summaries.append(
            {
                "candidate": "O3",
                "baseline": baseline,
                "events": int(len(rows)),
                "both": int(counts.get("both", 0)),
                "gained_by_O3": int(counts.get("gained_by_O3", 0)),
                "lost_by_O3": int(counts.get("lost_by_O3", 0)),
                "neither": int(counts.get("neither", 0)),
                "net_gain": int(counts.get("gained_by_O3", 0) - counts.get("lost_by_O3", 0)),
            }
        )
    return pd.DataFrame(events), pd.DataFrame(summaries)


def _summary_row(summary: pd.DataFrame, model: str, scope: str = "service_calendar") -> pd.Series | None:
    subset = summary.loc[
        summary["period"].eq("2020_2025")
        & summary["evaluation_scope"].eq(scope)
        & summary["slice"].eq("A_plus_B")
        & summary["model_code"].eq(model)
    ]
    return subset.iloc[0] if len(subset) else None


def _format_metric(row: pd.Series | None) -> str:
    if row is None:
        return "нет результата"
    return (
        f"{int(row['timely_hits'])}/{int(row['events_with_warning_opportunity'])} "
        f"({100 * float(row['timely_recall']):.1f}%), "
        f"{int(row['messages'])} сообщений, "
        f"{float(row['messages_per_30_field_days']):.3f} на 30 поле-дней, "
        f"{100 * float(row['active_alarm_fraction']):.1f}% тревожных дней, "
        f"вычислимость {100 * float(row['computable_fraction']):.1f}%"
    )


def write_report(
    run_dir: Path,
    contract: dict,
    audit: dict,
    seasons: pd.DataFrame,
    pooled: pd.DataFrame,
    bootstrap: pd.DataFrame,
    annual_burden: pd.DataFrame,
) -> None:
    target = str(contract["target_key"])
    target_name = TARGET_NAMES_RU[target]
    rows = {name: _summary_row(pooled, name) for name in ("calendar_window", "O0", "O1", "O2", "O3", "O4")}
    o3, o2, o1, o0, cal = rows["O3"], rows["O2"], rows["O1"], rows["O0"], rows["calendar_window"]
    comparisons: list[str] = []
    for baseline, label in ((o2, "O2"), (o1, "O1"), (o0, "O0"), (cal, "календарного окна")):
        if o3 is not None and baseline is not None:
            delta = int(o3["timely_hits"]) - int(baseline["timely_hits"])
            delta_messages = float(o3["messages_per_30_field_days"]) - float(
                baseline["messages_per_30_field_days"]
            )
            delta_alarm = float(o3["active_alarm_fraction"]) - float(
                baseline["active_alarm_fraction"]
            )
            comparisons.append(
                f"O3 против {label}: {delta:+d} своевременных событий, "
                f"Δ сообщений/30 {delta_messages:+.3f}, Δ тревожных дней {100 * delta_alarm:+.1f} п.п."
            )

    boot = bootstrap.loc[
        bootstrap["period"].eq("2020_2025")
        & bootstrap["evaluation_scope"].eq("service_calendar")
        & bootstrap["slice"].eq("A_plus_B")
        & bootstrap["candidate"].eq("O3")
        & bootstrap["baseline"].isin(["O2", "O1", "O0", "calendar_window"])
    ].copy()
    uncertainty_lines = []
    for row in boot.itertuples(index=False):
        low = getattr(row, "delta_timely_recall_low", np.nan)
        high = getattr(row, "delta_timely_recall_high", np.nan)
        uncertainty_lines.append(
            f"- O3 против {row.baseline}: Δ timely recall {100 * row.delta_timely_recall:+.1f} п.п.; "
            f"описательный 95% bootstrap [{100 * low:+.1f}; {100 * high:+.1f}] п.п."
        )
    if not uncertainty_lines:
        uncertainty_lines = ["- Парный годовой интервал не удалось оценить."]

    budget = contract["notification_policy"]["research_budget"]
    exceed = annual_burden.loc[
        annual_burden["evaluation_scope"].eq("service_calendar")
        & annual_burden["slice"].eq("A_plus_B")
        & annual_burden["season"].between(2020, 2025)
        & (
            annual_burden["messages_per_30_field_days"].gt(
                float(budget["messages_per_30_field_days_max"])
            )
            | annual_burden["active_alarm_fraction"].gt(
                float(budget["active_alarm_fraction_max"])
            )
        )
    ]
    if len(exceed):
        grouped = []
        for season, group in exceed.groupby("season", sort=True):
            values = ", ".join(
                f"{row.model_code}={row.messages_per_30_field_days:.3f} сообщ./30, "
                f"{100 * row.active_alarm_fraction:.1f}% тревожных дней"
                for row in group.itertuples(index=False)
            )
            grouped.append(f"{int(season)}: {values}")
        budget_note = (
            "На внешних годах зафиксированы превышения исследовательского бюджета: "
            + "; ".join(grouped)
            + ". Порог не перенастраивался по test."
        )
    else:
        budget_note = "На внешних годах 2020–2025 исследовательский бюджет не превышен."

    first_events = int(seasons["first_recorded_event_date"].notna().sum())
    warnable = int(seasons["warnable_first_event"].sum())
    result_statement = ""
    if o3 is not None and o2 is not None and cal is not None:
        best_baseline = max(
            int(o2["timely_hits"]),
            int(o1["timely_hits"]),
            int(cal["timely_hits"]),
            int(o0["timely_hits"]),
        )
        within_budget = bool(
            float(o3["messages_per_30_field_days"])
            <= float(contract["notification_policy"]["research_budget"]["messages_per_30_field_days_max"])
            and float(o3["active_alarm_fraction"])
            <= float(contract["notification_policy"]["research_budget"]["active_alarm_fraction_max"])
        )
        better_than_all = int(o3["timely_hits"]) > best_baseline
        no_more_load_than_calendar = bool(
            float(o3["messages_per_30_field_days"])
            <= float(cal["messages_per_30_field_days"])
            and float(o3["active_alarm_fraction"]) <= float(cal["active_alarm_fraction"])
        )
        if better_than_all and within_budget and no_more_load_than_calendar:
            result_statement = (
                "O3 превысила все календарные и погодные контроли без увеличения нагрузки относительно "
                "простого календарного окна. Это ретроспективный кандидат, который всё равно требует "
                "проверки на будущих сезонах."
            )
        elif better_than_all and within_budget:
            result_statement = (
                "O3 имеет лучший точечный охват в разрешённом абсолютном бюджете, но достигает его с "
                "большей нагрузкой, чем простое календарное окно, а годовые интервалы включают отсутствие "
                "эффекта. Более полезный, чем календарь, предсказатель пока не подтверждён."
            )
        else:
            result_statement = (
                "O3 не превысила все календарные и погодные контроли. Более полезный, чем календарь, "
                "предсказатель в этом цикле не получен. Отрицательный результат сохранён без смены окна, "
                "выборки или порогов по внешним годам."
            )

    partial_rows: dict[str, pd.Series | None] = {}
    for name in ("calendar_window", "O0", "O2", "O3"):
        subset = pooled.loc[
            pooled["period"].eq("2026_partial")
            & pooled["evaluation_scope"].eq("service_calendar")
            & pooled["slice"].eq("A_plus_B")
            & pooled["model_code"].eq(name)
        ]
        partial_rows[name] = subset.iloc[0] if len(subset) else None

    special = (
        "У плодожорки основная цель — первая регистрация личинки; взрослые особи в ловушке используются "
        "только как доступный со следующего дня biofix-proxy. Пороговые суммы 126/230 являются старыми "
        "гипотезами проекта, а не подтверждёнными нормативами."
        if target == "codling_moth"
        else "Погодные признаки O3 — суточный тёпло-влажный proxy. Это не Mills: в источнике нет "
        "непрерывной длительности смачивания листа и внутрисуточной температуры влажного периода."
    )
    lines = [
        f"# Первый цикл раннего предупреждения: {target_name}",
        "",
        f"Запуск: `{run_dir.name}`. Результаты ретроспективные; 2023–2025 уже изучались и не являются новым независимым тестом.",
        "",
        "## Что оценивалось",
        "",
        f"Цель — первое зарегистрированное событие с предупреждением за 3–10 дней. Всего в локальном реестре {first_events} первых событий, из них {warnable} имели хотя бы трёхдневную возможность после подключения.",
        "Отсутствие записи или визита не трактуется как подтверждённое отсутствие организма. Модели оцениваются по первым событиям и нагрузке сообщений, а не по визитной accuracy/specificity.",
        "",
        special,
        "",
        "## Основной pooled-результат 2020–2025, полный сервисный календарь",
        "",
        *[f"- **{code}**: {_format_metric(rows[code])}." for code in ("calendar_window", "O0", "O1", "O2", "O3", "O4")],
        "",
        "Сравнение O3: " + "; ".join(comparisons) + ".",
        "",
        budget_note,
        "",
        "## Неопределённость",
        "",
        *uncertainty_lines,
        "",
        "Интервалы получены парным bootstrap шести внешних лет. Они не учитывают повторное использование полей, перекрывающиеся обучающие наборы, выбор модели и неоднородность мониторинга.",
        "",
        "## Неполный 2026 год — только описательно",
        "",
        *[f"- **{code}**: {_format_metric(partial_rows[code])}." for code in ("calendar_window", "O0", "O2", "O3")],
        "",
        "Этот срез не входит в основной pooled-результат и не использовался для выбора параметров или порогов.",
        "",
        "## Прямой вывод",
        "",
        result_statement,
        "",
        "## Данные и ограничения",
        "",
        f"- После фильтров: {audit['visit_preparation']['deduplicated_visits']} визитов, {audit['visit_preparation']['field_seasons']} поле-сезонов и {audit['visit_preparation']['fields']} полей.",
        f"- Замороженная NASA-погода вычислима на {100 * audit['weather']['common_weather_computable_fraction']:.1f}% активных дней. Признаки заканчиваются на issue_date−2; фактический исторический журнал публикации NASA отсутствует.",
        "- Координаты и идентификаторы полей остаются только в локальных технических артефактах. Отчёт содержит агрегаты.",
        "- Дата первой регистрации не равна биологическому началу процесса и не является рекомендацией по обработке.",
        "",
        "## Воспроизведение",
        "",
        "Команда и хеши сохранены в `REPRODUCE.md` и `execution_manifest.json`. Проверка целостности: `check --run-dir <каталог>`.",
    ]
    (run_dir / "report_ru.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_hashes(run_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "execution_manifest.json"
    }


def _source_hashes() -> dict[str, str | None]:
    return {
        path: sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in SOURCE_FILES
    }


def _write_source_snapshot(run_dir: Path) -> None:
    destination = run_dir / "source_snapshot"
    for relative in SOURCE_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _write_review_package(
    run_dir: Path,
    reporting: dict[str, pd.DataFrame],
    bootstrap: pd.DataFrame,
    gained_lost: pd.DataFrame,
    audit: dict,
) -> None:
    package = run_dir / "review_package"
    package.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_dir / "report_ru.md", package / "report_ru.md")
    reporting["pooled_summary"].to_csv(package / "pooled_summary.csv", index=False)
    bootstrap.to_csv(package / "paired_year_bootstrap.csv", index=False)
    gained_lost.to_csv(package / "gained_lost_summary.csv", index=False)
    safe_audit = {
        "target_key": audit["visit_preparation"]["target_key"],
        "visits": audit["visit_preparation"]["deduplicated_visits"],
        "fields": audit["visit_preparation"]["fields"],
        "field_seasons": audit["visit_preparation"]["field_seasons"],
        "label_status_counts": audit["visit_preparation"]["label_status_counts"],
        "first_events": audit["field_seasons"]["first_events"],
        "warnable_first_events": audit["field_seasons"]["warnable_first_events"],
        "weather_computable_fraction": audit["weather"]["common_weather_computable_fraction"],
        "privacy": "aggregate_only_no_field_ids_coordinates_or_raw_observation_text",
    }
    _write_json(package / "aggregate_data_audit.json", safe_audit)
    (package / "README.md").write_text(
        "# Обезличенный пакет для просмотра\n\n"
        "Пакет содержит только агрегаты, методический отчёт и неопределённость. "
        "Идентификаторы полей, координаты, сырые записи и ежедневные прогнозы остаются вне пакета.\n",
        encoding="utf-8",
    )


def _run_tests() -> dict:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_apple_target_labels.py",
        "tests/test_target_early_warning.py",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_pipeline(contract_path: Path, run_id: str) -> Path:
    contract_path = contract_path.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_contract(contract)
    run_dir = _resolve(contract["output_root"]) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty and will not be overwritten: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "target_key": contract["target_key"],
        "status": "running",
        "started_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "git_revision": _git("rev-parse", "HEAD"),
        "git_status_short_at_start": _git("status", "--short").splitlines(),
        "source_hashes": _source_hashes(),
    }
    _write_json(run_dir / "execution_manifest.json", manifest)
    try:
        _write_source_snapshot(run_dir)
        shutil.copy2(contract_path, run_dir / "evaluation_contract.json")
        protocol = contract_path.with_name("protocol.md")
        if protocol.is_file():
            shutil.copy2(protocol, run_dir / "protocol.md")

        csv_path = _resolve(contract["inputs"]["conservative_v2_csv"])
        nasa_path = _resolve(contract["weather_availability"]["input_path"])
        if not csv_path.is_file() or not nasa_path.is_file():
            raise FileNotFoundError(f"Required input missing: csv={csv_path.is_file()} nasa={nasa_path.is_file()}")
        inventory = {
            "conservative_v2_csv": {
                "path": str(csv_path),
                "sha256": sha256_file(csv_path),
                "bytes": csv_path.stat().st_size,
            },
            "nasa_daily": {
                "path": str(nasa_path),
                "sha256": sha256_file(nasa_path),
                "bytes": nasa_path.stat().st_size,
            },
            "network_used": False,
        }
        _write_json(run_dir / "input_inventory.json", inventory)

        visits, visit_audit = prepare_visits(csv_path, contract)
        seasons = build_field_seasons(visits, snapshot_date=contract["global_snapshot_date"])
        decisions = build_daily_decisions(
            seasons,
            timezone=contract["timezone"],
            issue_time=contract["daily_issue_time"],
            minimum_lead=int(contract["timeliness_window_days"]["minimum"]),
            maximum_lead=int(contract["timeliness_window_days"]["maximum"]),
            weather_cutoff_days=int(contract["weather_availability"]["cutoff_days_before_issue"]),
        )
        decisions, mapping, weather_audit = add_daily_features(
            decisions, nasa_daily_path=nasa_path, target_key=contract["target_key"]
        )
        invariants = _invariants(visits, seasons, decisions, mapping, contract)
        if invariants["status"] != "passed":
            raise AssertionError(f"Pipeline invariants failed: {invariants['failed']}")
        audit = {
            "visit_preparation": visit_audit,
            "field_seasons": {
                "rows": int(len(seasons)),
                "first_events": int(seasons["first_recorded_event_date"].notna().sum()),
                "warnable_first_events": int(seasons["warnable_first_event"].sum()),
                "positive_at_entry": int(seasons["positive_at_first_visit"].sum()),
                "entry_categories": {
                    str(k): int(v) for k, v in seasons["entry_category"].value_counts().items()
                },
            },
            "daily_decisions": {
                "rows": int(len(decisions)),
                "service_field_days": int(decisions["service_active"].sum()),
                "observable_rows": int(decisions["target_observable"].sum()),
                "target_class_counts": {
                    str(k): int(v) for k, v in decisions["target_class"].value_counts().items()
                },
            },
            "weather": weather_audit,
            "invariants": invariants,
        }
        _write_json(run_dir / "data_audit.json", audit)
        visits.to_parquet(run_dir / "apple_visits.parquet", index=False)
        seasons.to_parquet(run_dir / "field_seasons.parquet", index=False)
        seasons.loc[seasons["first_recorded_event_date"].notna()].to_parquet(
            run_dir / "events.parquet", index=False
        )
        decisions.to_parquet(run_dir / "daily_decisions.parquet", index=False)
        mapping.to_parquet(run_dir / "nasa_coordinate_mapping_apple.parquet", index=False)
        split_manifest = []
        for fold in contract["rolling_origin_folds"]:
            split_manifest.append(
                {
                    **fold,
                    "train_rows": int(decisions["season"].between(*fold["train_years"]).sum()),
                    "validation_rows": int(
                        decisions["season"].between(*fold["validation_years"]).sum()
                    ),
                    "test_rows": int(decisions["season"].between(*fold["test_years"]).sum()),
                    "test_events": int(
                        seasons.loc[
                            seasons["season"].between(*fold["test_years"]),
                            "first_recorded_event_date",
                        ].notna().sum()
                    ),
                }
            )
        _write_json(run_dir / "split_manifest.json", split_manifest)

        frames = run_experiments(decisions, seasons, contract, run_dir)
        _save_frames(run_dir, frames)
        gained_lost_events, gained_lost_summary = _gained_lost(frames)
        gained_lost_events.to_parquet(run_dir / "gained_lost_events.parquet", index=False)
        gained_lost_summary.to_csv(run_dir / "gained_lost_summary.csv", index=False)
        reporting = aggregate_pooled_metrics(
            frames["event_metrics"],
            frames["burden_metrics"],
            event_hits=frames["event_hits"],
            alarm_states=frames["alarm_states"],
            periods={
                "2020_2025": (2020, 2025),
                "2023_2025": (2023, 2025),
                "2026_partial": (2026, 2026),
            },
        )
        for name, frame in reporting.items():
            frame.to_csv(run_dir / f"{name}.csv", index=False)
        comparisons = (
            ("O1", "O0"),
            ("O2", "O0"),
            ("O3", "O0"),
            ("O3", "O1"),
            ("O3", "O2"),
            ("O4", "O2"),
            ("O2", "calendar_window"),
            ("O3", "calendar_window"),
        )
        bootstrap = paired_year_bootstrap(
            frames["event_hits"],
            frames["alarm_states"],
            seed=int(contract["random_seed"]),
            n_bootstrap=2000,
            periods={
                "2020_2025": (2020, 2025),
                "2023_2025": (2023, 2025),
                "2026_partial": (2026, 2026),
            },
            comparisons=comparisons,
        )
        bootstrap.to_csv(run_dir / "paired_year_bootstrap.csv", index=False)
        write_report(
            run_dir,
            contract,
            audit,
            seasons,
            reporting["pooled_summary"],
            bootstrap,
            frames["burden_metrics"],
        )
        _write_review_package(
            run_dir, reporting, bootstrap, gained_lost_summary, audit
        )
        command = (
            f"PYTHONPATH=src .venv/bin/python -m agro_phenology.target_early_warning_pipeline run "
            f"--contract {contract_path.relative_to(REPO_ROOT)} --run-id <new_empty_run_id>"
        )
        (run_dir / "REPRODUCE.md").write_text(
            "# Воспроизведение\n\nСоздайте новый пустой `--run-id`; завершённый каталог не перезаписывается.\n\n"
            f"```bash\n{command}\n```\n\nПроверка сохранённого запуска:\n\n"
            f"```bash\nPYTHONPATH=src .venv/bin/python -m agro_phenology.target_early_warning_pipeline check --run-dir {run_dir.relative_to(REPO_ROOT)}\n```\n",
            encoding="utf-8",
        )
        tests = _run_tests()
        _write_json(run_dir / "test_results.json", tests)
        if tests["status"] != "passed":
            raise RuntimeError("Focused orchard tests failed")
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "input_hashes": {name: value["sha256"] for name, value in inventory.items() if isinstance(value, dict)},
                "output_hashes": _output_hashes(run_dir),
                "scientific_status": "retrospective_first_registration_proxy_already_studied_years",
            }
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        return run_dir
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_output_hashes": _output_hashes(run_dir),
            }
        )
        _write_json(run_dir / "execution_manifest.json", manifest)
        raise


def check_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for relative, expected in manifest.get("output_hashes", {}).items():
        path = run_dir / relative
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
        else:
            actual = sha256_file(path)
            if actual != expected:
                mismatches.append(
                    {"path": relative, "reason": "hash_mismatch", "expected": expected, "actual": actual}
                )
    current_source_mismatches = []
    for relative, expected in manifest.get("source_hashes", {}).items():
        current = REPO_ROOT / relative
        actual = sha256_file(current) if current.is_file() else None
        if actual != expected:
            current_source_mismatches.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    input_mismatches = []
    inventory_path = run_dir / "input_inventory.json"
    if inventory_path.is_file():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for name, record in inventory.items():
            if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
                continue
            path = Path(record["path"])
            actual = sha256_file(path) if path.is_file() else None
            if actual != record["sha256"]:
                input_mismatches.append(
                    {"name": name, "path": str(path), "expected": record["sha256"], "actual": actual}
                )
    result = {
        "run_dir": str(run_dir),
        "manifest_status": manifest.get("status"),
        "checked_outputs": len(manifest.get("output_hashes", {})),
        "mismatches": mismatches,
        "input_mismatches": input_mismatches,
        "current_source_mismatches": current_source_mismatches,
        "stored_source_snapshot_available": (run_dir / "source_snapshot").is_dir(),
        "status": "passed"
        if manifest.get("status") == "complete" and not mismatches and not input_mismatches
        else "failed",
    }
    return result


def _parser(default_contract: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--contract", type=Path, default=default_contract, required=default_contract is None)
    run.add_argument("--run-id", required=True)
    check = sub.add_parser("check")
    check.add_argument("--run-dir", type=Path, required=True)
    return parser


def _main(argv: list[str] | None, default_contract: Path | None = None) -> int:
    args = _parser(default_contract).parse_args(argv)
    if args.command == "run":
        path = run_pipeline(_resolve(args.contract), args.run_id)
        print(path)
        return 0
    result = check_run(_resolve(args.run_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    return _main(argv)


def main_codling(argv: list[str] | None = None) -> int:
    return _main(argv, DEFAULT_CONTRACTS["codling_moth"])


def main_scab(argv: list[str] | None = None) -> int:
    return _main(argv, DEFAULT_CONTRACTS["apple_scab"])


if __name__ == "__main__":
    raise SystemExit(main())
