"""Immutable reporting bundle for late-blight shadow readiness.

The generator consumes already prepared machine-readable audits.  It performs
no model fitting, network access, scheduling, field lookup, or notification.
The destination is created atomically and is never updated in place.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid
import zipfile


REPORT_SCHEMA_VERSION = "shadow-readiness-report-v1"
DEFAULT_REPRODUCTION_COMMANDS = (
    "python -m agro_phenology.shadow_cli readiness --project-root . --registry <readiness-run>/shadow_registry.json",
    "python -m agro_phenology.shadow_cli technical-demo --project-root . --registry <readiness-run>/shadow_registry.json --database <new-demo.sqlite> --offline",
    "python -m agro_phenology.shadow_cli replay-from-log --project-root . --registry <readiness-run>/shadow_registry.json --database <shadow.sqlite>",
    "python -m agro_phenology.shadow_cli verify-log --project-root . --registry <readiness-run>/shadow_registry.json --database <shadow.sqlite>",
    "python -m agro_phenology.shadow_cli export-review-package --run-dir <readiness-run> --destination <new-output.zip>",
)


def _utc_text(value: str | datetime | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at_utc must be timezone-aware")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("created_at_utc must be expressed in UTC")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_blockers(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
        if "blockers" not in result:
            result = {"blockers": [result]}
    else:
        result = {"blockers": deepcopy(list(value))}
    blockers = result.get("blockers")
    if not isinstance(blockers, list):
        raise ValueError("blockers must be a list or an object containing a blockers list")
    return result


def _rows(value: Any, preferred_keys: Iterable[str]) -> list[dict[str, Any]]:
    def normalise(row: dict[str, Any]) -> dict[str, Any]:
        row.setdefault(
            "id",
            row.get("source_id", row.get("model_id", row.get("name", "—"))),
        )
        aliases = {
            "required": ("required_contract", "frozen_requirement", "required_input"),
            "actual": ("actual_source", "operational_source", "training_source"),
            "freshness": ("measured_freshness", "available_through", "freshness_status"),
            "status": ("operational_status", "compatibility_status", "decision"),
            "evidence": ("availability_evidence", "evidence_kind", "evidence_source"),
            "role": ("participant_role",),
            "required_inputs": ("inputs", "features", "requirements"),
            "fallback": ("fallback_behavior", "missing_input_behavior"),
            "reason": ("status_reason", "blocker", "notes"),
        }
        for target, candidates in aliases.items():
            if target not in row:
                for candidate in candidates:
                    if candidate in row:
                        row[target] = row[candidate]
                        break
        return row

    if isinstance(value, list):
        return [normalise(dict(row)) for row in value if isinstance(row, Mapping)]
    if not isinstance(value, Mapping):
        return []
    for key in preferred_keys:
        candidate = value.get(key)
        if isinstance(candidate, list):
            return [
                normalise(dict(row)) for row in candidate if isinstance(row, Mapping)
            ]
        if isinstance(candidate, Mapping):
            return [
                normalise({"id": name, **dict(row)})
                if isinstance(row, Mapping)
                else {"id": name, "status": row}
                for name, row in candidate.items()
            ]
    return []


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(item) for item in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{key}={_cell(item)}" for key, item in value.items())
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "Детализация отсутствует в переданном машинном аудите."
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_cell(row.get(key)) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _blocker_rows(blockers: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(blockers.get("blockers", []), start=1):
        row = dict(raw) if isinstance(raw, Mapping) else {"detail": str(raw)}
        result.append(
            {
                "id": row.get("id", f"B{index}"),
                "status": row.get("status", "blocked"),
                "detail": row.get("detail", row.get("description", row.get("reason", "—"))),
                "required_action": row.get(
                    "required_action", row.get("needed_from_user", row.get("resolution", "—"))
                ),
            }
        )
    return result


def _extract_bool(value: Mapping[str, Any], names: Sequence[str], default: bool) -> bool:
    pending: list[Mapping[str, Any]] = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        for name in names:
            candidate = current.get(name)
            if isinstance(candidate, bool):
                return candidate
        pending.extend(
            candidate for candidate in current.values() if isinstance(candidate, Mapping)
        )
    return default


def _find_value(value: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in value:
            return value[name]
    for candidate in value.values():
        if isinstance(candidate, Mapping):
            found = _find_value(candidate, names)
            if found is not None:
                return found
    return None


def _demo_metric_rows(demo: Mapping[str, Any]) -> list[dict[str, Any]]:
    definitions = (
        ("expected_daily_slots_per_field", "Ожидаемые ежедневные слоты на поле", ("expected_daily_slots_per_field",)),
        ("executed_daily_slots_per_field", "Выполненные ежедневные слоты на поле", ("executed_daily_slots_per_field",)),
        ("scheduled_slots", "Запланированные слоты", ("scheduled_slots", "slots_scheduled")),
        ("expected_decisions", "Ожидаемые решения", ("expected_decisions", "expected_records")),
        ("decision_records", "Записанные решения", ("decision_records", "completed_slots", "decisions", "decision_count", "records", "unique_decisions")),
        ("computed_fraction", "Доля вычисленных решений", ("computed_fraction", "computed_decision_fraction", "scorable_fraction")),
        ("missed_slots", "Пропущенные слоты", ("missed_slots", "slots_missed")),
        ("late_slots", "Поздние запуски", ("late_slots", "late_runs")),
        ("abstentions", "Воздержания", ("abstentions", "abstention_count")),
        ("abstentions_by_reason", "Воздержания по причинам", ("abstentions_by_reason", "abstention_reasons")),
        ("virtual_messages", "Виртуальные сообщения", ("virtual_messages", "messages", "message_count", "virtual_message_candidates")),
        ("active_alarm_days", "Тревожные дни", ("active_alarm_days", "alarm_days")),
        ("active_alarm_fraction", "Доля тревожных дней", ("active_alarm_fraction", "alarm_day_fraction")),
        ("compatible_weather_fraction", "Доля совместимой погоды", ("compatible_weather_fraction", "weather_compatible_fraction")),
        ("weather_correction_fraction", "Доля активной weather-поправки в C6", ("weather_correction_fraction", "weather_active_fraction", "active_weather_fraction_in_c6")),
        ("c0_fallback_fraction", "Доля C0 fallback", ("c0_fallback_fraction", "fallback_fraction")),
        ("effective_c0_fraction", "Доля эффективного C0 в C6", ("effective_c0_fraction", "effective_c0_fraction_in_c6")),
        ("abstention_fraction", "Доля воздержаний", ("abstention_fraction", "abstain_fraction")),
        ("field_registry_status", "Статус реестра полей", ("field_registry_status", "active_field_registry_status")),
        ("network_status", "Статус сети", ("network_status", "live_network_status", "network_used")),
        ("live_run_executed", "Реальный run-once выполнен", ("live_run_executed", "prospective_live_run", "prospective_live_run_performed")),
        ("real_notifications_sent", "Реальные уведомления", ("real_notifications_sent", "notifications_sent")),
        ("schedule_activated", "Расписание активировано", ("schedule_activated",)),
    )
    rows: list[dict[str, Any]] = []
    for metric_id, label, aliases in definitions:
        found = _find_value(demo, aliases)
        if found is not None:
            rows.append({"id": metric_id, "metric": label, "value": found})
    return rows


def _report(
    *,
    run_id: str,
    created_at: str,
    registry: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    demo: Mapping[str, Any],
    tests: Mapping[str, Any],
    blockers: Mapping[str, Any],
) -> str:
    source_rows = _rows(source_manifest, ("sources", "inputs", "source_checks"))
    branch_rows = _rows(source_manifest, ("branches", "models", "participants"))
    weather_live = _extract_bool(
        source_manifest,
        ("weather_branch_executable_today", "weather_branch_operational_today"),
        False,
    )
    demo_mode = str(demo.get("mode", demo.get("run_mode", "offline_technical_demo")))
    demo_status = str(demo.get("status", "completed"))
    test_status = str(tests.get("status", "unknown"))
    demo_metrics = _demo_metric_rows(demo)
    start_boundary = source_manifest.get("shadow_start_boundary", {})
    first_honest_slot = start_boundary.get(
        "first_honest_scheduled_slot_local", "не определён"
    )
    current_slot_allowed = bool(
        start_boundary.get("current_issue_slot_can_be_prospective", False)
    )
    if start_boundary:
        start_boundary_statement = (
            (
                "Readiness lock создан до scheduled-слота 08:00 Europe/Riga: "
                if current_slot_allowed
                else "Readiness lock создан после scheduled-слота 08:00 Europe/Riga 10.09.2026: "
            )
            + (
                "текущий слот может быть prospective только при наличии всех входов."
                if current_slot_allowed
                else "текущий слот **не может быть превращён в prospective задним числом**."
            )
            + f" Первый честный будущий слот — `{first_honest_slot}`, также только при "
            "наличии разрешённых входов. Он не был запущен и расписание для него не активировалось."
        )
    else:
        start_boundary_statement = (
            "Временная граница начала live-сбора не передана; это само по себе запрещает "
            "считать техническую демонстрацию prospective-запуском."
        )
    return f"""# Готовность теневого контура раннего предупреждения фитофтороза картофеля

Идентификатор запуска: `{run_id}`

Снимок сформирован: `{created_at}`

## Решение о готовности

Инженерный контур подготовлен для неизменяемого по версии хранения входов, раздельного журнала решений и наблюдений, последовательного состояния политик, однократного запуска и воспроизведения. В этом запуске выполнена только техническая offline-демонстрация (`{demo_mode}`, статус `{demo_status}`). Она **не является проспективным запуском**, не подтверждает качество предупреждений и не создаёт данных о пользе модели.

Погодная ветка frozen C6 на дату проверки {'может' if weather_live else '**не может**'} штатно использовать ERA5 до `issue_date−2`. Официально описанная задержка ERA5/ERA5T около пяти дней несовместима с этим cutoff. Cutoff не сдвигался, IFS, Best Match и прогноз Open-Meteo не подставлялись. При недоступной совместимой погоде C6 использует сохранённый C0 fallback; если fallback составляет 100%, это проверяет только календарную ветку и **не проверяет погодную поправку C6**. C4/C5 в такой ситуации должны воздержаться.

Реальные уведомления не отправлялись. Фоновое расписание не установлено и не активировано. Новые модели, признаки, alpha, пороги и политики не подбирались.

{start_boundary_statement}

## Совместимость источников

{_table(source_rows, (("id", "Источник/вход"), ("required", "Требование frozen"), ("actual", "Фактический источник"), ("freshness", "Свежесть"), ("status", "Статус"), ("evidence", "Основание")))}

Документированная задержка, фактически измеренная свежесть, предположение и отсутствие проверки должны оставаться разными видами доказательств. Инициализация прогноза не считается временем публикации. Поздно полученный реанализ хранится, но не меняет старое решение и не датируется задним числом.

## Исполнимость frozen-веток

{_table(branch_rows, (("id", "Ветка"), ("role", "Роль"), ("required_inputs", "Входы"), ("status", "Статус"), ("fallback", "Поведение при отказе"), ("reason", "Причина")))}

Календарное окно, C0 и календарный калибровочный контроль вычислимы без погоды после подключения актуального реестра действующих полей. C1/C4/C5 остаются диагностическими. C6 ведёт одну историю состояния для weather и fallback: смена источника score не создаёт рост риска и не сбрасывает cooldown.

## Что проверено технически

- реестр frozen-моделей и политик содержит пути, хеши, порядок классов и происхождение выбора последнего по времени совместимого bundle;
- исходные ответы хранятся content-addressed, а retrieval, наблюдения и решения добавляются отдельными версиями;
- решение и состояние политики фиксируются атомарно и идемпотентно;
- replay читает ровно те версии входов, которые были доступны на момент решения;
- истории политик разделены, а weather/fallback внутри C6 используют единую историю;
- будущие прогнозы допускаются только как `archive_only` и не входят автоматически в frozen past-only признаки;
- тестовый статус переданного прогона: `{test_status}`.

## Агрегаты технической демонстрации

{_table(demo_metrics, (("metric", "Показатель"), ("value", "Значение")))}

Эти числа характеризуют только исполнение кода на offline-fixture. Они не являются оценкой биологического качества, а нулевые/полные доли weather или fallback нельзя переносить на будущий live-сбор.

SQLite-триггеры, хеши и манифест помогают обнаружить случайное изменение, но без внешнего неизменяемого якоря не дают криптографического доказательства времени создания записи.

## Ограничения результата

Техническая демонстрация не оценивает чувствительность, PPV, specificity или биологический эффект. Синтетические ID и offline-fixtures не заменяют реальные поля, получение источника в момент решения и независимые плановые осмотры. Сам факт накопления прогнозов также не доказывает, что погода улучшает C0.

Прежняя схема будущей оценки сохраняется: окно успеха 3–10 календарных дней до первой зарегистрированной положительной записи, реальные даты подключения и выхода, зрелость исходов, неизвестность и цензурирование, нагрузка виртуальных сообщений и заранее установленная точка анализа. Будущая динамическая популяция не обязана иметь знаменатель 87.

## Блокеры живого сбора

{_table(_blocker_rows(blockers), (("id", "ID"), ("status", "Статус"), ("detail", "Что блокирует"), ("required_action", "Что требуется")))}

Для живого теневого сбора нужны актуальный реестр действующих полей с псевдонимными ID и защищённой привязкой к погодным ячейкам, разрешённый маршрут получения данных, а также независимый от score план полевых осмотров и ответственный процесс внесения исходов. Отдельное явное решение потребуется для активации расписания. До выполнения этих условий корректный статус: **контур подготовлен, live-сбор не запущен**.

## Источники сведений о задержке

- [ECMWF ERA5 data documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation) — ERA5T обычно доступен примерно с пятидневной задержкой; время ежедневной публикации не фиксировано.
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — ERA5 и ERA5-Land имеют заявленную задержку пять дней; Best Match смешивает продукты.
- [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api) — время инициализации run не равно подтверждённому времени публикации.
"""


def _decision_record(created_at: str) -> str:
    return f"""# Decision record: переход к теневому сбору

Дата фиксации: `{created_at}`

## Замороженное решение

Следующий этап — инженерная готовность к проспективному теневому сбору. Это не четвёртый подбор моделей на изученных годах. Окно предупреждения 3–10 дней, семантика первого зарегистрированного события, внешние разбиения, score, alpha, пороги и политики остаются замороженными.

## Результаты, на которых основано решение

| Система | Своевременно | Сообщений / 30 дней | Тревожных дней |
| --- | ---: | ---: | ---: |
| C6 + P_growth, полный сервис | 62/87 | 1.798 | 39.919% |
| C0 + P_growth | 53/87 | 1.759 | 38.807% |
| C1 + P_growth | 45/87 | 1.820 | 40.829% |
| Календарное окно | 45/87 | 1.244 | 27.426% |
| C4 + P_growth | 65/87 | 1.685 | 37.551% |

C6 против C1 имеет описательный интервал дельты recall `[0.010; 0.400]`. C6 против калибровочного контроля даёт 62 против 52 и интервал `[-0.021; 0.278]`. Поэтому самостоятельная польза погодной информации не установлена, а сильный выигрыш при не большей нагрузке не показан.

C4 вычислим только на 59.258% общей маски и не представляет полную оперативную систему. Для C6 + P_growth историческая сырая недоступность погодной поправки равна 40.742%, фактический fallback при положительной alpha — 23.108%, эффективный C0 — 65.728%. Эти величины имеют разные определения.

Интервалы получены по шести уже изученным годам и не учитывают полный адаптивный путь исследования. Они не являются независимым подтверждением новой политики. Новые результаты качества по старым годам в readiness-этапе не вычисляются.
"""


def _runbook(commands: Sequence[str]) -> str:
    command_lines = "\n".join(f"```bash\n{command}\n```" for command in commands)
    return f"""# RUNBOOK: теневой контур фитофтороза картофеля

## Предусловия

1. Использовать только версию `shadow_registry.json` из конкретного readiness-run.
2. Передать отдельный актуальный реестр действующих полей, прошедший проверку `schemas/active_field_registry.schema.json`. Исторические координаты 2020–2025 годов не назначаются действующим полям.
3. Хранить точную географическую привязку вне review package; в журнале решений использовать псевдонимную ссылку.
4. Получать только источник, совместимый с frozen-профилем. IFS, Best Match, ERA5-Land и forecast API не заменяют ERA5.
5. Оставить каналы доставки отключёнными. Контур создаёт только виртуальные сообщения.

## Команды

{command_lines}

`run-once` должен использовать фактическое время выполнения. Пропущенный слот не создаётся задним числом; поздний запуск записывается как `late_run`. Повтор той же комбинации registry/policy/field/slot возвращает существующее решение и не создаёт сообщение.

## Отказ источника

- при отсутствии совместимой погоды C6 использует C0 fallback в своей прежней единой истории;
- C4/C5 записывают явное воздержание;
- отсутствие поля, модели, истории или совместимого источника записывается отдельным статусом, а не score=0;
- запоздалая ERA5 сохраняется как поздняя версия и не меняет прежнее решение;
- будущий прогноз сохраняется отдельно с `archive_only` и не входит в старые признаки.

## Восстановление

После сбоя сначала выполнить `verify-log`, затем повторить тот же `run-once`. Атомарная фиксация не допускает решения без соответствующего состояния. Конкурирующая попытка должна получить конфликт версии состояния; менять уже записанное решение запрещено.

## Расписание

Расписание **не установлено**. Ниже только неактивный шаблон для будущего отдельного решения оператора:

```text
# НЕ АКТИВИРОВАТЬ БЕЗ ОТДЕЛЬНОГО РЕШЕНИЯ
# <minute> <hour> * * * cd <project> && <python> -m agro_phenology.shadow_cli run-once ...
```

Установка cron, launchd или system service не входит в этот запуск.
"""


def _field_instructions() -> str:
    return """# Инструкция по полевым наблюдениям

Целевой организм — фитофтороз картофеля. Реестр содержит только реально подключённые поля и реальные даты подключения/выхода. Техническая готовность интерфейса не означает, что партнёры или план осмотров уже существуют.

## До начала сбора

- назначить календарь части плановых осмотров заранее и независимо от model score;
- определить ответственных, способ подтверждения и срок внесения записи;
- не менять обычную защиту поля из-за теневого результата;
- использовать псевдонимный ID поля, а точное местоположение хранить в защищённом реестре;
- фиксировать сорт, дату/интервал посадки, BBCH, обработки и полив, если они известны; неизвестное оставлять `null`.

## Для каждого слота наблюдения

Сохранить `observed_at`, подтверждённое время публикации/регистрации при наличии, `first_seen_at`, отдельные `retrieval_started_at` и `retrieval_completed_at`, `ingested_at`, источник и целевой организм. Для планового визита дополнительно записать время создания плана, причину визита, был ли план создан до решения, роль наблюдателя и объём осмотра. Статус визита хранить отдельно от результата поиска.

Допустимые исходные категории:

- `positive` — целевой организм подтверждён;
- `target_specific_negative` — проведён целевой осмотр, признаков фитофтороза не найдено;
- `generic_absent` — общая запись об отсутствии без достаточного подтверждения целевого поиска;
- `unassessed` — фитофтороз не оценивался;
- `conflict_unknown` — источники конфликтуют или вывод неизвестен;
- `not_visited` — визита не было.

Нельзя превращать отсутствие записи, `not_visited`, `unassessed` или общий комментарий «болезни нет» в подтверждённый отрицательный исход.

Для положительной записи по возможности указать метод подтверждения, дату первых симптомов и её точность: точная дата либо интервал. Позднее сообщение о прошлом осмотре добавляется новой версией и не меняет выпущенные решения задним числом. Исправление содержит `supersedes_observation_id` и причину; старую запись не удаляют.

Оценка качества подключает журнал наблюдений отдельно после созревания исходов. Decision engine не читает будущий исход, дату следующего визита или будущий состав событий.
"""


def _reproduce(commands: Sequence[str]) -> str:
    return f"""# Воспроизведение readiness-пакета

Запуск выполняется из корня репозитория в окружении с зависимостями проекта. Он не требует сети для offline-демонстрации и не отправляет уведомления.

## Проверка тестов

```bash
python -m pytest -q tests/test_shadow_storage.py tests/test_shadow_sources.py tests/test_shadow_registry.py tests/test_shadow_audit.py tests/test_shadow_engine.py tests/test_shadow_reporting.py tests/test_shadow_readiness_pipeline.py
```

## Команды контура

{chr(10).join(f'```bash{chr(10)}{command}{chr(10)}```' for command in commands)}

`technical-demo` использует только обезличенные синтетические ID и offline-fixtures. Даже успешное воспроизведение остаётся техническим тестом, а не `prospective_live` и не заявлением о качестве на VAAD.
"""


def _prospective_evaluation_plan(created_at: str) -> dict[str, Any]:
    return {
        "artifact_schema_version": "prospective-evaluation-plan-v1",
        "frozen_at_utc": created_at,
        "scope": "potato_late_blight_shadow_observation",
        "objective": "timely warning before the first recorded positive event",
        "primary_warning_window_calendar_days": {"minimum": 3, "maximum": 10},
        "population": {
            "kind": "dynamic_prospective_field_seasons",
            "entry": "actual_enrolment date from active field registry",
            "exit": "documented withdrawal, season close, or censoring",
            "fixed_historical_denominator": None,
            "historical_87_events_reused_as_new_population": False,
        },
        "outcome": {
            "event": "first recorded target-specific positive after enrolment",
            "repeat_positives_are_new_events": False,
            "positive_known_at_entry": "reported separately and not warnable",
            "missing_or_not_visited_is_negative": False,
            "unknown_categories": [
                "generic_absent",
                "unassessed",
                "conflict_unknown",
                "not_visited",
            ],
        },
        "information_boundary": {
            "decision_uses_inputs_ingested_no_later_than_actual_decision_time": True,
            "decision_engine_reads_outcome_log": False,
            "late_correction_changes_past_decision": False,
            "future_forecast_is_scoring_input": False,
        },
        "maturity_and_censoring": {
            "event_assessment_requires_follow_up_through_window_end": True,
            "withdrawal_or_missing_follow_up": "censored_or_not_scorable",
            "registration_delay": "reported separately from symptom interval when available",
            "analysis_date": None,
            "analysis_date_rule": "must be fixed before inspecting mature new outcomes",
        },
        "metrics": {
            "event": ["timely_first_events", "eligible_first_events", "event_recall"],
            "burden": [
                "virtual_messages",
                "messages_per_30_observed_field_days",
                "active_alarm_days",
                "active_alarm_fraction",
                "soft_budget_exceedances",
            ],
            "availability": [
                "scheduled_slots",
                "completed_slots",
                "missed_or_late_slots",
                "compatible_weather_fraction",
                "c0_fallback_fraction",
                "effective_c0_fraction",
                "abstention_fraction",
            ],
        },
        "analysis_governance": {
            "repeated_winner_selection_on_accumulating_outcomes": False,
            "models_and_policies_remain_frozen": True,
            "predeclared_analysis_point_required": True,
            "shadow_predictions_do_not_change_field_protection": True,
        },
    }


def _schemas() -> dict[str, dict[str, Any]]:
    uri = "https://json-schema.org/draft/2020-12/schema"
    timestamp = {"type": "string", "format": "date-time"}
    nullable_timestamp = {"oneOf": [timestamp, {"type": "null"}]}
    weather = {
        "$schema": uri,
        "$id": "urn:agro-phenology:shadow:weather-retrieval:v1",
        "title": "Immutable weather retrieval record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "retrieval_id", "source", "product", "model", "location_ref",
            "content_sha256", "raw_archive_ref", "retrieval_started_at_utc",
            "retrieval_completed_at_utc", "first_seen_at_utc", "ingested_at_utc",
            "valid_start_utc", "availability_evidence", "use_mode",
        ],
        "properties": {
            "retrieval_id": {"type": "string", "minLength": 1},
            "source": {"type": "string", "minLength": 1},
            "product": {"type": "string", "minLength": 1},
            "model": {"type": "string", "minLength": 1},
            "provider_version": {"type": ["string", "null"]},
            "endpoint": {"type": "string"},
            "location_ref": {"type": "string", "minLength": 1},
            "request_parameters_sanitised": {"type": "object"},
            "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "raw_archive_ref": {"type": "string", "minLength": 1},
            "run_initialization_at_utc": nullable_timestamp,
            "provider_published_at_utc": nullable_timestamp,
            "retrieval_started_at_utc": timestamp,
            "retrieval_completed_at_utc": timestamp,
            "first_seen_at_utc": timestamp,
            "ingested_at_utc": timestamp,
            "valid_start_utc": timestamp,
            "valid_end_utc": nullable_timestamp,
            "availability_evidence": {
                "enum": ["actual_retrieval", "confirmed_publication", "assumption", "unknown"]
            },
            "grid": {"type": ["object", "null"]},
            "timezone": {"type": ["string", "null"]},
            "units": {"type": "object", "additionalProperties": {"type": "string"}},
            "use_mode": {"enum": ["frozen_scoring", "archive_only", "delayed_reference"]},
            "compatibility_status": {
                "enum": ["compatible", "incompatible", "late", "unknown", "requires_separate_source_bridge_study"]
            },
        },
    }
    field_registry = {
        "$schema": uri,
        "$id": "urn:agro-phenology:shadow:active-field-registry:v1",
        "title": "Active field registry without direct coordinates",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "registry_id",
            "registry_purpose",
            "generated_at_utc",
            "available_at_utc",
            "ingested_at_utc",
            "timezone",
            "historical_coordinates_used",
            "fields",
        ],
        "properties": {
            "schema_version": {"const": "1.0.0"},
            "registry_id": {"type": "string", "minLength": 1},
            "registry_purpose": {
                "enum": ["prospective_active_fields", "technical_demo_synthetic"]
            },
            "generated_at_utc": timestamp,
            "available_at_utc": timestamp,
            "ingested_at_utc": timestamp,
            "timezone": {"const": "Europe/Riga"},
            "historical_coordinates_used": {"const": False},
            "field_registry_content_sha256": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "field_pseudo_id", "season_id", "field_season", "season",
                        "enrolled_from", "enrolled_until", "crop", "region_code",
                        "weather_location_ref", "status",
                    ],
                    "properties": {
                        "field_pseudo_id": {"type": "string", "minLength": 8},
                        "season_id": {"type": "string", "minLength": 8},
                        "field_season": {"type": "string", "minLength": 8},
                        "season": {"type": "integer", "minimum": 2026},
                        "enrolled_from": {"type": "string", "format": "date"},
                        "enrolled_until": {"type": ["string", "null"], "format": "date"},
                        "crop": {"const": "potato"},
                        "region_code": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Coarse, non-coordinate operational region code",
                        },
                        "weather_location_ref": {"type": "string", "minLength": 1},
                        "status": {"enum": ["active", "paused", "closed"]},
                        "cultivar": {"type": ["string", "null"]},
                        "planting_date": {"type": ["string", "null"], "format": "date"},
                        "emergence_date": {"type": ["string", "null"], "format": "date"},
                        "irrigation_logging_capability": {
                            "enum": ["available", "unavailable", "unknown", None]
                        },
                        "registry_source": {"type": ["string", "null"]},
                        "source_available_at_utc": nullable_timestamp,
                    },
                },
            },
        },
    }
    observation = {
        "$schema": uri,
        "$id": "urn:agro-phenology:shadow:field-observation:v1",
        "title": "Versioned potato late-blight observation",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "observation_id",
            "observation_key",
            "information_role",
            "payload",
            "first_seen_at_utc",
            "retrieval_started_at_utc",
            "retrieval_completed_at_utc",
            "ingested_at_utc",
            "valid_from_utc",
        ],
        "properties": {
            "observation_id": {"type": "string", "minLength": 1},
            "observation_key": {"type": "string", "minLength": 1},
            "information_role": {"const": "outcome"},
            "source_retrieval_id": {"type": ["string", "null"]},
            "initialized_at_utc": nullable_timestamp,
            "published_at_utc": nullable_timestamp,
            "first_seen_at_utc": timestamp,
            "retrieval_started_at_utc": timestamp,
            "retrieval_completed_at_utc": timestamp,
            "retrieved_at_utc": {
                **timestamp,
                "description": "Legacy alias for retrieval_completed_at_utc",
            },
            "ingested_at_utc": timestamp,
            "valid_from_utc": timestamp,
            "valid_to_utc": nullable_timestamp,
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "field_season",
                    "observed_at_utc",
                    "visit_status",
                    "target_organism",
                    "outcome_category",
                ],
                "properties": {
                    "field_pseudo_id": {"type": ["string", "null"]},
                    "field_season": {"type": "string", "minLength": 1},
                    "observed_at_utc": timestamp,
                    "registered_published_at_utc": nullable_timestamp,
                    "result_available_at_utc": nullable_timestamp,
                    "supersedes_observation_id": {"type": ["string", "null"]},
                    "correction_reason": {"type": ["string", "null"]},
                    "visit_status": {"enum": ["visited", "not_visited", "unknown"]},
                    "visit_plan_created_at_utc": nullable_timestamp,
                    "visit_trigger": {
                        "enum": [
                            "preplanned_independent",
                            "routine_protection",
                            "score_triggered",
                            "ad_hoc",
                            "unknown",
                            None,
                        ]
                    },
                    "planned_before_decision": {"type": ["boolean", "null"]},
                    "observer_role": {"type": ["string", "null"]},
                    "examined_extent": {"type": ["object", "null"]},
                    "target_organism": {"const": "potato_late_blight"},
                    "outcome_category": {
                        "enum": ["positive", "target_specific_negative", "generic_absent", "unassessed", "conflict_unknown", "not_visited"]
                    },
                    "targeted_search": {"type": ["boolean", "null"]},
                    "confirmation_method": {"type": ["string", "null"]},
                    "first_symptom_date": {"type": ["string", "null"], "format": "date"},
                    "first_symptom_date_lower": {"type": ["string", "null"], "format": "date"},
                    "first_symptom_date_upper": {"type": ["string", "null"], "format": "date"},
                    "first_symptom_date_precision": {
                        "enum": ["exact_date", "bounded_interval", "unbounded", "unknown", None]
                    },
                    "censoring_status": {
                        "enum": [
                            "mature", "not_yet_mature", "withdrawn", "lost_to_follow_up",
                            "not_applicable", "unknown", None,
                        ]
                    },
                    "bbch": {"type": ["string", "number", "null"]},
                    "cultivar": {"type": ["string", "null"]},
                    "planting_date": {"type": ["string", "null"], "format": "date"},
                    "treatments": {"type": ["array", "null"], "items": {"type": "object"}},
                    "irrigation": {"type": ["array", "null"], "items": {"type": "object"}},
                    "source": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
            },
        },
    }
    decision = {
        "$schema": uri,
        "$id": "urn:agro-phenology:shadow:decision:v1",
        "title": "Persisted shadow decision payload without outcomes",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "registry_id",
            "registry_content_sha256",
            "field_pseudo_id",
            "season_id",
            "season",
            "issue_local_date",
            "scheduled_for_utc",
            "actual_decision_at_utc",
            "run_mode",
            "late_run",
            "model_id",
            "model_version",
            "policy",
            "input_observation_ids",
            "input_hashes",
            "input_status",
            "score_origin",
            "score_status",
            "score",
            "threshold",
            "decision_action",
            "decision_reason",
            "transition",
            "virtual_message_issued",
            "actually_sent",
            "delivery_attempted",
        ],
        "properties": {
            "schema_version": {"const": "1.0.0"},
            "registry_id": {"type": "string"},
            "registry_content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "field_registry_id": {"type": "string"},
            "field_registry_content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "field_pseudo_id": {"type": "string"},
            "season_id": {"type": "string"},
            "season": {"type": "integer"},
            "region_code": {"type": "string"},
            "issue_local_date": {"type": "string", "format": "date"},
            "forecast_window_start_local_date": {"type": "string", "format": "date"},
            "forecast_window_end_local_date": {"type": "string", "format": "date"},
            "scheduled_for_utc": timestamp,
            "actual_decision_at_utc": timestamp,
            "late_by_seconds": {"type": "number", "minimum": 0},
            "run_mode": {"enum": ["prospective_live", "retrospective_replay", "delayed_reference", "offline_technical_demo"]},
            "late_run": {"type": "boolean"},
            "service_active": {"type": "boolean"},
            "data_timezone": {"const": "Europe/Riga"},
            "model_id": {"enum": ["calendar_window", "C0", "C1", "C4", "C5", "C6_weather", "C6_calibration_control"]},
            "model_version": {"type": "string"},
            "actionable_class": {"const": 2},
            "class_order": {"type": "array", "prefixItems": [{"const": 0}, {"const": 1}, {"const": 2}], "items": False},
            "policy": {"type": "object"},
            "input_observation_ids": {"type": "array", "items": {"type": "string"}},
            "input_hashes": {
                "type": "object",
                "additionalProperties": {
                    "oneOf": [
                        {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        {"type": "null"},
                    ]
                },
            },
            "input_reason": {"type": "string"},
            "feature_snapshot": {
                "type": "object",
                "additionalProperties": {"type": ["number", "null"]},
            },
            "feature_snapshot_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "input_status": {"type": "string"},
            "weather_availability": {"type": "object"},
            "probabilities": {
                "oneOf": [
                    {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}},
                    {"type": "null"},
                ]
            },
            "score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "score_origin": {
                "enum": [
                    "calendar_window",
                    "calendar_features",
                    "exact_era5_episode",
                    "c0_fallback_missing_exact_era5",
                    "c0_alpha_zero_identity",
                    "unavailable_exact_era5_episode",
                ]
            },
            "score_status": {"enum": ["computed", "computed_c0_fallback", "abstained", "not_in_service"]},
            "alpha": {"type": ["number", "null"]},
            "alpha_zero_identity": {"type": "boolean"},
            "used_c0_fallback": {"type": "boolean"},
            "threshold": {"type": ["number", "null"]},
            "comparison_segment_id": {"type": ["integer", "string", "null"]},
            "transition": {
                "type": "object",
                "required": [
                    "score_comparison_segment_id",
                    "growth_reference_comparable",
                    "alarm_active",
                    "shadow_message_would_be_issued",
                ],
                "properties": {
                    "score_comparison_segment_id": {"type": ["integer", "string", "null"]},
                    "score_comparison_segment_started": {"type": "boolean"},
                    "growth_reference_comparable": {"type": "boolean"},
                    "growth_reference_status": {"type": "string"},
                    "previous_message_score": {"type": ["number", "null"]},
                    "previous_message_date": {"type": ["string", "null"], "format": "date"},
                    "previous_message_origin": {"type": ["string", "null"]},
                    "elapsed_calendar_days_since_message": {"type": ["integer", "null"]},
                    "logit_growth_from_previous_message": {"type": ["number", "null"]},
                    "growth_minimum_interval_satisfied": {"type": "boolean"},
                    "growth_override_used": {"type": "boolean"},
                    "ordinary_cooldown_satisfied": {"type": "boolean"},
                    "threshold_reached": {"type": "boolean"},
                    "suppressed_repeat": {"type": "boolean"},
                    "shadow_message_would_be_issued": {"type": "boolean"},
                    "message_kind": {"type": ["string", "null"]},
                    "action_reason": {"type": "string"},
                    "active_from": {"type": ["string", "null"], "format": "date"},
                    "active_through": {"type": ["string", "null"], "format": "date"},
                    "alarm_active": {"type": "boolean"},
                    "cumulative_messages_field_season": {"type": "integer", "minimum": 0},
                    "cumulative_alarm_days_field_season": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            "decision_action": {"enum": ["abstain", "message_candidate", "message_suppressed", "no_message", "not_in_service"]},
            "decision_reason": {"type": "string"},
            "message_id": {"type": ["string", "null"]},
            "virtual_message_issued": {"type": "boolean"},
            "rolling_soft_budget_monitor": {"type": "object"},
            "delivery_mode": {"const": "shadow"},
            "delivery_attempted": {"const": False},
            "actually_sent": {"const": False},
            "research_only": {"const": True},
        },
    }
    return {
        "weather_retrieval.schema.json": weather,
        "active_field_registry.schema.json": field_registry,
        "observation.schema.json": observation,
        "decision.schema.json": decision,
    }


_SENSITIVE_KEY = re.compile(
    r"(^|_)(latitude|longitude|coordinates?|location|exact_location|weather_location_ref|location_ref|weather_cell|field_uid|field_id|active_field_id|field_season|field_season_id|site_id|request_key|raw_body|raw_payload)($|_)",
    re.IGNORECASE,
)


def _sanitise_for_review(value: Any, *, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        if isinstance(value, list):
            return ["<обезличено>" for _ in value]
        return "<обезличено>"
    if isinstance(value, Mapping):
        return {str(child_key): _sanitise_for_review(child, key=str(child_key)) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_sanitise_for_review(child, key=key) for child in value]
    if isinstance(value, str):
        if value.startswith("/Users/") or value.startswith("/home/"):
            return "<локальный путь исключён>"
        value = re.sub(r"/Users/[^/]+/", "<локальный путь>/", value)
        value = re.sub(r"/home/[^/]+/", "<локальный путь>/", value)
        if re.search(r"(?:latitude|longitude|lat|lon)=[+-]?\d", value, re.IGNORECASE):
            return "<URL или параметры с координатами исключены>"
        return value
    return value


def _write_deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def _safe_relative_path(name: str, *, label: str) -> Path:
    """Return one portable relative artifact path without traversal."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label} name must be a non-empty relative path")
    posix = PurePosixPath(name.replace("\\", "/"))
    windows = PureWindowsPath(name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or any(part in {"", "."} for part in posix.parts)
    ):
        raise ValueError(f"unsafe {label} name: {name!r}")
    return Path(*posix.parts)


def _review_readme(created_at: str) -> str:
    return f"""# Обезличенный пакет shadow readiness

Сформирован `{created_at}`. Пакет предназначен для независимого обсуждения инженерной готовности. В нём нет моделей, raw-погодных ответов, точных координат, персональных данных и подробных полевых журналов.

Техническая демонстрация не является проспективным запуском и не доказывает качество модели. На дату проверки frozen ERA5 не обеспечивает данные до `issue_date−2`; C6 может выполнить только C0 fallback. Реальные уведомления не отправлялись, расписание не активировалось.

Содержимое: readiness-отчёт, decision record, машинные агрегаты, блокеры, обезличенные source manifest и registry, краткие результаты тестов.
"""


def generate_shadow_readiness_run(
    run_dir: str | Path,
    *,
    registry: Mapping[str, Any],
    source_compatibility_manifest: Mapping[str, Any],
    demo_summary: Mapping[str, Any],
    test_summary: Mapping[str, Any],
    blockers: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    created_at_utc: str | datetime | None = None,
    reproduction_commands: Sequence[str] | None = None,
    execution_context: Mapping[str, Any] | None = None,
    technical_demo_artifacts: Mapping[str, str | Path] | None = None,
    supplemental_json: Mapping[str, Any] | None = None,
) -> Path:
    """Create one versioned readiness result atomically and refuse overwrite.

    All inputs must already be aggregates or registries.  This function does
    not read model files or field data and does not execute the shadow engine.
    """
    destination = Path(run_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"readiness run already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    created_at = _utc_text(created_at_utc)
    commands = tuple(reproduction_commands or DEFAULT_REPRODUCTION_COMMANDS)
    if not commands or not all(isinstance(item, str) and item.strip() for item in commands):
        raise ValueError("reproduction_commands must contain non-empty strings")

    demo = deepcopy(dict(demo_summary))
    mode = str(
        demo.get(
            "mode",
            demo.get("run_mode", demo.get("demonstration_mode", "offline_technical_demo")),
        )
    )
    if mode not in {"offline_technical_demo", "retrospective_replay"}:
        raise ValueError("readiness generator accepts only an offline technical demo/replay")
    live_run_executed = bool(
        demo.get(
            "live_run_executed",
            demo.get("prospective_live_run", demo.get("prospective_live_run_performed", False)),
        )
    )
    notifications_sent = int(
        demo.get("real_notifications_sent", demo.get("notifications_sent", 0))
    )
    if live_run_executed:
        raise ValueError("live run cannot be represented as the offline readiness demo")
    if notifications_sent != 0:
        raise ValueError("readiness demo must not send real notifications")
    if bool(demo.get("schedule_activated", False)):
        raise ValueError("readiness demo must not activate a schedule")
    demo.setdefault("mode", mode)
    demo.setdefault("live_run_executed", live_run_executed)
    demo.setdefault("real_notifications_sent", notifications_sent)
    demo.setdefault("schedule_activated", False)
    if "network_status" not in demo and "network_used" in demo:
        demo["network_status"] = "used" if bool(demo["network_used"]) else "not_called"
    if "field_registry_status" not in demo and bool(demo.get("synthetic_inputs_only", False)):
        demo["field_registry_status"] = "synthetic_only"
    unique_decisions = demo.get("unique_decisions")
    if isinstance(unique_decisions, int) and unique_decisions > 0:
        if "computed_fraction" not in demo and isinstance(demo.get("abstentions"), int):
            demo["computed_fraction"] = (
                unique_decisions - int(demo["abstentions"])
            ) / unique_decisions
    if "weather_correction_fraction" not in demo and isinstance(
        demo.get("active_weather_fraction_in_c6"), (int, float)
    ):
        demo["weather_correction_fraction"] = float(
            demo["active_weather_fraction_in_c6"]
        )
    if "effective_c0_fraction" not in demo and isinstance(
        demo.get("effective_c0_fraction_in_c6"), (int, float)
    ):
        demo["effective_c0_fraction"] = float(
            demo["effective_c0_fraction_in_c6"]
        )

    registry_value = deepcopy(dict(registry))
    source_value = deepcopy(dict(source_compatibility_manifest))
    blocker_value = _normalise_blockers(blockers)
    tests_value = deepcopy(dict(test_summary))
    source_value.setdefault("artifact_schema_version", REPORT_SCHEMA_VERSION)
    source_value.setdefault("generated_at_utc", created_at)
    blocker_value.setdefault("artifact_schema_version", REPORT_SCHEMA_VERSION)
    blocker_value.setdefault("generated_at_utc", created_at)

    weather_live = _extract_bool(
        source_value,
        ("weather_branch_executable_today", "weather_branch_operational_today"),
        False,
    )
    readiness = {
        "artifact_schema_version": REPORT_SCHEMA_VERSION,
        "run_id": destination.name,
        "generated_at_utc": created_at,
        "stage": "prospective_shadow_readiness",
        "status": "partially_ready" if blocker_value["blockers"] else "ready_for_authorised_live_inputs",
        "technical_demo": {
            **demo,
            "mode": mode,
            "is_prospective": False,
            "quality_claim_allowed": False,
        },
        "prospective_live": {
            "executed": False,
            "real_notifications_sent": 0,
            "schedule_activated": False,
            "current_issue_slot_can_be_prospective": bool(
                source_value.get("shadow_start_boundary", {}).get(
                    "current_issue_slot_can_be_prospective", False
                )
            ),
            "first_honest_scheduled_slot_local": source_value.get(
                "shadow_start_boundary", {}
            ).get("first_honest_scheduled_slot_local"),
        },
        "weather_branch": {
            "executable_today": weather_live,
            "frozen_cutoff": "issue_date-2 calendar days",
            "documented_era5_delay": "approximately 5 days",
            "decision": "compatible" if weather_live else "blocked_by_freshness",
            "c6_behavior": "weather_score" if weather_live else "c0_fallback_only",
            "weather_effect_tested": bool(weather_live and demo.get("weather_correction_applied", False)),
            "source_substitution_used": False,
        },
        "frozen_research_semantics": {
            "warning_window_days": [3, 10],
            "endpoint": "first_recorded_positive_event_after_enrolment",
            "missing_visit_is_negative": False,
            "retraining_or_retuning_performed": False,
        },
        "blocker_count": len(blocker_value["blockers"]),
        "test_status": tests_value.get("status", "unknown"),
    }
    evaluation_plan = _prospective_evaluation_plan(created_at)

    try:
        temporary.mkdir(parents=False, exist_ok=False)
        _write_json(temporary / "shadow_registry.json", registry_value)
        _write_json(temporary / "source_compatibility_manifest.json", source_value)
        _write_json(temporary / "blockers.json", blocker_value)
        _write_json(temporary / "readiness_summary.json", readiness)
        _write_json(temporary / "technical_demo_summary.json", demo)
        _write_json(temporary / "test_results.json", tests_value)
        _write_json(temporary / "prospective_evaluation_plan.json", evaluation_plan)
        _write_text(
            temporary / "operational_readiness_ru.md",
            _report(
                run_id=destination.name,
                created_at=created_at,
                registry=registry_value,
                source_manifest=source_value,
                demo=demo,
                tests=tests_value,
                blockers=blocker_value,
            ),
        )
        _write_text(temporary / "decision_record.md", _decision_record(created_at))
        _write_text(temporary / "RUNBOOK.md", _runbook(commands))
        _write_text(
            temporary / "field_observation_instructions_ru.md", _field_instructions()
        )
        _write_text(temporary / "REPRODUCE.md", _reproduce(commands))
        for name, schema in _schemas().items():
            _write_json(temporary / "schemas" / name, schema)

        for raw_name, source_path in (technical_demo_artifacts or {}).items():
            relative = _safe_relative_path(raw_name, label="technical demo artifact")
            source = Path(source_path)
            if not source.is_file():
                raise FileNotFoundError(source)
            target = temporary / "technical_demo" / relative
            if target.exists():
                raise ValueError(f"duplicate technical demo artifact: {raw_name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        reserved = {
            "execution_manifest.json",
            "shadow_registry.json",
            "source_compatibility_manifest.json",
            "blockers.json",
            "readiness_summary.json",
            "technical_demo_summary.json",
            "test_results.json",
            "prospective_evaluation_plan.json",
        }
        for raw_name, payload in (supplemental_json or {}).items():
            relative = _safe_relative_path(raw_name, label="supplemental JSON artifact")
            if relative.suffix.lower() != ".json":
                raise ValueError("supplemental JSON artifact names must end in .json")
            relative_text = relative.as_posix()
            if relative_text in reserved or relative.parts[0] in {"review_package", "technical_demo", "schemas"}:
                raise ValueError(f"reserved supplemental JSON artifact name: {raw_name!r}")
            target = temporary / relative
            if target.exists():
                raise ValueError(f"duplicate supplemental JSON artifact: {raw_name!r}")
            _write_json(target, deepcopy(payload))

        review = temporary / "review_package"
        review.mkdir()
        _write_text(review / "README.md", _review_readme(created_at))
        review_registry = _sanitise_for_review(registry_value)
        review_source = _sanitise_for_review(source_value)
        review_blockers = _sanitise_for_review(blocker_value)
        review_demo = _sanitise_for_review(demo)
        review_tests = _sanitise_for_review(tests_value)
        _write_text(
            review / "operational_readiness_ru.md",
            _report(
                run_id=destination.name,
                created_at=created_at,
                registry=review_registry,
                source_manifest=review_source,
                demo=review_demo,
                tests=review_tests,
                blockers=review_blockers,
            ),
        )
        shutil.copy2(temporary / "decision_record.md", review / "decision_record.md")
        _write_json(review / "source_compatibility_manifest.json", review_source)
        _write_json(review / "shadow_registry.json", review_registry)
        _write_json(review / "blockers.json", review_blockers)
        _write_json(review / "readiness_summary.json", _sanitise_for_review(readiness))
        _write_json(review / "technical_demo_summary.json", review_demo)
        _write_json(review / "test_results.json", review_tests)
        _write_json(
            review / "prospective_evaluation_plan.json",
            _sanitise_for_review(evaluation_plan),
        )
        _write_deterministic_zip(review, temporary / "review_package.zip")

        output_hashes: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in temporary.rglob("*") if item.is_file()):
            relative = path.relative_to(temporary).as_posix()
            if relative == "execution_manifest.json":
                continue
            output_hashes[relative] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        manifest = {
            "artifact_schema_version": REPORT_SCHEMA_VERSION,
            "run_id": destination.name,
            "stage": "prospective_shadow_readiness",
            "status": readiness["status"],
            "completed_at_utc": created_at,
            "output_hashes": output_hashes,
            "manifest_scope_excludes": ["execution_manifest.json"],
            "execution_context": deepcopy(dict(execution_context or {})),
            "side_effects": {
                "network_requests": 0,
                "real_notifications_sent": 0,
                "schedule_activated": False,
                "models_trained_or_tuned": 0,
            },
            "trust_boundary": (
                "Local hashes and SQLite append-only guards detect accidental changes; "
                "without an external timestamp or immutable store they do not prove creation time."
            ),
        }
        _write_json(temporary / "execution_manifest.json", manifest)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "DEFAULT_REPRODUCTION_COMMANDS",
    "REPORT_SCHEMA_VERSION",
    "generate_shadow_readiness_run",
]
