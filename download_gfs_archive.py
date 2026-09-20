#!/usr/bin/env python3
"""Download the frozen retrospective GDEX GFS request plan safely.

The script is deliberately a thin overnight runner around the audited GFS
builder.  Raw responses and per-date checkpoints are the resume boundary: a
second invocation validates and skips completed work, then retries only what is
missing or incomplete.  It never starts model training.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping, Sequence
import uuid


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import numpy as np
    import pandas as pd

    from agro_phenology.early_warning_gfs_experiment import (
        load_forecast_feature_table,
    )
    from agro_phenology.gfs_archive import FORECAST_FEATURE_COLUMNS
    from agro_phenology.gfs_feature_builder import (
        CHECKPOINT_SCHEMA_VERSION,
        DEFAULT_CONTRACT,
        DEFAULT_PARENT_DECISIONS,
        FAST_NCSS_ACCESS_MODE,
        FEATURE_FILE_NAME,
        PLAN_FILE_NAME,
        PLAN_MANIFEST_FILE_NAME,
        PLAN_SCHEMA_VERSION,
        SOURCE_MANIFEST_FILE_NAME,
        FastNCSSCandidateClient,
        _checkpoint_path,
        assemble_feature_table,
        assert_no_private_or_target_columns,
        fetch_request_plan,
        requests_for_plan_row,
        sha256_file,
        write_request_plan,
    )
except ImportError as error:  # pragma: no cover - exercised by the real CLI only
    raise SystemExit(
        "Не найдены зависимости проекта. Запустите скрипт так:\n"
        f"  {REPO_ROOT / '.venv/bin/python'} {Path(__file__).resolve()}\n"
        f"Исходная ошибка: {error}"
    ) from error


DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "results/late_blight_early_warning/20260910_gfs_archive_inputs_v2"
)
LOG_FILE_NAME = "download.log"
SUMMARY_FILE_NAME = "download_summary.json"
LOCK_FILE_NAME = "download.lock"
QUARANTINE_DIR_NAME = "download_quarantine"
SOURCE_UNAVAILABILITY_MANIFEST_FILE_NAME = (
    "gfs_source_unavailability_manifest_v1.json"
)
SOURCE_UNAVAILABILITY_SCHEMA_VERSION = "gfs_source_unavailability_manifest_v1"
SOURCE_UNAVAILABILITY_RULE_ID = "gdex_terminal_for_frozen_access_method_v1"
PLAN_COMPATIBILITY_MANIFEST_FILE_NAME = (
    "gfs_request_plan_manifest_compatibility_v1.json"
)
PLAN_COMPATIBILITY_SCHEMA_VERSION = "gfs_request_plan_manifest_compatibility_v1"
MIN_SOURCE_UNAVAILABILITY_ATTEMPTS = 3

_HTTP_404_FILE_NOT_FOUND_SUFFIX = ": FileNotFound: No such file or directory"
_NO_PRECIPITATION_CANDIDATE_ERROR = (
    "neither frozen precipitation candidate exists in archive file"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "неизвестно"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} ч {minutes:02d} мин"
    if minutes:
        return f"{minutes} мин {secs:02d} с"
    return f"{secs} с"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RunLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, message: str) -> None:
        line = f"{utc_text()}  {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class StopController:
    """Turn the first signal into a graceful checkpoint boundary."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None
        self._count = 0

    def handle(self, signum: int, _frame: Any) -> None:
        self._count += 1
        self.requested = True
        self.signal_number = signum
        if self._count == 1:
            print(
                "\nПолучен сигнал остановки. Завершаю текущую дату и сохраняю "
                "состояние; повторный запуск продолжит загрузку.",
                flush=True,
            )
            return
        raise KeyboardInterrupt


class CheckpointProgress:
    """Show live progress while the builder downloads one 28-file checkpoint."""

    def __init__(
        self,
        *,
        run_root: Path,
        row: Mapping[str, Any],
        completed_checkpoints: int,
        total_checkpoints: int,
        total_snapshot_files: int,
        eta_seconds: float | None,
    ) -> None:
        self.run_root = run_root
        self.row = row
        self.completed_checkpoints = completed_checkpoints
        self.total_checkpoints = total_checkpoints
        self.total_snapshot_files = total_snapshot_files
        self.eta_seconds = eta_seconds
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.is_terminal = sys.stdout.isatty()
        self.interval_seconds = 5.0 if self.is_terminal else 30.0
        self._last_width = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _cached_count(self) -> int:
        client = FastNCSSCandidateClient(self.run_root / "gfs_raw_cache")
        count = 0
        for request in requests_for_plan_row(self.row):
            directory = client._cache_dir(request)
            if (directory / "subset.nc").is_file() and (
                directory / "provenance.json"
            ).is_file():
                count += 1
        return count

    @staticmethod
    def _bar(done: int, total: int, width: int = 20) -> str:
        filled = min(width, max(0, round(width * done / max(1, total))))
        return "█" * filled + "░" * (width - filled)

    def _render(self) -> None:
        requested = int(self.row["requested_snapshot_count"])
        cached = self._cached_count()
        elapsed = time.monotonic() - self.started
        fractional_done = self.completed_checkpoints + cached / max(1, requested)
        overall_percent = 100.0 * fractional_done / max(1, self.total_checkpoints)
        archive_done = self.completed_checkpoints * requested + cached
        eta = (
            format_duration(max(0.0, self.eta_seconds - elapsed))
            if self.eta_seconds is not None
            else "после первого checkpoint"
        )
        line = (
            f"{self.row['checkpoint_id']}  "
            f"[{self._bar(cached, requested)}] {cached}/{requested} файлов  |  "
            f"checkpoint {self.completed_checkpoints}/{self.total_checkpoints}  |  "
            f"архив {archive_done}/{self.total_snapshot_files} "
            f"({overall_percent:.2f}%)  |  {format_duration(elapsed)}  |  ETA {eta}"
        )
        if self.is_terminal:
            padding = " " * max(0, self._last_width - len(line))
            print(f"\r{line}{padding}", end="", flush=True)
            self._last_width = len(line)
        else:
            print(line, flush=True)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._render()
            except Exception as error:
                print(f"индикатор прогресса недоступен: {error}", flush=True)
                return
            self.stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "CheckpointProgress":
        self.thread.start()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.is_terminal:
            print("\r" + " " * self._last_width + "\r", end="", flush=True)


@contextmanager
def exclusive_run_lock(path: Path) -> Iterator[None]:
    """Use a kernel lock; a stale lock file cannot block a later run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Уже запущен другой загрузчик для {path.parent}."
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started_at_utc={utc_text()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def start_caffeinate(enabled: bool, log: RunLog) -> subprocess.Popen[bytes] | None:
    if not enabled or sys.platform != "darwin":
        return None
    executable = Path("/usr/bin/caffeinate")
    if not executable.is_file():
        log.write("caffeinate не найден; автоматическая защита от сна недоступна")
        return None
    process = subprocess.Popen(
        [str(executable), "-i", "-w", str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log.write("macOS будет удерживаться от автоматического сна до завершения скрипта")
    return process


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ожидался JSON-объект: {path}")
    return payload


def _manifest_privacy_is_local(manifest: Mapping[str, Any]) -> bool:
    current = (
        manifest.get("contains_field_ids") is False
        and manifest.get("contains_outcomes") is False
        and manifest.get("contains_coarse_grid_coordinates") is True
        and manifest.get("distribution") == "local_private_do_not_publish"
    )
    legacy = (
        manifest.get("contains_field_ids_or_outcomes") is False
        and manifest.get("privacy") == "local_project_grid_selection_do_not_publish"
    )
    return bool(current or legacy)


def ensure_and_validate_plan(run_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a missing plan once, or validate the exact frozen existing pair."""

    plan_path = run_root / PLAN_FILE_NAME
    manifest_path = run_root / PLAN_MANIFEST_FILE_NAME
    if plan_path.exists() != manifest_path.exists():
        raise RuntimeError(
            "План или его манифест отсутствует. Неполную пару нельзя угадывать: "
            f"{plan_path}, {manifest_path}"
        )
    if not plan_path.exists():
        write_request_plan(
            run_root=run_root,
            parent_decisions_path=DEFAULT_PARENT_DECISIONS,
            contract_path=DEFAULT_CONTRACT,
        )

    manifest = _read_json(manifest_path)
    plan = pd.read_parquet(plan_path)
    if manifest.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("версия замороженного плана не совпадает с кодом")
    checks = (
        (manifest.get("request_plan_sha256"), sha256_file(plan_path), "хеш плана"),
        (
            manifest.get("contract_sha256"),
            sha256_file(DEFAULT_CONTRACT),
            "хеш evaluation contract",
        ),
        (
            manifest.get("parent_daily_decisions_sha256"),
            sha256_file(DEFAULT_PARENT_DECISIONS),
            "хеш календаря решений v3",
        ),
    )
    for declared, actual, label in checks:
        if declared != actual:
            raise ValueError(f"{label} изменился: {declared!r} != {actual!r}")
    if not _manifest_privacy_is_local(manifest):
        raise ValueError("манифест плана не фиксирует локальный приватный режим")
    required = {
        "plan_schema_version",
        "checkpoint_id",
        "issue_date",
        "issue_year",
        "availability_scenario_hours",
        "required_gfs_cell_ids_json",
        "required_gfs_cell_count",
        "requested_snapshot_count",
        "source_dataset_id",
    }
    missing = sorted(required.difference(plan.columns))
    if missing:
        raise ValueError(f"в плане отсутствуют столбцы: {missing}")
    if plan.empty or plan["checkpoint_id"].astype(str).duplicated().any():
        raise ValueError("план пуст или содержит повторяющиеся checkpoint_id")
    if not plan["plan_schema_version"].astype(str).eq(PLAN_SCHEMA_VERSION).all():
        raise ValueError("строки плана имеют другую версию схемы")
    if not plan["source_dataset_id"].astype(str).eq("d084001").all():
        raise ValueError("план содержит источник, отличный от GDEX d084001")
    scenarios = sorted(
        pd.to_numeric(plan["availability_scenario_hours"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if scenarios != [4, 7] or sorted(manifest.get("scenario_hours", [])) != scenarios:
        raise ValueError(f"ожидались только сценарии 4 и 7 часов, получено {scenarios}")
    if int(manifest.get("checkpoints", -1)) != len(plan):
        raise ValueError("число checkpoint в манифесте не совпадает с планом")
    planned_requests = int(
        pd.to_numeric(plan["requested_snapshot_count"], errors="raise").sum()
    )
    if int(manifest.get("archive_subset_requests", -1)) != planned_requests:
        raise ValueError("число файлов в манифесте не совпадает с планом")
    assert_no_private_or_target_columns(plan)
    return (
        plan.sort_values(
            ["issue_date", "availability_scenario_hours"], kind="mergesort"
        ).reset_index(drop=True),
        manifest,
    )


def ensure_plan_compatibility_manifest(
    run_root: Path,
    plan: pd.DataFrame,
    frozen_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Create an immutable current-key view of the frozen legacy manifest.

    The original v2 manifest is never rewritten.  The compatibility file keeps
    all frozen hashes/counts, adds the explicit privacy keys required by the
    model bundle validator, and records the exact legacy bytes it represents.
    """

    legacy_path = run_root / PLAN_MANIFEST_FILE_NAME
    path = run_root / PLAN_COMPATIBILITY_MANIFEST_FILE_NAME
    stable = dict(frozen_manifest)
    # The compatibility file has its own creation timestamp.  The legacy
    # timestamp remains recoverable from the byte-identical legacy manifest and
    # its hash, so it must not participate in the stable payload comparison.
    stable.pop("created_at_utc", None)
    stable.update(
        {
            "schema_version": PLAN_SCHEMA_VERSION,
            "compatibility_schema_version": PLAN_COMPATIBILITY_SCHEMA_VERSION,
            "manifest_role": "current_privacy_keys_view_of_frozen_legacy_manifest",
            "legacy_manifest_path": str(legacy_path.resolve()),
            "legacy_manifest_sha256": sha256_file(legacy_path),
            "request_plan_path": str((run_root / PLAN_FILE_NAME).resolve()),
            "request_plan_sha256": sha256_file(run_root / PLAN_FILE_NAME),
            "checkpoints": int(len(plan)),
            "archive_subset_requests": int(
                pd.to_numeric(
                    plan["requested_snapshot_count"], errors="raise"
                ).sum()
            ),
            "scenario_hours": sorted(
                pd.to_numeric(
                    plan["availability_scenario_hours"], errors="raise"
                )
                .astype(int)
                .unique()
                .tolist()
            ),
            "contains_field_ids": False,
            "contains_outcomes": False,
            "contains_coarse_grid_coordinates": True,
            "distribution": "local_private_do_not_publish",
        }
    )

    if path.exists():
        current = _read_json(path)
        expected = dict(stable)
        current_without_time = {
            key: value for key, value in current.items() if key != "created_at_utc"
        }
        if current_without_time != expected:
            raise ValueError(
                "versioned compatibility manifest differs from the frozen plan; "
                f"refusing to replace {path}"
            )
        return path, current

    payload = {**stable, "created_at_utc": utc_text()}
    atomic_json(path, payload)
    return path, payload


@dataclass(frozen=True)
class CheckpointInspection:
    state: str  # missing, incomplete, complete, invalid
    reason: str


def validate_raw_cache_for_checkpoint(
    run_root: Path, row: Mapping[str, Any], declared_hashes: Sequence[str]
) -> None:
    """Verify that every response behind a completed checkpoint is still present."""

    client = FastNCSSCandidateClient(run_root / "gfs_raw_cache")
    actual_hashes: list[str] = []
    for request in requests_for_plan_row(row):
        artifact = client._load_cached(request)
        if artifact is None:
            raise FileNotFoundError(
                f"нет raw cache для {request.archive_path} и bbox этого checkpoint"
            )
        actual_hashes.append(artifact.data_sha256)
    if sorted(set(actual_hashes)) != sorted(set(declared_hashes)):
        raise ValueError("хеши raw cache не совпадают с checkpoint")


def _parse_uniform_json_list(frame: pd.DataFrame, column: str) -> list[Any]:
    values = frame[column].astype(str).unique().tolist()
    if len(values) != 1:
        raise ValueError(f"{column} различается между GFS-ячейками checkpoint")
    parsed = json.loads(values[0])
    if not isinstance(parsed, list):
        raise ValueError(f"{column} должен содержать JSON-массив")
    return parsed


def validate_incomplete_checkpoint(
    frame: pd.DataFrame,
    row: Mapping[str, Any],
    *,
    run_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Validate a fetched-but-source-unavailable checkpoint without accepting it."""

    requested = int(row["requested_snapshot_count"])
    retrieved = pd.to_numeric(frame["retrieved_snapshot_count"], errors="raise")
    failed = pd.to_numeric(frame["failed_snapshot_count"], errors="raise")
    if retrieved.nunique() != 1 or failed.nunique() != 1:
        raise ValueError("счётчики incomplete checkpoint различаются между строками")
    retrieved_count = int(retrieved.iloc[0])
    failed_count = int(failed.iloc[0])
    if failed_count < 1 or retrieved_count < 0 or retrieved_count + failed_count != requested:
        raise ValueError("retrieved + failed не совпадает с requested")
    if frame["checkpoint_complete"].fillna(False).astype(bool).any():
        raise ValueError("incomplete checkpoint смешивает complete=true/false")
    if frame["forecast_available"].fillna(False).astype(bool).any():
        raise ValueError("incomplete checkpoint ошибочно помечен forecast_available")

    failures = _parse_uniform_json_list(frame, "failed_requests_json")
    if len(failures) != failed_count or not all(
        isinstance(item, dict) for item in failures
    ):
        raise ValueError("список ошибок не совпадает с failed_snapshot_count")
    request_by_lead = {
        request.lead_hours: request for request in requests_for_plan_row(row)
    }
    failure_leads: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for raw in failures:
        if not {"lead_hours", "archive_path", "error_type", "error"}.issubset(raw):
            raise ValueError("ошибка source gap не содержит обязательные поля")
        lead = int(raw["lead_hours"])
        if lead in failure_leads or lead not in request_by_lead:
            raise ValueError("ошибка содержит повторный или незапланированный lead")
        failure_leads.add(lead)
        expected_path = request_by_lead[lead].archive_path
        if str(raw["archive_path"]) != expected_path:
            raise ValueError("archive_path ошибки не совпадает с замороженным запросом")
        normalized.append(
            {
                "lead_hours": lead,
                "archive_path": expected_path,
                "error_type": str(raw["error_type"]),
                "error": str(raw["error"]),
            }
        )

    hashes = _parse_uniform_json_list(frame, "source_hashes_json")
    if len(hashes) != retrieved_count or any(
        not isinstance(item, str)
        or re.fullmatch(r"[0-9a-f]{64}", item.lower()) is None
        for item in hashes
    ):
        raise ValueError("хеши полученных raw-ответов не совпадают с retrieved")

    if run_root is not None:
        client = FastNCSSCandidateClient(run_root / "gfs_raw_cache")
        actual_hashes: list[str] = []
        for lead, request in request_by_lead.items():
            artifact = client._load_cached(request)
            if lead in failure_leads:
                if artifact is not None:
                    raise ValueError("source gap уже имеет валидный raw cache")
            else:
                if artifact is None:
                    raise FileNotFoundError(
                        f"нет успешного raw cache для {request.archive_path}"
                    )
                actual_hashes.append(artifact.data_sha256)
        if sorted(set(actual_hashes)) != sorted(set(str(item) for item in hashes)):
            raise ValueError("хеши частичного raw cache не совпадают с checkpoint")
    return sorted(normalized, key=lambda item: int(item["lead_hours"]))


def inspect_checkpoint(
    path: Path,
    row: Mapping[str, Any],
    *,
    run_root: Path | None = None,
) -> CheckpointInspection:
    if not path.is_file():
        return CheckpointInspection("missing", "файл отсутствует")
    try:
        frame = pd.read_parquet(path)
        required_columns = {
            "checkpoint_schema_version",
            "checkpoint_id",
            "gfs_cell_id",
            "issue_date",
            "availability_scenario_hours",
            "forecast_available",
            "checkpoint_complete",
            "source_access_mode",
            "requested_snapshot_count",
            "retrieved_snapshot_count",
            "failed_snapshot_count",
            "failed_requests_json",
            "source_hashes_json",
            *FORECAST_FEATURE_COLUMNS,
        }
        missing = sorted(required_columns.difference(frame.columns))
        if frame.empty or missing:
            raise ValueError(f"пустой checkpoint или нет столбцов {missing}")
        expected_cells = set(json.loads(str(row["required_gfs_cell_ids_json"])))
        actual_cells = set(frame["gfs_cell_id"].astype(str))
        if actual_cells != expected_cells or frame["gfs_cell_id"].astype(str).duplicated().any():
            raise ValueError("набор GFS-ячеек не совпадает с планом")
        if not frame["checkpoint_schema_version"].astype(str).eq(
            CHECKPOINT_SCHEMA_VERSION
        ).all():
            raise ValueError("другая версия схемы checkpoint")
        if not frame["checkpoint_id"].astype(str).eq(str(row["checkpoint_id"])).all():
            raise ValueError("checkpoint_id не совпадает с планом")
        if set(frame["source_access_mode"].astype(str)) != {FAST_NCSS_ACCESS_MODE}:
            raise ValueError("неожиданный режим доступа к источнику")
        issue_date = pd.Timestamp(row["issue_date"]).normalize()
        if not pd.to_datetime(frame["issue_date"], errors="raise").dt.normalize().eq(
            issue_date
        ).all():
            raise ValueError("issue_date не совпадает с планом")
        scenario = int(row["availability_scenario_hours"])
        if not pd.to_numeric(
            frame["availability_scenario_hours"], errors="raise"
        ).astype(int).eq(scenario).all():
            raise ValueError("сценарий задержки не совпадает с планом")
        requested = int(row["requested_snapshot_count"])
        if not pd.to_numeric(frame["requested_snapshot_count"], errors="raise").eq(
            requested
        ).all():
            raise ValueError("число запросов не совпадает с планом")

        complete = frame["checkpoint_complete"].fillna(False).astype(bool)
        available = frame["forecast_available"].fillna(False).astype(bool)
        failed = pd.to_numeric(frame["failed_snapshot_count"], errors="raise")
        retrieved = pd.to_numeric(frame["retrieved_snapshot_count"], errors="raise")
        if not complete.all():
            validate_incomplete_checkpoint(frame, row, run_root=run_root)
            return CheckpointInspection(
                "incomplete",
                f"получено {int(retrieved.min())}/{requested}, ошибок {int(failed.max())}",
            )
        if not available.all() or not failed.eq(0).all() or not retrieved.eq(requested).all():
            raise ValueError("флаг complete противоречит числу полученных файлов")
        if not frame["failed_requests_json"].astype(str).eq("[]").all():
            raise ValueError("complete checkpoint содержит список ошибок")
        feature_values = frame[list(FORECAST_FEATURE_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(feature_values.to_numpy(dtype=float)).all():
            raise ValueError("complete checkpoint содержит нечисловые признаки")
        declared_hashes: list[str] | None = None
        for raw_hashes in frame["source_hashes_json"].astype(str):
            parsed = json.loads(raw_hashes)
            if (
                not isinstance(parsed, list)
                or len(parsed) != requested
                or any(
                    not isinstance(item, str)
                    or len(item) != 64
                    or any(char not in "0123456789abcdef" for char in item.lower())
                    for item in parsed
                )
            ):
                raise ValueError("некорректный список хешей исходных ответов")
            if declared_hashes is None:
                declared_hashes = parsed
            elif parsed != declared_hashes:
                raise ValueError("GFS-ячейки checkpoint ссылаются на разные raw hashes")
        if run_root is not None:
            if declared_hashes is None:
                raise ValueError("checkpoint не содержит raw hashes")
            validate_raw_cache_for_checkpoint(run_root, row, declared_hashes)
        return CheckpointInspection("complete", "проверен")
    except Exception as error:
        return CheckpointInspection("invalid", f"{type(error).__name__}: {error}")


def _quarantine_destination(run_root: Path, source: Path, category: str) -> Path:
    stamp = utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    try:
        relative = source.resolve().relative_to(run_root.resolve())
    except ValueError:
        relative = Path(source.name)
    destination = run_root / QUARANTINE_DIR_NAME / stamp / category / relative
    if destination.exists():
        destination = destination.with_name(
            f"{destination.name}.{uuid.uuid4().hex[:8]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def quarantine_path(run_root: Path, source: Path, category: str, log: RunLog) -> Path:
    destination = _quarantine_destination(run_root, source, category)
    shutil.move(str(source), str(destination))
    log.write(f"повреждённый/оборванный файл сохранён в quarantine: {destination}")
    return destination


def recover_interrupted_raw_writes(run_root: Path, log: RunLog) -> int:
    """Preserve half-written cache entries and let the normal fetch recreate them."""

    cache_root = run_root / "gfs_raw_cache"
    if not cache_root.exists():
        return 0
    moved = 0
    candidates = list(cache_root.rglob(".candidate.*"))
    leaf_directories = {
        path.parent for name in ("subset.nc", "provenance.json") for path in cache_root.rglob(name)
    }
    for directory in sorted(leaf_directories):
        data_exists = (directory / "subset.nc").is_file()
        provenance_exists = (directory / "provenance.json").is_file()
        if data_exists == provenance_exists:
            continue
        if directory.exists():
            quarantine_path(run_root, directory, "raw_incomplete", log)
            moved += 1
    for candidate in candidates:
        if candidate.exists():
            quarantine_path(run_root, candidate, "raw_temporary", log)
            moved += 1
    return moved


def recover_integrity_failures(
    run_root: Path,
    checkpoint_path: Path,
    row: Mapping[str, Any],
    log: RunLog,
) -> int:
    """Quarantine only cache entries explicitly reported as integrity failures."""

    if not checkpoint_path.is_file():
        return 0
    try:
        frame = pd.read_parquet(checkpoint_path)
        raw = frame["failed_requests_json"].dropna().astype(str).iloc[0]
        failures = json.loads(raw)
    except Exception:
        return 0
    bad_leads = {
        int(item["lead_hours"])
        for item in failures
        if isinstance(item, dict)
        and item.get("error_type") == "GFSCacheIntegrityError"
        and item.get("lead_hours") is not None
    }
    if not bad_leads:
        return 0
    client = FastNCSSCandidateClient(run_root / "gfs_raw_cache")
    moved = 0
    for request in requests_for_plan_row(row):
        if request.lead_hours not in bad_leads:
            continue
        directory = client._cache_dir(request)  # exact cache key owned by the builder
        if directory.exists():
            quarantine_path(run_root, directory, "raw_integrity", log)
            moved += 1
    return moved


def scan_checkpoints(
    run_root: Path, plan: pd.DataFrame
) -> tuple[dict[str, CheckpointInspection], list[Path]]:
    inspections: dict[str, CheckpointInspection] = {}
    expected_paths: set[Path] = set()
    for row in plan.to_dict(orient="records"):
        path = _checkpoint_path(run_root, row)
        expected_paths.add(path.resolve())
        inspections[str(row["checkpoint_id"])] = inspect_checkpoint(
            path, row, run_root=run_root
        )
    actual_paths = {
        path.resolve()
        for path in (run_root / "checkpoints").glob(
            "year=*/scenario_hours=*/*.parquet"
        )
    }
    extras = sorted(actual_paths - expected_paths)
    return inspections, extras


def _terminal_source_evidence(failure: Mapping[str, Any]) -> dict[str, Any]:
    archive_path = str(failure["archive_path"])
    error_type = str(failure["error_type"])
    error = str(failure["error"])
    expected_url_prefix = (
        "HTTP 404 for https://tds.gdex.ucar.edu/thredds/ncss/grid/"
        f"{archive_path}?"
    )
    if (
        error_type == "GFSArchiveError"
        and error.startswith(expected_url_prefix)
        and error.endswith(_HTTP_404_FILE_NOT_FOUND_SUFFIX)
    ):
        return {
            "evidence_code": "gdex_ncss_http_404_file_not_found",
            "http_status": 404,
            "provider_detail": "FileNotFound: No such file or directory",
        }
    if error_type == "GFSMetadataError" and error == _NO_PRECIPITATION_CANDIDATE_ERROR:
        return {
            "evidence_code": "both_frozen_precipitation_candidates_absent",
            "provider_detail": (
                "FastNCSSCandidateClient reached its terminal error only after "
                "both frozen precipitation candidates were rejected as unknown"
            ),
        }
    raise ValueError(
        "ошибка не является разрешённым terminal source-unavailable evidence: "
        f"{error_type}: {error}"
    )


def _attempt_evidence(log_path: Path, checkpoint_id: str) -> dict[str, Any]:
    pattern = re.compile(
        rf"^(?P<timestamp>\S+)\s+{re.escape(checkpoint_id)}: "
        r"checkpoint пока не полон: (?P<observation>.+)$"
    )
    observations: list[dict[str, Any]] = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                observations.append(
                    {
                        "timestamp_utc": match.group("timestamp"),
                        "observation": match.group("observation"),
                    }
                )
    if len(observations) < MIN_SOURCE_UNAVAILABILITY_ATTEMPTS:
        raise ValueError(
            f"{checkpoint_id}: для source-unavailable нужно минимум "
            f"{MIN_SOURCE_UNAVAILABILITY_ATTEMPTS} записанных попытки, найдено "
            f"{len(observations)}"
        )
    return {
        "recorded_attempt_count": len(observations),
        "observations": observations,
    }


def validate_source_unavailability_manifest(
    path: Path,
    *,
    run_root: Path,
    plan: pd.DataFrame,
    inspections: Mapping[str, CheckpointInspection],
    compatibility_manifest_path: Path,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != SOURCE_UNAVAILABILITY_SCHEMA_VERSION:
        raise ValueError("неизвестная версия source-unavailability manifest")
    checks = (
        (payload.get("request_plan_sha256"), sha256_file(run_root / PLAN_FILE_NAME)),
        (
            payload.get("legacy_plan_manifest_sha256"),
            sha256_file(run_root / PLAN_MANIFEST_FILE_NAME),
        ),
        (
            payload.get("plan_compatibility_manifest_sha256"),
            sha256_file(compatibility_manifest_path),
        ),
        (payload.get("contract_sha256"), sha256_file(DEFAULT_CONTRACT)),
    )
    if any(declared != actual for declared, actual in checks):
        raise ValueError("source-unavailability manifest привязан к другим входам")
    if payload.get("decision_rule_id") != SOURCE_UNAVAILABILITY_RULE_ID:
        raise ValueError("source-unavailability manifest использует другое правило")
    if (
        payload.get("dataset_id") != "d084001"
        or payload.get("source_access_mode") != FAST_NCSS_ACCESS_MODE
    ):
        raise ValueError("source-unavailability manifest относится к другому источнику")
    frozen_method_flags = {
        "no_data_substitution": True,
        "terminal_scope": "frozen_gdex_thredds_ncss_access_method_only",
        "archive_wide_absence_claimed": False,
        "alternate_transport_used": False,
        "alternate_cycle_used": False,
        "reanalysis_used": False,
        "imputation_used": False,
    }
    if not _manifest_privacy_is_local(payload) or any(
        payload.get(key) != value for key, value in frozen_method_flags.items()
    ):
        raise ValueError("source-unavailability manifest не фиксирует privacy/no-substitution")

    incomplete_ids = sorted(
        checkpoint_id
        for checkpoint_id, item in inspections.items()
        if item.state == "incomplete"
    )
    if any(item.state in {"missing", "invalid"} for item in inspections.values()):
        raise ValueError("source-unavailability нельзя принять при missing/invalid checkpoint")
    declared_ids = sorted(str(value) for value in payload.get("checkpoint_ids", []))
    if declared_ids != incomplete_ids:
        raise ValueError("manifest должен перечислять каждый и только incomplete checkpoint")

    plan_by_id = {
        str(row["checkpoint_id"]): row for row in plan.to_dict(orient="records")
    }
    expected_snapshots: list[dict[str, Any]] = []
    checkpoint_records = payload.get("checkpoints")
    if not isinstance(checkpoint_records, list):
        raise ValueError("source-unavailability checkpoints должен быть массивом")
    declared_by_id = {
        str(item.get("checkpoint_id")): item
        for item in checkpoint_records
        if isinstance(item, dict)
    }
    if set(declared_by_id) != set(incomplete_ids) or len(declared_by_id) != len(
        checkpoint_records
    ):
        raise ValueError("source-unavailability checkpoint records неполны или повторяются")

    for checkpoint_id in incomplete_ids:
        row = plan_by_id[checkpoint_id]
        checkpoint_path = _checkpoint_path(run_root, row)
        frame = pd.read_parquet(checkpoint_path)
        failures = validate_incomplete_checkpoint(frame, row, run_root=run_root)
        record = declared_by_id[checkpoint_id]
        expected_record = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "issue_date": pd.Timestamp(row["issue_date"]).date().isoformat(),
            "availability_scenario_hours": int(row["availability_scenario_hours"]),
            "requested_snapshot_count": int(row["requested_snapshot_count"]),
            "retrieved_snapshot_count": int(frame["retrieved_snapshot_count"].iloc[0]),
            "failed_snapshot_count": int(frame["failed_snapshot_count"].iloc[0]),
        }
        if any(record.get(key) != value for key, value in expected_record.items()):
            raise ValueError(f"{checkpoint_id}: manifest не совпадает с checkpoint")
        attempts = record.get("attempt_evidence")
        if (
            not isinstance(attempts, dict)
            or int(attempts.get("recorded_attempt_count", 0))
            < MIN_SOURCE_UNAVAILABILITY_ATTEMPTS
            or len(attempts.get("observations", []))
            != int(attempts.get("recorded_attempt_count", -1))
        ):
            raise ValueError(f"{checkpoint_id}: недостаточно evidence повторных попыток")
        for failure in failures:
            expected_snapshots.append(
                {
                    "checkpoint_id": checkpoint_id,
                    **failure,
                    **_terminal_source_evidence(failure),
                }
            )

    expected_snapshots.sort(
        key=lambda item: (str(item["checkpoint_id"]), int(item["lead_hours"]))
    )
    if payload.get("unavailable_snapshots") != expected_snapshots:
        raise ValueError("source-unavailability snapshot evidence изменилось")
    if int(payload.get("checkpoint_count", -1)) != len(incomplete_ids):
        raise ValueError("source-unavailability checkpoint_count неверен")
    if int(payload.get("unavailable_snapshot_count", -1)) != len(expected_snapshots):
        raise ValueError("source-unavailability snapshot count неверен")
    return payload


def ensure_source_unavailability_manifest(
    run_root: Path,
    *,
    plan: pd.DataFrame,
    inspections: Mapping[str, CheckpointInspection],
    compatibility_manifest_path: Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Freeze explicit terminal gaps after repeated incomplete observations."""

    path = run_root / SOURCE_UNAVAILABILITY_MANIFEST_FILE_NAME
    incomplete_ids = sorted(
        checkpoint_id
        for checkpoint_id, item in inspections.items()
        if item.state == "incomplete"
    )
    if not incomplete_ids:
        if path.exists():
            raise ValueError("source-unavailability manifest exists but no gaps remain")
        return None
    if path.exists():
        return path, validate_source_unavailability_manifest(
            path,
            run_root=run_root,
            plan=plan,
            inspections=inspections,
            compatibility_manifest_path=compatibility_manifest_path,
        )

    if any(item.state in {"missing", "invalid"} for item in inspections.values()):
        raise ValueError("нельзя фиксировать source gaps при missing/invalid checkpoint")
    plan_by_id = {
        str(row["checkpoint_id"]): row for row in plan.to_dict(orient="records")
    }
    checkpoint_records: list[dict[str, Any]] = []
    unavailable_snapshots: list[dict[str, Any]] = []
    for checkpoint_id in incomplete_ids:
        row = plan_by_id[checkpoint_id]
        checkpoint_path = _checkpoint_path(run_root, row)
        frame = pd.read_parquet(checkpoint_path)
        failures = validate_incomplete_checkpoint(frame, row, run_root=run_root)
        attempt_evidence = _attempt_evidence(run_root / LOG_FILE_NAME, checkpoint_id)
        checkpoint_records.append(
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "issue_date": pd.Timestamp(row["issue_date"]).date().isoformat(),
                "availability_scenario_hours": int(
                    row["availability_scenario_hours"]
                ),
                "requested_snapshot_count": int(row["requested_snapshot_count"]),
                "retrieved_snapshot_count": int(
                    frame["retrieved_snapshot_count"].iloc[0]
                ),
                "failed_snapshot_count": int(frame["failed_snapshot_count"].iloc[0]),
                "attempt_evidence": attempt_evidence,
            }
        )
        for failure in failures:
            unavailable_snapshots.append(
                {
                    "checkpoint_id": checkpoint_id,
                    **failure,
                    **_terminal_source_evidence(failure),
                }
            )
    unavailable_snapshots.sort(
        key=lambda item: (str(item["checkpoint_id"]), int(item["lead_hours"]))
    )
    payload = {
        "schema_version": SOURCE_UNAVAILABILITY_SCHEMA_VERSION,
        "created_at_utc": utc_text(),
        "decision_rule_id": SOURCE_UNAVAILABILITY_RULE_ID,
        "dataset_id": "d084001",
        "source_access_mode": FAST_NCSS_ACCESS_MODE,
        "request_plan_path": str((run_root / PLAN_FILE_NAME).resolve()),
        "request_plan_sha256": sha256_file(run_root / PLAN_FILE_NAME),
        "legacy_plan_manifest_path": str(
            (run_root / PLAN_MANIFEST_FILE_NAME).resolve()
        ),
        "legacy_plan_manifest_sha256": sha256_file(
            run_root / PLAN_MANIFEST_FILE_NAME
        ),
        "plan_compatibility_manifest_path": str(
            compatibility_manifest_path.resolve()
        ),
        "plan_compatibility_manifest_sha256": sha256_file(
            compatibility_manifest_path
        ),
        "contract_sha256": sha256_file(DEFAULT_CONTRACT),
        "minimum_recorded_attempts_per_checkpoint": (
            MIN_SOURCE_UNAVAILABILITY_ATTEMPTS
        ),
        "download_log_sha256_at_creation": sha256_file(run_root / LOG_FILE_NAME),
        "checkpoint_count": len(checkpoint_records),
        "unavailable_snapshot_count": len(unavailable_snapshots),
        "checkpoint_ids": incomplete_ids,
        "checkpoints": checkpoint_records,
        "unavailable_snapshots": unavailable_snapshots,
        "no_data_substitution": True,
        "terminal_scope": "frozen_gdex_thredds_ncss_access_method_only",
        "archive_wide_absence_claimed": False,
        "alternate_transport_used": False,
        "alternate_cycle_used": False,
        "reanalysis_used": False,
        "imputation_used": False,
        "contains_field_ids": False,
        "contains_outcomes": False,
        "contains_coarse_grid_coordinates": True,
        "distribution": "local_private_do_not_publish",
    }
    atomic_json(path, payload)
    return path, validate_source_unavailability_manifest(
        path,
        run_root=run_root,
        plan=plan,
        inspections=inspections,
        compatibility_manifest_path=compatibility_manifest_path,
    )


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def interruptible_sleep(seconds: float, stop: StopController) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while not stop.requested:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def bind_assembly_manifests(
    run_root: Path,
    *,
    compatibility_manifest_path: Path,
    source_unavailability: tuple[Path, Mapping[str, Any]] | None,
) -> None:
    """Bind derived feature provenance to the two versioned audit manifests."""

    source_path = run_root / SOURCE_MANIFEST_FILE_NAME
    manifest = _read_json(source_path)
    manifest["request_plan_compatibility_manifest"] = {
        "path": str(compatibility_manifest_path.resolve()),
        "sha256": sha256_file(compatibility_manifest_path),
        "compatibility_schema_version": PLAN_COMPATIBILITY_SCHEMA_VERSION,
    }
    if source_unavailability is None:
        manifest["source_unavailability_manifest"] = None
    else:
        path, payload = source_unavailability
        manifest["source_unavailability_manifest"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "schema_version": SOURCE_UNAVAILABILITY_SCHEMA_VERSION,
            "checkpoint_count": int(payload["checkpoint_count"]),
            "unavailable_snapshot_count": int(payload["unavailable_snapshot_count"]),
        }
    atomic_json(source_path, manifest)


def validate_existing_assembly(
    run_root: Path,
    plan: pd.DataFrame,
    *,
    compatibility_manifest_path: Path,
    source_unavailability: tuple[Path, Mapping[str, Any]] | None,
) -> bool:
    feature_path = run_root / FEATURE_FILE_NAME
    manifest_path = run_root / SOURCE_MANIFEST_FILE_NAME
    if not feature_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        if manifest.get("feature_table_sha256") != sha256_file(feature_path):
            return False
        if manifest.get("request_plan_sha256") != sha256_file(
            run_root / PLAN_FILE_NAME
        ):
            return False
        if int(manifest.get("planned_checkpoints", -1)) != len(plan):
            return False
        if int(manifest.get("assembled_checkpoints", -1)) != len(plan):
            return False
        if int(manifest.get("missing_checkpoint_count", -1)) != 0:
            return False
        compatibility_link = manifest.get("request_plan_compatibility_manifest")
        if not isinstance(compatibility_link, dict) or compatibility_link != {
            "path": str(compatibility_manifest_path.resolve()),
            "sha256": sha256_file(compatibility_manifest_path),
            "compatibility_schema_version": PLAN_COMPATIBILITY_SCHEMA_VERSION,
        }:
            return False
        expected_unavailable_ids: set[str] = set()
        if source_unavailability is None:
            if manifest.get("source_unavailability_manifest") is not None:
                return False
        else:
            unavailable_path, unavailable_payload = source_unavailability
            unavailable_link = manifest.get("source_unavailability_manifest")
            if not isinstance(unavailable_link, dict) or unavailable_link != {
                "path": str(unavailable_path.resolve()),
                "sha256": sha256_file(unavailable_path),
                "schema_version": SOURCE_UNAVAILABILITY_SCHEMA_VERSION,
                "checkpoint_count": int(unavailable_payload["checkpoint_count"]),
                "unavailable_snapshot_count": int(
                    unavailable_payload["unavailable_snapshot_count"]
                ),
            }:
                return False
            expected_unavailable_ids = set(unavailable_payload["checkpoint_ids"])
        features = load_forecast_feature_table(feature_path)
        unavailable_ids = set(
            features.loc[
                ~features["forecast_available"].fillna(False).astype(bool),
                "checkpoint_id",
            ].astype(str)
        )
        if unavailable_ids != expected_unavailable_ids:
            return False
        for checkpoint_id, group in features.groupby("checkpoint_id", sort=False):
            available = group["forecast_available"].fillna(False).astype(bool)
            if (str(checkpoint_id) in expected_unavailable_ids) != (not available.any()):
                return False
        return True
    except Exception:
        return False


def make_summary(
    *,
    state: str,
    run_root: Path,
    plan: pd.DataFrame,
    plan_manifest: Mapping[str, Any],
    started_at: datetime,
    inspections: Mapping[str, CheckpointInspection],
    failed_this_run: Sequence[str],
    message: str,
) -> dict[str, Any]:
    by_state = {
        name: sorted(key for key, item in inspections.items() if item.state == name)
        for name in ("complete", "missing", "incomplete", "invalid")
    }
    feature_path = run_root / FEATURE_FILE_NAME
    compatibility_path = run_root / PLAN_COMPATIBILITY_MANIFEST_FILE_NAME
    unavailability_path = run_root / SOURCE_UNAVAILABILITY_MANIFEST_FILE_NAME
    source_manifest_path = run_root / SOURCE_MANIFEST_FILE_NAME
    documented_ids: list[str] = []
    if unavailability_path.exists():
        try:
            documented_ids = sorted(
                str(value)
                for value in _read_json(unavailability_path).get("checkpoint_ids", [])
            )
        except Exception:
            documented_ids = []
    unresolved_ids = sorted(
        by_state["missing"]
        + by_state["invalid"]
        + [
            checkpoint_id
            for checkpoint_id in by_state["incomplete"]
            if checkpoint_id not in set(documented_ids)
        ]
    )
    return {
        "schema_version": "gfs_overnight_download_summary_v1",
        "state": state,
        "message": message,
        "started_at_utc": utc_text(started_at),
        "finished_at_utc": utc_text(),
        "elapsed_seconds": (utc_now() - started_at).total_seconds(),
        "run_root": str(run_root.resolve()),
        "dataset_id": "d084001",
        "plan_path": str((run_root / PLAN_FILE_NAME).resolve()),
        "plan_sha256": sha256_file(run_root / PLAN_FILE_NAME),
        "plan_manifest_sha256": sha256_file(run_root / PLAN_MANIFEST_FILE_NAME),
        "plan_compatibility_manifest_path": (
            str(compatibility_path.resolve()) if compatibility_path.exists() else None
        ),
        "plan_compatibility_manifest_sha256": (
            sha256_file(compatibility_path) if compatibility_path.exists() else None
        ),
        "source_unavailability_manifest_path": (
            str(unavailability_path.resolve()) if unavailability_path.exists() else None
        ),
        "source_unavailability_manifest_sha256": (
            sha256_file(unavailability_path) if unavailability_path.exists() else None
        ),
        "source_manifest_path": (
            str(source_manifest_path.resolve()) if source_manifest_path.exists() else None
        ),
        "source_manifest_sha256": (
            sha256_file(source_manifest_path) if source_manifest_path.exists() else None
        ),
        "contract_sha256": plan_manifest.get("contract_sha256"),
        "planned_checkpoints": int(len(plan)),
        "planned_snapshot_files": int(
            pd.to_numeric(plan["requested_snapshot_count"], errors="raise").sum()
        ),
        "checkpoint_counts": {name: len(values) for name, values in by_state.items()},
        "documented_source_unavailable_checkpoint_ids": documented_ids,
        "resolved_checkpoint_count": len(by_state["complete"]) + len(documented_ids),
        "unfinished_checkpoint_ids": unresolved_ids,
        "failed_checkpoint_ids_this_run": sorted(set(failed_this_run)),
        "feature_table_path": str(feature_path.resolve()) if feature_path.exists() else None,
        "feature_table_sha256": sha256_file(feature_path) if feature_path.exists() else None,
        "resume_command": (
            f"{REPO_ROOT / '.venv/bin/python'} "
            f"{Path(__file__).resolve()}"
        ),
        "model_training_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ночная возобновляемая загрузка замороженного архива GDEX GFS; "
            "модели не запускаются"
        )
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0)
    parser.add_argument("--outage-after", type=int, default=4)
    parser.add_argument("--outage-pause-seconds", type=float, default=180.0)
    parser.add_argument("--minimum-free-gb", type=float, default=2.0)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="проверить план и текущий прогресс, ничего не скачивая",
    )
    parser.add_argument(
        "--allow-sleep",
        action="store_true",
        help="не включать caffeinate на macOS",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.attempts < 1:
        raise ValueError("--attempts должен быть положительным")
    if args.retry_backoff_seconds < 0 or args.outage_pause_seconds < 0:
        raise ValueError("паузы не могут быть отрицательными")
    if args.outage_after < 1:
        raise ValueError("--outage-after должен быть положительным")
    if args.minimum_free_gb < 0:
        raise ValueError("--minimum-free-gb не может быть отрицательным")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    run_root = args.run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    log = RunLog(run_root / LOG_FILE_NAME)
    started_at = utc_now()
    plan: pd.DataFrame | None = None
    plan_manifest: dict[str, Any] | None = None
    compatibility_manifest_path: Path | None = None
    source_unavailability: tuple[Path, dict[str, Any]] | None = None
    failed_this_run: list[str] = []
    stop = StopController()
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, stop.handle)
    caffeinate: subprocess.Popen[bytes] | None = None
    try:
        with exclusive_run_lock(run_root / LOCK_FILE_NAME):
            log.write(f"старт; каталог {run_root}")
            caffeinate = start_caffeinate(not args.allow_sleep, log)
            plan, plan_manifest = ensure_and_validate_plan(run_root)
            compatibility_manifest_path, _ = ensure_plan_compatibility_manifest(
                run_root, plan, plan_manifest
            )
            recover_interrupted_raw_writes(run_root, log)
            inspections, extras = scan_checkpoints(run_root, plan)
            if extras:
                raise ValueError(
                    "найдены checkpoint вне замороженного плана: "
                    + ", ".join(str(path) for path in extras[:5])
                )
            unavailability_path = run_root / SOURCE_UNAVAILABILITY_MANIFEST_FILE_NAME
            if unavailability_path.exists():
                source_unavailability = (
                    unavailability_path,
                    validate_source_unavailability_manifest(
                        unavailability_path,
                        run_root=run_root,
                        plan=plan,
                        inspections=inspections,
                        compatibility_manifest_path=compatibility_manifest_path,
                    ),
                )
            elif not args.check_only:
                try:
                    source_unavailability = ensure_source_unavailability_manifest(
                        run_root,
                        plan=plan,
                        inspections=inspections,
                        compatibility_manifest_path=compatibility_manifest_path,
                    )
                    if source_unavailability is not None:
                        log.write(
                            "неоднократно неполные checkpoint с текущей точной "
                            "terminal-причиной замороженного NCSS-метода "
                            "зафиксированы без подмены данных"
                        )
                except ValueError as error:
                    log.write(
                        "текущие incomplete checkpoint пока не допускают финализацию: "
                        f"{error}"
                    )
            documented_ids = (
                set(source_unavailability[1]["checkpoint_ids"])
                if source_unavailability is not None
                else set()
            )
            complete_at_start = sum(
                item.state == "complete" for item in inspections.values()
            )
            log.write(
                f"план проверен: {len(plan)} checkpoint, "
                f"{int(plan['requested_snapshot_count'].sum())} файлов; "
                f"уже завершено {complete_at_start}"
            )
            if args.check_only:
                summary = make_summary(
                    state="check_only",
                    run_root=run_root,
                    plan=plan,
                    plan_manifest=plan_manifest,
                    started_at=started_at,
                    inspections=inspections,
                    failed_this_run=(),
                    message="Проверка выполнена, сеть не использовалась.",
                )
                atomic_json(run_root / SUMMARY_FILE_NAME, summary)
                log.write("check-only завершён; сеть не использовалась")
                return 0

            if free_gib(run_root) < args.minimum_free_gb:
                raise RuntimeError(
                    f"свободно менее {args.minimum_free_gb:g} ГиБ; загрузка не начата"
                )

            successful_seconds: list[float] = []
            consecutive_failures = 0
            complete_count = complete_at_start
            rows = plan.to_dict(orient="records")
            total_snapshot_files = int(
                pd.to_numeric(plan["requested_snapshot_count"], errors="raise").sum()
            )
            for row in rows:
                checkpoint_id = str(row["checkpoint_id"])
                checkpoint_path = _checkpoint_path(run_root, row)
                inspection = inspect_checkpoint(
                    checkpoint_path, row, run_root=run_root
                )
                if inspection.state == "complete" or checkpoint_id in documented_ids:
                    continue
                if stop.requested:
                    break
                if inspection.state == "invalid" and checkpoint_path.exists():
                    quarantine_path(
                        run_root, checkpoint_path, "checkpoint_invalid", log
                    )
                success = False
                checkpoint_started = time.monotonic()
                for attempt in range(1, args.attempts + 1):
                    if stop.requested:
                        break
                    if free_gib(run_root) < args.minimum_free_gb:
                        log.write(
                            f"остановка: свободно менее {args.minimum_free_gb:g} ГиБ"
                        )
                        stop.requested = True
                        break
                    if attempt > 1:
                        delay = min(
                            args.retry_backoff_seconds * (2 ** (attempt - 2)), 120.0
                        )
                        log.write(
                            f"{checkpoint_id}: повтор {attempt}/{args.attempts} "
                            f"через {format_duration(delay)}"
                        )
                        interruptible_sleep(delay, stop)
                        if stop.requested:
                            break
                    log.write(
                        f"{checkpoint_id}: загружаю, попытка "
                        f"{attempt}/{args.attempts}"
                    )
                    try:
                        eta_seconds = (
                            statistics.median(successful_seconds)
                            * (len(plan) - complete_count)
                            if successful_seconds
                            else None
                        )
                        with CheckpointProgress(
                            run_root=run_root,
                            row=row,
                            completed_checkpoints=complete_count,
                            total_checkpoints=len(plan),
                            total_snapshot_files=total_snapshot_files,
                            eta_seconds=eta_seconds,
                        ):
                            fetch_request_plan(
                                run_root=run_root,
                                dates=[
                                    pd.Timestamp(row["issue_date"])
                                    .date()
                                    .isoformat()
                                ],
                                scenario_hours=[
                                    int(row["availability_scenario_hours"])
                                ],
                                max_workers=args.workers,
                                retry_incomplete=True,
                                fast_ncss_candidates=True,
                            )
                    except Exception as error:
                        log.write(
                            f"{checkpoint_id}: попытка {attempt} завершилась "
                            f"{type(error).__name__}: {error}"
                        )
                    inspection = inspect_checkpoint(
                        checkpoint_path, row, run_root=run_root
                    )
                    if inspection.state == "complete":
                        success = True
                        break
                    recover_integrity_failures(
                        run_root, checkpoint_path, row, log
                    )
                    log.write(
                        f"{checkpoint_id}: checkpoint пока не полон: "
                        f"{inspection.reason}"
                    )

                elapsed = time.monotonic() - checkpoint_started
                if success:
                    consecutive_failures = 0
                    complete_count += 1
                    successful_seconds.append(elapsed)
                    successful_seconds = successful_seconds[-50:]
                    remaining = len(plan) - complete_count
                    typical = statistics.median(successful_seconds)
                    eta = typical * remaining
                    log.write(
                        f"[{complete_count}/{len(plan)}] {checkpoint_id} OK за "
                        f"{format_duration(elapsed)}; осталось {remaining}; "
                        f"ETA {format_duration(eta)}"
                    )
                else:
                    failed_this_run.append(checkpoint_id)
                    consecutive_failures += 1
                    log.write(
                        f"{checkpoint_id}: не завершён после {args.attempts} попыток; "
                        "будет повторён при следующем запуске"
                    )
                    if (
                        not stop.requested
                        and consecutive_failures >= args.outage_after
                        and args.outage_pause_seconds > 0
                    ):
                        log.write(
                            f"похоже на общий сбой источника; пауза "
                            f"{format_duration(args.outage_pause_seconds)}"
                        )
                        interruptible_sleep(args.outage_pause_seconds, stop)
                        consecutive_failures = 0

            inspections, extras = scan_checkpoints(run_root, plan)
            complete = sum(item.state == "complete" for item in inspections.values())
            if stop.requested:
                message = (
                    f"Остановлено безопасно: готово {complete}/{len(plan)}. "
                    "Повторите ту же команду для продолжения."
                )
                state = "stopped"
                exit_code = 130
            elif extras:
                message = "Найдены checkpoint вне плана; сборка не выполнена."
                state = "invalid"
                exit_code = 1
            else:
                missing_or_invalid = {
                    checkpoint_id: item.reason
                    for checkpoint_id, item in inspections.items()
                    if item.state in {"missing", "invalid"}
                }
                unavailability_error: str | None = None
                if not missing_or_invalid:
                    try:
                        source_unavailability = ensure_source_unavailability_manifest(
                            run_root,
                            plan=plan,
                            inspections=inspections,
                            compatibility_manifest_path=compatibility_manifest_path,
                        )
                    except ValueError as error:
                        unavailability_error = str(error)
                documented_ids = (
                    set(source_unavailability[1]["checkpoint_ids"])
                    if source_unavailability is not None
                    else set()
                )
                unresolved_ids = sorted(
                    checkpoint_id
                    for checkpoint_id, item in inspections.items()
                    if item.state in {"missing", "invalid"}
                    or (item.state == "incomplete" and checkpoint_id not in documented_ids)
                )
                if unresolved_ids:
                    detail = (
                        f" Причина: {unavailability_error}" if unavailability_error else ""
                    )
                    message = (
                        f"После повторов разрешено {len(plan) - len(unresolved_ids)}/"
                        f"{len(plan)} checkpoint; незавершённые checkpoint останутся "
                        f"для следующего запуска.{detail}"
                    )
                    state = "incomplete"
                    exit_code = 2
                elif validate_existing_assembly(
                    run_root,
                    plan,
                    compatibility_manifest_path=compatibility_manifest_path,
                    source_unavailability=source_unavailability,
                ):
                    log.write("готовая итоговая таблица уже существует и проверена")
                else:
                    log.write(
                        "все checkpoint разрешены; собираю итоговую таблицу без "
                        "подмены документированных source-unavailable прогнозов"
                    )
                    manifest = assemble_feature_table(run_root=run_root)
                    if int(manifest.get("missing_checkpoint_count", -1)) != 0:
                        raise RuntimeError("сборка сообщила о пропущенных checkpoint")
                    bind_assembly_manifests(
                        run_root,
                        compatibility_manifest_path=compatibility_manifest_path,
                        source_unavailability=source_unavailability,
                    )
                    if not validate_existing_assembly(
                        run_root,
                        plan,
                        compatibility_manifest_path=compatibility_manifest_path,
                        source_unavailability=source_unavailability,
                    ):
                        raise RuntimeError("итоговая таблица не прошла проверку")
                if not unresolved_ids:
                    documented_count = len(documented_ids)
                    message = (
                        f"Загрузка и проверка завершены: {complete} полностью доступных "
                        f"и {documented_count} документированных source-unavailable "
                        f"checkpoint из {len(plan)}. Модели не запускались."
                    )
                    state = (
                        "complete_with_documented_source_unavailability"
                        if documented_count
                        else "complete"
                    )
                    exit_code = 0
            summary = make_summary(
                state=state,
                run_root=run_root,
                plan=plan,
                plan_manifest=plan_manifest,
                started_at=started_at,
                inspections=inspections,
                failed_this_run=failed_this_run,
                message=message,
            )
            atomic_json(run_root / SUMMARY_FILE_NAME, summary)
            log.write(message)
            return exit_code
    except KeyboardInterrupt:
        if plan is not None and plan_manifest is not None:
            inspections, _ = scan_checkpoints(run_root, plan)
            summary = make_summary(
                state="stopped",
                run_root=run_root,
                plan=plan,
                plan_manifest=plan_manifest,
                started_at=started_at,
                inspections=inspections,
                failed_this_run=failed_this_run,
                message="Принудительно остановлено; повторный запуск продолжит по checkpoint.",
            )
            atomic_json(run_root / SUMMARY_FILE_NAME, summary)
        log.write("принудительная остановка; сохранённые checkpoint не затронуты")
        return 130
    except Exception as error:
        log.write(f"ОШИБКА: {type(error).__name__}: {error}")
        return 1
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
