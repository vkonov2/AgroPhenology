"""One-shot offline builder for the potato late-blight shadow readiness run.

This orchestrator has no network, scheduler, or delivery adapter.  It verifies
the frozen parents, creates a frozen registry, audits local source readiness,
runs the synthetic technical demonstration, replays the saved decisions, and
atomically creates a new versioned reporting bundle.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from .shadow_audit import build_source_compatibility_manifest
from .shadow_engine import readiness, replay_from_log, technical_demo, verify_log
from .shadow_registry import (
    PARENT_RUNS,
    verify_parent_run,
    verify_shadow_registry,
    write_shadow_registry,
)
from .shadow_reporting import generate_shadow_readiness_run


RUN_NAME_PATTERN = re.compile(r"^(?P<day>\d{8})_shadow_readiness_v(?P<version>[1-9]\d*)$")
DEFAULT_RESULTS_RELATIVE = Path("results/late_blight_early_warning")
DEFAULT_CONTRACT_RELATIVE = Path(
    "docs/research/late_blight_early_warning/shadow_readiness_contract.json"
)


def _utc_datetime(value: str | datetime | None) -> datetime:
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
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def next_readiness_run_id(results_root: str | Path, issue_day: date) -> str:
    """Choose the next unused version without treating an old run as writable."""
    root = Path(results_root)
    prefix = issue_day.strftime("%Y%m%d")
    versions: list[int] = []
    if root.is_dir():
        for path in root.iterdir():
            match = RUN_NAME_PATTERN.fullmatch(path.name)
            if match and match.group("day") == prefix:
                versions.append(int(match.group("version")))
    return f"{prefix}_shadow_readiness_v{max(versions, default=0) + 1}"


def _validate_explicit_run_id(run_id: str) -> str:
    if RUN_NAME_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must have form YYYYMMDD_shadow_readiness_vN and contain no path"
        )
    return run_id


def _parent_snapshot(project_root: Path) -> dict[str, Any]:
    checks = {cycle: verify_parent_run(project_root, cycle) for cycle in PARENT_RUNS}
    return {
        "status": "passed",
        "parents": checks,
        "all_outputs_match": all(row["all_outputs_match"] for row in checks.values()),
    }


def _parent_snapshots_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    fields = ("run_id", "manifest_sha256", "outputs_checked", "all_outputs_match")
    return all(
        all(before["parents"][cycle][field] == after["parents"][cycle][field] for field in fields)
        for cycle in PARENT_RUNS
    )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    parents = contract.get("parents")
    if not isinstance(parents, Mapping):
        raise ValueError("shadow readiness contract has no parent lock")
    for cycle, expected in PARENT_RUNS.items():
        actual = parents.get(cycle)
        if not isinstance(actual, Mapping):
            raise ValueError(f"shadow readiness contract misses parent {cycle}")
        if actual.get("run_id") != expected["run_id"]:
            raise ValueError(f"shadow readiness contract has wrong {cycle} run_id")
        if actual.get("execution_manifest_sha256") != expected["manifest_sha256"]:
            raise ValueError(f"shadow readiness contract has wrong {cycle} manifest hash")
    prohibited = set(contract.get("prohibited", []))
    required_prohibitions = {
        "send_real_notifications",
        "install_or_activate_a_schedule",
        "overwrite_parent_results_commit_or_push",
    }
    if not required_prohibitions.issubset(prohibited):
        raise ValueError("shadow readiness contract misses required side-effect prohibitions")


def _source_snapshot(project_root: Path) -> dict[str, Any]:
    relatives = [
        "src/agro_phenology/shadow_storage.py",
        "src/agro_phenology/shadow_sources.py",
        "src/agro_phenology/shadow_registry.py",
        "src/agro_phenology/shadow_engine.py",
        "src/agro_phenology/shadow_audit.py",
        "src/agro_phenology/shadow_cli.py",
        "src/agro_phenology/shadow_reporting.py",
        "src/agro_phenology/shadow_readiness_pipeline.py",
        "docs/research/late_blight_early_warning/shadow_readiness_contract.json",
        "docs/research/late_blight_early_warning/prospective_logging_spec.md",
        "pyproject.toml",
        "requirements-early-warning.txt",
        "tests/test_shadow_storage.py",
        "tests/test_shadow_sources.py",
        "tests/test_shadow_registry.py",
        "tests/test_shadow_audit.py",
        "tests/test_shadow_engine.py",
        "tests/test_shadow_reporting.py",
        "tests/test_shadow_readiness_pipeline.py",
    ]
    files: list[dict[str, Any]] = []
    for relative in relatives:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "relative_path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "status": "complete",
        "files": files,
        "source_set_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _abstentions_by_reason(database: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with sqlite3.connect(database) as connection:
        for (raw,) in connection.execute(
            "SELECT decision_payload_json FROM decisions ORDER BY decision_slot_utc"
        ):
            payload = json.loads(raw)
            if payload.get("decision_action") == "abstain":
                reason = str(
                    payload.get("input_reason")
                    or payload.get("decision_reason")
                    or "unspecified"
                )
                counts[reason] += 1
    return dict(sorted(counts.items()))


def _demo_summary(raw: Mapping[str, Any], database: Path) -> dict[str, Any]:
    decisions = int(raw.get("unique_decisions", 0))
    abstentions = int(raw.get("abstentions", 0))
    alarm_days = int(raw.get("active_alarm_days", 0))
    expected_per_field = int(raw.get("expected_daily_slots_per_field", 0))
    executed_per_field = int(raw.get("executed_daily_slots_per_field", 0))
    return {
        "status": raw.get("status", "unknown"),
        "mode": raw.get("demonstration_mode", "retrospective_replay"),
        "technical_demo_not_prospective": True,
        "synthetic_inputs_only": bool(raw.get("synthetic_inputs_only", True)),
        "disease_quality_metrics_computed": bool(
            raw.get("disease_quality_metrics_computed", False)
        ),
        "live_run_executed": bool(raw.get("prospective_live_run", False)),
        "network_status": "used" if raw.get("network_used") else "not_called",
        "network_used": bool(raw.get("network_used", False)),
        "real_notifications_sent": int(raw.get("notifications_sent", 0)),
        "schedule_activated": False,
        "field_registry_status": "synthetic_only",
        "expected_daily_slots_per_field": expected_per_field,
        "executed_daily_slots_per_field": executed_per_field,
        "scheduled_slots": int(raw.get("first_slot", {}).get("fields", 0))
        * expected_per_field,
        "missed_slots": int(raw.get("missed_daily_slots", 0)),
        "late_slots": 0,
        "expected_decisions": decisions,
        "decision_records": decisions,
        "computed_fraction": ((decisions - abstentions) / decisions) if decisions else 0.0,
        "abstentions": abstentions,
        "abstention_fraction": (abstentions / decisions) if decisions else 0.0,
        "abstentions_by_reason": _abstentions_by_reason(database),
        "weather_correction_fraction": float(raw.get("active_weather_fraction_in_c6", 0.0)),
        "compatible_weather_fraction": float(raw.get("active_weather_fraction_in_c6", 0.0)),
        "c0_fallback_fraction": float(raw.get("effective_c0_fraction_in_c6", 0.0)),
        "effective_c0_fraction": float(raw.get("effective_c0_fraction_in_c6", 0.0)),
        "virtual_messages": int(raw.get("virtual_message_candidates", 0)),
        "active_alarm_days": alarm_days,
        "active_alarm_fraction": (alarm_days / decisions) if decisions else 0.0,
        "source_switch_breaks_growth_comparability": bool(
            raw.get("source_switch_breaks_growth_comparability", False)
        ),
        "source_switch_preserves_cooldown_state": bool(
            raw.get("source_switch_preserves_cooldown_state", False)
        ),
        "replay_status": raw.get("verification", {}).get("replay", {}).get("status"),
    }


def _build_start_boundary(
    *,
    source_manifest: Mapping[str, Any],
    created_at: datetime,
) -> dict[str, Any]:
    scheduled = datetime.fromisoformat(
        str(source_manifest["scheduled_slot_utc"]).replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    issue = date.fromisoformat(str(source_manifest["issue_local_date"]))
    current_slot_allowed = created_at <= scheduled
    first_day = issue if current_slot_allowed else issue + timedelta(days=1)
    first_local = datetime.combine(
        first_day,
        time(8, 0),
        tzinfo=ZoneInfo("Europe/Riga"),
    )
    return {
        "readiness_locked_at_utc": _utc_text(created_at),
        "current_scheduled_slot_local": source_manifest["scheduled_slot_local"],
        "current_scheduled_slot_utc": source_manifest["scheduled_slot_utc"],
        "current_issue_slot_can_be_prospective": current_slot_allowed,
        "backdating_allowed": False,
        "first_honest_scheduled_slot_local": first_local.isoformat(),
        "first_honest_scheduled_slot_utc": _utc_text(
            first_local.astimezone(timezone.utc)
        ),
        "conditional_on_required_inputs": True,
    }


def _collect_blockers(
    source_manifest: Mapping[str, Any],
    runtime_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    start_boundary = source_manifest.get("shadow_start_boundary", {})
    if not bool(start_boundary.get("current_issue_slot_can_be_prospective", False)):
        blockers.append(
            {
                "id": "readiness_lock_after_current_scheduled_slot",
                "status": "blocks_backdated_current_slot",
                "detail": (
                    "Readiness lock создан после scheduled 08:00 Europe/Riga; слот текущей "
                    "issue-date нельзя объявить prospective_live или восстановить задним числом."
                ),
                "required_action": (
                    "Начать не раньше первого будущего слота "
                    f"{start_boundary.get('first_honest_scheduled_slot_local', 'после lock')} "
                    "и только при наличии разрешённых входов."
                ),
            }
        )
    active = source_manifest.get("active_field_registry", {})
    runtime_field = runtime_readiness.get("field_registry", {})
    if active.get("status") != "available" or runtime_field.get("status") != "available":
        registry_detail = (
            "Актуальный реестр найден, но не прошёл runtime-проверку."
            if active.get("status") == "available"
            else "Не найден актуальный реестр реально подключённых полей картофеля."
        )
        blockers.append(
            {
                "id": "missing_actual_active_field_registry",
                "status": "blocks_live_run_once",
                "detail": registry_detail,
                "required_action": (
                    "Передать versioned registry с псевдонимными field/season ID, реальными "
                    "датами подключения, crop=potato, укрупнённым region_code и защищённой "
                    "weather_location_ref; исторические координаты не использовать."
                ),
            }
        )
    if not bool(source_manifest.get("weather_branch_executable_today", False)):
        blockers.append(
            {
                "id": "exact_era5_t_minus_2_unavailable",
                "status": "blocks_weather_correction_only",
                "detail": (
                    "Frozen C4/C5/C6 требуют ERA5 до issue_date-2, а документированная "
                    "задержка около пяти дней не позволяет считать этот вход доступным."
                ),
                "required_action": (
                    "Не подменять ERA5 и не менять cutoff. Для иной оперативной погоды нужен "
                    "отдельный source-bridge protocol; до него C6 остаётся на C0 fallback, "
                    "C4/C5 воздерживаются."
                ),
            }
        )
    probe = source_manifest.get("current_api_probe", {})
    if not bool(probe.get("performed", False)):
        blockers.append(
            {
                "id": "live_source_freshness_not_measured",
                "status": "blocks_claim_of_live_source_availability",
                "detail": "Live endpoint не проверялся без разрешённой актуальной location reference.",
                "required_action": (
                    "Разрешить конкретный источник и защищённую привязку действующих полей; "
                    "фактическую свежесть записывать по времени получения, не задним числом."
                ),
            }
        )
    blockers.append(
        {
            "id": "independent_field_observation_workflow_not_supplied",
            "status": "blocks_prospective_quality_evaluation",
            "detail": (
                "Не предоставлен операционный план независимых от score осмотров и внесения "
                "target-specific исходов."
            ),
            "required_action": (
                "Назначить ответственных, заранее определить независимые визиты, способ "
                "подтверждения, сроки регистрации и правила версий/цензурирования."
            ),
        }
    )
    return {
        "status": "blocked_for_live_collection_inputs",
        "blockers": blockers,
        "runtime_blocker_codes": list(runtime_readiness.get("blockers", [])),
        "live_collection_started": False,
    }


def verify_readiness_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    manifest_path = run / "execution_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _json_object(manifest_path, label="execution manifest")
    failures: list[dict[str, str]] = []
    for relative, expected in manifest.get("output_hashes", {}).items():
        path = run / relative
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
        elif _sha256(path) != expected.get("sha256"):
            failures.append({"path": relative, "reason": "sha256_mismatch"})
        elif path.stat().st_size != int(expected.get("bytes", -1)):
            failures.append({"path": relative, "reason": "byte_length_mismatch"})
    return {
        "status": "passed" if not failures else "failed",
        "run_id": manifest.get("run_id"),
        "manifest_sha256": _sha256(manifest_path),
        "outputs_checked": len(manifest.get("output_hashes", {})),
        "failures": failures,
    }


def export_review_package(run_dir: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy the already-sanitised package only after verifying the frozen run."""
    verification = verify_readiness_run(run_dir)
    if verification["status"] != "passed":
        raise ValueError("readiness run integrity failed")
    source = Path(run_dir).resolve() / "review_package.zip"
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite review package: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {
        "status": "exported",
        "source_run_id": verification["run_id"],
        "destination": str(target),
        "sha256": _sha256(target),
        "bytes": target.stat().st_size,
    }


def run_shadow_readiness(
    project_root: str | Path,
    *,
    results_root: str | Path | None = None,
    run_id: str | None = None,
    issue_local_date: date | str | None = None,
    created_at_utc: str | datetime | None = None,
    active_field_registry_path: str | Path | None = None,
    test_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one complete offline readiness cycle into a new run directory."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    created_dt = _utc_datetime(created_at_utc)
    created_text = _utc_text(created_dt)
    if issue_local_date is None:
        issue_day = created_dt.astimezone(ZoneInfo("Europe/Riga")).date()
    elif isinstance(issue_local_date, str):
        issue_day = date.fromisoformat(issue_local_date)
    else:
        issue_day = issue_local_date
    output_root = (
        Path(results_root).resolve()
        if results_root is not None
        else root / DEFAULT_RESULTS_RELATIVE
    )
    output_root.mkdir(parents=True, exist_ok=True)
    parent_directories = [
        root / DEFAULT_RESULTS_RELATIVE / item["run_id"] for item in PARENT_RUNS.values()
    ]
    if any(output_root == parent or output_root.is_relative_to(parent) for parent in parent_directories):
        raise ValueError("results_root cannot be inside an immutable parent run")
    selected_run_id = (
        _validate_explicit_run_id(run_id)
        if run_id is not None
        else next_readiness_run_id(output_root, issue_day)
    )
    destination = output_root / selected_run_id
    if destination.exists():
        raise FileExistsError(f"readiness run already exists: {destination}")

    parent_before = _parent_snapshot(root)
    contract_path = root / DEFAULT_CONTRACT_RELATIVE
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = _json_object(contract_path, label="shadow readiness contract")
    _validate_contract(contract)
    source_snapshot = _source_snapshot(root)

    with TemporaryDirectory(prefix=".shadow-readiness-stage-", dir=output_root) as raw_stage:
        stage = Path(raw_stage)
        registry_path = write_shadow_registry(
            stage / "shadow_registry.json",
            root,
            locked_at_utc=created_text,
        )
        registry = _json_object(registry_path, label="shadow registry")
        registry_verification = verify_shadow_registry(registry_path, root)
        registry_verification["registry_path"] = "shadow_registry.json"

        source_manifest = build_source_compatibility_manifest(
            root,
            issue_local_date=issue_day,
            audited_at_utc=created_text,
            active_field_registry_path=active_field_registry_path,
        )
        source_manifest["shadow_start_boundary"] = _build_start_boundary(
            source_manifest=source_manifest,
            created_at=created_dt,
        )
        demo_database = stage / "technical_demo.sqlite"
        raw_demo = technical_demo(
            project_root=root,
            registry_path=registry_path,
            database_path=demo_database,
            created_at_utc=created_text,
        )
        replay = replay_from_log(
            database_path=demo_database,
            registry_path=registry_path,
            project_root=root,
        )
        log_verification = verify_log(
            database_path=demo_database,
            registry_path=registry_path,
            project_root=root,
        )
        runtime_readiness = readiness(
            project_root=root,
            registry_path=registry_path,
            database_path=None,
            field_registry=active_field_registry_path,
            decision_at_utc=created_text,
        )
        checks = {
            "registry": registry_verification["status"],
            "technical_demo": raw_demo.get("status"),
            "replay": replay.get("status"),
            "verify_log": log_verification.get("status"),
        }
        if set(checks.values()) != {"passed"}:
            raise RuntimeError(f"shadow readiness technical checks failed: {checks}")

        demo_summary = _demo_summary(raw_demo, demo_database)
        blockers = _collect_blockers(source_manifest, runtime_readiness)
        parent_after = _parent_snapshot(root)
        if not _parent_snapshots_equal(parent_before, parent_after):
            raise RuntimeError("parent runs changed during shadow readiness")
        parent_after["identical_to_before"] = True

        raw_demo_path = stage / "technical_demo_raw.json"
        replay_path = stage / "replay.json"
        verification_path = stage / "verification.json"
        runtime_readiness_path = stage / "runtime_readiness.json"
        _json_write(raw_demo_path, raw_demo)
        _json_write(replay_path, replay)
        _json_write(verification_path, log_verification)
        _json_write(runtime_readiness_path, runtime_readiness)

        combined_tests: dict[str, Any] = {
            "status": "passed",
            "orchestration_checks": checks,
            "technical_demo_decisions_checked": replay.get("decisions_checked"),
            "synthetic_only": True,
        }
        if test_summary is not None:
            combined_tests["repository_test_run"] = dict(test_summary)
            if test_summary.get("status") not in {None, "passed"}:
                combined_tests["status"] = "failed"
                raise RuntimeError("supplied repository test summary is not passed")

        relative_run = destination.relative_to(root).as_posix() if destination.is_relative_to(root) else destination.name
        commands = (
            f".venv/bin/python -m agro_phenology.shadow_cli readiness --project-root . --registry {relative_run}/shadow_registry.json",
            f".venv/bin/python -m agro_phenology.shadow_cli replay-from-log --project-root . --registry {relative_run}/shadow_registry.json --database {relative_run}/technical_demo/shadow.sqlite",
            f".venv/bin/python -m agro_phenology.shadow_cli verify-log --project-root . --registry {relative_run}/shadow_registry.json --database {relative_run}/technical_demo/shadow.sqlite",
            f".venv/bin/python -m agro_phenology.shadow_readiness_pipeline verify-run --run-dir {relative_run}",
            f".venv/bin/python -m agro_phenology.shadow_readiness_pipeline export-review-package --run-dir {relative_run} --output <new-output.zip>",
        )
        generate_shadow_readiness_run(
            destination,
            registry=registry,
            source_compatibility_manifest=source_manifest,
            demo_summary=demo_summary,
            test_summary=combined_tests,
            blockers=blockers,
            created_at_utc=created_text,
            reproduction_commands=commands,
            execution_context={
                "orchestrator": "agro_phenology.shadow_readiness_pipeline",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "issue_local_date": str(issue_day),
                "network_used": False,
                "schedule_activated": False,
                "notifications_sent": 0,
            },
            technical_demo_artifacts={
                "shadow.sqlite": demo_database,
                "technical_demo_raw.json": raw_demo_path,
                "replay.json": replay_path,
                "verification.json": verification_path,
                "runtime_readiness.json": runtime_readiness_path,
            },
            supplemental_json={
                "parent_integrity_before.json": parent_before,
                "parent_integrity_after.json": parent_after,
                "registry_verification.json": registry_verification,
                "contracts/shadow_readiness_contract.json": contract,
                "source_snapshot.json": source_snapshot,
            },
        )

    final_integrity = verify_readiness_run(destination)
    final_parent_check = _parent_snapshot(root)
    if not _parent_snapshots_equal(parent_after, final_parent_check):
        raise RuntimeError("parent runs changed while final readiness bundle was written")
    return {
        "status": "complete",
        "run_id": selected_run_id,
        "run_dir": str(destination),
        "manifest_sha256": final_integrity["manifest_sha256"],
        "outputs_checked": final_integrity["outputs_checked"],
        "weather_branch_executable_today": bool(
            source_manifest.get("weather_branch_executable_today", False)
        ),
        "prospective_live_run_executed": False,
        "network_used": False,
        "notifications_sent": 0,
        "schedule_activated": False,
        "blocker_count": len(blockers["blockers"]),
        "technical_demo": demo_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agro-late-blight-shadow-readiness",
        description="Build or inspect an offline potato late-blight shadow readiness run.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="create a new immutable readiness result")
    run.add_argument("--project-root", default=".")
    run.add_argument("--results-root")
    run.add_argument("--run-id")
    run.add_argument("--issue-date")
    run.add_argument("--created-at")
    run.add_argument("--field-registry")
    run.add_argument("--test-summary")
    run.add_argument("--output")

    verify = sub.add_parser("verify-run", help="verify every hash in an existing run")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--output")

    export = sub.add_parser(
        "export-review-package", help="copy the verified sanitised review zip"
    )
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output", required=True)
    return parser


def _emit(payload: Mapping[str, Any], destination: str | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination is None:
        sys.stdout.write(rendered)
        return
    path = Path(destination)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        supplied_tests = (
            _json_object(Path(args.test_summary), label="test summary")
            if args.test_summary
            else None
        )
        result = run_shadow_readiness(
            args.project_root,
            results_root=args.results_root,
            run_id=args.run_id,
            issue_local_date=args.issue_date,
            created_at_utc=args.created_at,
            active_field_registry_path=args.field_registry,
            test_summary=supplied_tests,
        )
    elif args.command == "verify-run":
        result = verify_readiness_run(args.run_dir)
    elif args.command == "export-review-package":
        result = export_review_package(args.run_dir, args.output)
    else:  # pragma: no cover - argparse enforces choices
        raise AssertionError(args.command)
    _emit(result, getattr(args, "output", None) if args.command != "export-review-package" else None)
    return 0 if result.get("status") in {"complete", "passed", "exported"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "export_review_package",
    "main",
    "next_readiness_run_id",
    "run_shadow_readiness",
    "verify_readiness_run",
]
