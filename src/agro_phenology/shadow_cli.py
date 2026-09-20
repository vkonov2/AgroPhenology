"""Command-line interface for the offline late-blight shadow runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence
import zipfile

from .shadow_engine import (
    PROSPECTIVE_MODE,
    readiness,
    replay_from_log,
    run_once,
    technical_demo,
    verify_log,
)
from .shadow_storage import ShadowStorage


def _load_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _write_result(payload: Any, output: str | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")


def _add_common_registry(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--registry", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agro-late-blight-shadow",
        description=(
            "Offline shadow runtime for frozen potato late-blight models; "
            "contains no network, scheduler, or delivery adapter."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ready = sub.add_parser("readiness", help="verify frozen artifacts and input readiness")
    _add_common_registry(ready)
    ready.add_argument("--database")
    ready.add_argument("--field-registry")
    ready.add_argument("--decision-at")
    ready.add_argument("--output")

    verify = sub.add_parser("verify", help="alias for readiness")
    _add_common_registry(verify)
    verify.add_argument("--database")
    verify.add_argument("--field-registry")
    verify.add_argument("--decision-at")
    verify.add_argument("--output")

    run = sub.add_parser("run-once", help="run one actual slot from saved inputs only")
    _add_common_registry(run)
    run.add_argument("--database", required=True)
    run.add_argument("--field-registry", required=True)
    run.add_argument(
        "--decision-at",
        help="actual UTC decision timestamp; defaults to the invocation time",
    )
    run.add_argument("--output")

    replay = sub.add_parser("replay-from-log", help="replay immutable saved decisions")
    _add_common_registry(replay)
    replay.add_argument("--database", required=True)
    replay.add_argument("--output")

    check = sub.add_parser("verify-log", help="verify storage, state chains and replay")
    _add_common_registry(check)
    check.add_argument("--database", required=True)
    check.add_argument("--output")

    demo = sub.add_parser(
        "technical-demo", help="offline demonstration with synthetic identifiers and weather"
    )
    _add_common_registry(demo)
    demo.add_argument("--database", required=True)
    demo.add_argument(
        "--offline",
        action="store_true",
        help="explicit marker; the command is always offline",
    )
    demo.add_argument("--output")

    export = sub.add_parser(
        "export-review-package",
        help="verify and optionally copy the prebuilt redacted review archive",
    )
    export.add_argument("--run-dir", required=True)
    export.add_argument("--destination")
    export.add_argument("--output")

    archive = sub.add_parser(
        "archive-source", help="archive already retrieved raw bytes; performs no network request"
    )
    archive.add_argument("--database", required=True)
    archive.add_argument("--record-json", required=True)
    archive.add_argument("--raw-file", required=True)
    archive.add_argument("--output")

    observation = sub.add_parser(
        "append-observation", help="append a versioned predictor/forecast/outcome/metadata record"
    )
    observation.add_argument("--database", required=True)
    observation.add_argument("--record-json", required=True)
    observation.add_argument("--output")
    return parser


def _archive_source(args: argparse.Namespace) -> dict[str, Any]:
    record = _load_object(args.record_json, name="source record")
    raw_file = Path(args.raw_file)
    if not raw_file.is_file():
        raise FileNotFoundError(raw_file)
    required = {
        "retrieval_id",
        "source",
        "request_key",
        "media_type",
        "retrieval_started_at_utc",
        "retrieval_completed_at_utc",
        "first_seen_at_utc",
        "ingested_at_utc",
        "valid_from_utc",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"source record misses {missing}")
    completed = record["retrieval_completed_at_utc"]
    stored = ShadowStorage(args.database).record_retrieval(
        retrieval_id=record["retrieval_id"],
        source=record["source"],
        request_key=record["request_key"],
        body=raw_file.read_bytes(),
        media_type=record["media_type"],
        initialized_at_utc=record.get("initialized_at_utc"),
        published_at_utc=record.get("published_at_utc"),
        retrieval_started_at_utc=record["retrieval_started_at_utc"],
        retrieval_completed_at_utc=completed,
        retrieved_at_utc=completed,
        first_seen_at_utc=record["first_seen_at_utc"],
        ingested_at_utc=record["ingested_at_utc"],
        valid_from_utc=record["valid_from_utc"],
        valid_to_utc=record.get("valid_to_utc"),
        metadata=record.get("metadata", {}),
    )
    # Raw bytes never go to stdout or the machine-readable summary.
    return {
        "status": "archived",
        "network_used": False,
        "retrieval_id": stored["retrieval_id"],
        "content_sha256": stored["content_sha256"],
        "byte_length": raw_file.stat().st_size,
        "retrieval_started_at_utc": stored["retrieval_started_at_utc"],
        "retrieval_completed_at_utc": stored["retrieval_completed_at_utc"],
        "first_seen_at_utc": stored["first_seen_at_utc"],
        "ingested_at_utc": stored["ingested_at_utc"],
    }


def _append_observation(args: argparse.Namespace) -> dict[str, Any]:
    record = _load_object(args.record_json, name="observation record")
    required = {
        "observation_id",
        "observation_key",
        "information_role",
        "payload",
        "retrieval_started_at_utc",
        "retrieval_completed_at_utc",
        "first_seen_at_utc",
        "ingested_at_utc",
        "valid_from_utc",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"observation record misses {missing}")
    role = record["information_role"]
    if role == "forecast" and record["payload"].get("archive_only") is not True:
        raise ValueError("forecast observations must explicitly set payload.archive_only=true")
    completed = record["retrieval_completed_at_utc"]
    stored = ShadowStorage(args.database).append_observation(
        observation_id=record["observation_id"],
        observation_key=record["observation_key"],
        payload=record["payload"],
        information_role=role,
        source_retrieval_id=record.get("source_retrieval_id"),
        initialized_at_utc=record.get("initialized_at_utc"),
        published_at_utc=record.get("published_at_utc"),
        retrieval_started_at_utc=record["retrieval_started_at_utc"],
        retrieval_completed_at_utc=completed,
        retrieved_at_utc=completed,
        first_seen_at_utc=record["first_seen_at_utc"],
        ingested_at_utc=record["ingested_at_utc"],
        valid_from_utc=record["valid_from_utc"],
        valid_to_utc=record.get("valid_to_utc"),
    )
    return {
        "status": "appended",
        "network_used": False,
        "observation_id": stored["observation_id"],
        "observation_key": stored["observation_key"],
        "version": stored["version"],
        "information_role": stored["information_role"],
        "retrieval_started_at_utc": stored["retrieval_started_at_utc"],
        "retrieval_completed_at_utc": stored["retrieval_completed_at_utc"],
        "first_seen_at_utc": stored["first_seen_at_utc"],
        "ingested_at_utc": stored["ingested_at_utc"],
    }


def _export_review_package(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    source = run_dir / "review_package.zip"
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = run_dir / "execution_manifest.json"
    manifest_verified = False
    if manifest_path.is_file():
        manifest = _load_object(manifest_path, name="execution manifest")
        expected = (
            manifest.get("output_hashes", {})
            .get("review_package.zip", {})
            .get("sha256")
        )
        if expected is not None and expected != digest:
            raise ValueError("review_package.zip hash differs from execution_manifest.json")
        manifest_verified = expected == digest
    forbidden_suffixes = {
        ".sqlite",
        ".db",
        ".parquet",
        ".csv",
        ".cbm",
        ".joblib",
        ".pkl",
        ".pickle",
    }
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"review archive has unsafe member {name!r}")
            if relative.suffix.lower() in forbidden_suffixes:
                raise ValueError(f"review archive contains forbidden artifact {name!r}")
    destination: Path | None = None
    if args.destination:
        destination = Path(args.destination).resolve()
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite export: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return {
        "status": "passed",
        "source": str(source),
        "sha256": digest,
        "bytes": source.stat().st_size,
        "members": len(names),
        "manifest_hash_verified": manifest_verified,
        "copied_to": None if destination is None else str(destination),
        "contains_raw_or_models": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"readiness", "verify"}:
        result = readiness(
            project_root=args.project_root,
            registry_path=args.registry,
            database_path=args.database,
            field_registry=args.field_registry,
            decision_at_utc=args.decision_at,
        )
    elif args.command == "run-once":
        result = run_once(
            project_root=args.project_root,
            registry_path=args.registry,
            field_registry=args.field_registry,
            database_path=args.database,
            decision_at_utc=args.decision_at or datetime.now(timezone.utc),
            mode=PROSPECTIVE_MODE,
        )
    elif args.command == "replay-from-log":
        result = replay_from_log(
            database_path=args.database,
            registry_path=args.registry,
            project_root=args.project_root,
        )
    elif args.command == "verify-log":
        result = verify_log(
            database_path=args.database,
            registry_path=args.registry,
            project_root=args.project_root,
        )
    elif args.command == "technical-demo":
        result = technical_demo(
            project_root=args.project_root,
            registry_path=args.registry,
            database_path=args.database,
        )
    elif args.command == "archive-source":
        result = _archive_source(args)
    elif args.command == "append-observation":
        result = _append_observation(args)
    elif args.command == "export-review-package":
        result = _export_review_package(args)
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(args.command)
    _write_result(result, getattr(args, "output", None))
    return 0 if result.get("status") not in {"failed", "blocked"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
