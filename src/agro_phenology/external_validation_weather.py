"""Plan and archive explicit ERA5 for dated external field cohorts.

This is retrospective weather reconstruction, not a reconstruction of weather
published before a historical decision. It never substitutes forecasts, IFS,
Best Match, actual future weather, or a manufactured leaf-wetness duration.
"""
from __future__ import annotations

import argparse
from datetime import date
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from .external_validation_io import ArchiveClient, DownloadError, digest_file, file_inventory, utc_now, write_json


AS_OF_DATE = date(2026, 11, 10)
AVAILABLE_END = date(2026, 11, 4)
PRIMARY_SOURCES = (
    "zenodo_6352735", "zenodo_6520484", "zenodo_6520937",
    "daks_cpb_2019_2022", "zenodo_6351794",
)
SECONDARY_SOURCES = ("zenodo_6352394", "zenodo_6521175", "zenodo_6390211")
HOURLY_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "wind_speed_10m", "shortwave_radiation",
)
API_URL = "https://archive-api.open-meteo.com/v1/archive"
FORMAT_VERSION = "external_era5_archive_v1"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _grid(value: float) -> float:
    # Planning deduplication uses an explicit regular ERA5 0.25-degree grid.
    # Returned provider coordinates are retained as authoritative metadata.
    return math.floor(float(value) * 4 + .5) / 4


def build_plan(collection_dir: Path, *, include_secondary: bool = False) -> dict:
    """Read dated normalized observations and public locations; do no I/O network."""
    collection_dir = Path(collection_dir).resolve()
    allowed = PRIMARY_SOURCES + (SECONDARY_SOURCES if include_secondary else ())
    inputs: dict[str, str] = {}
    manifest_path = collection_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"complete", "complete_with_documented_gaps"}:
            raise ValueError("Primary collection must be finalized before weather planning")
        from .external_validation_collection import check as check_collection
        if check_collection(collection_dir)["status"] != "passed":
            raise ValueError("Primary collection integrity check failed; do not match weather to changed inputs")
        inputs[str(manifest_path)] = digest_file(manifest_path)
    requests: dict[tuple[float, float, int], dict] = {}
    mappings, excluded = [], []
    normalized = collection_dir / "normalized"
    for folder in sorted(normalized.iterdir()) if normalized.is_dir() else []:
        sid = folder.name
        if sid not in allowed:
            excluded.append({"source_id": sid, "reason": "outside_dated_field_cohort_allowlist"})
            continue
        observation_path, location_path = folder / "observations.parquet", folder / "locations.parquet"
        if not observation_path.is_file() or not location_path.is_file():
            excluded.append({"source_id": sid, "reason": "normalized_observations_or_locations_missing"})
            continue
        observations = pd.read_parquet(observation_path)
        locations = pd.read_parquet(location_path)
        if not {"site_id", "observation_date"}.issubset(observations) or not {"site_id", "latitude", "longitude"}.issubset(locations):
            excluded.append({"source_id": sid, "reason": "exact_assessment_dates_or_geographic_coordinates_missing"})
            continue
        inputs[str(observation_path)] = digest_file(observation_path)
        inputs[str(location_path)] = digest_file(location_path)
        dates = pd.to_datetime(observations.observation_date, errors="raise", utc=True)
        dated = observations.loc[dates.notna(), ["site_id"]].copy()
        dated["year"] = dates.loc[dates.notna()].dt.year
        dated = dated.dropna(subset=["site_id"]).drop_duplicates()
        sites = locations[["site_id", "latitude", "longitude"]].copy()
        sites["latitude"] = pd.to_numeric(sites.latitude, errors="raise")
        sites["longitude"] = pd.to_numeric(sites.longitude, errors="raise")
        sites = sites.dropna().drop_duplicates()
        if not sites.latitude.between(-90, 90).all() or not sites.longitude.between(-180, 180).all():
            raise ValueError(f"Invalid source coordinate range: {sid}")
        if sites.groupby("site_id").size().gt(1).any():
            raise ValueError(f"Ambiguous site_id has multiple source coordinates: {sid}; fix source normalization before weather matching")
        rows = dated.merge(sites, on="site_id", how="inner", validate="many_to_one")
        if rows.empty:
            excluded.append({"source_id": sid, "reason": "no_matching_dated_georeferenced_sites"})
        priority = allowed.index(sid)
        for row in rows.itertuples(index=False):
            year = int(row.year)
            if year < 1940 or year > AS_OF_DATE.year:
                excluded.append({"source_id": sid, "site_id": str(row.site_id), "year": year, "reason": "outside_ERA5_available_years"})
                continue
            start, end = date(year, 1, 1), min(date(year, 12, 31), AVAILABLE_END)
            if end < start:
                continue
            lat, lon = _grid(row.latitude), _grid(row.longitude)
            key = (lat, lon, year)
            if key not in requests:
                params = {
                    "latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
                    "hourly": ",".join(HOURLY_VARIABLES), "models": "era5", "timezone": "UTC",
                    "cell_selection": "nearest", "elevation": "nan", "wind_speed_unit": "ms",
                }
                request_id = f"era5_{year}_{_stable_hash(params)[:16]}"
                requests[key] = {
                    "request_id": request_id, "year": year, "priority": priority,
                    "requested_grid_latitude": lat, "requested_grid_longitude": lon,
                    "start_date": start.isoformat(), "end_date": end.isoformat(),
                    "expected_hours": len(pd.date_range(start, end, freq="D")) * 24,
                    "partial_year": end < date(year, 12, 31), "parameters": params,
                    "url": API_URL + "?" + urlencode(params),
                    "source_ids": [],
                }
            item = requests[key]
            item["priority"] = min(item["priority"], priority)
            item["source_ids"] = sorted(set(item["source_ids"]) | {sid})
            mappings.append({
                "source_id": sid, "site_id": str(row.site_id), "year": year,
                "source_latitude": float(row.latitude), "source_longitude": float(row.longitude),
                "requested_grid_latitude": lat, "requested_grid_longitude": lon,
                "request_id": item["request_id"],
            })
    ordered = sorted(requests.values(), key=lambda x: (x["priority"], x["year"], x["requested_grid_latitude"], x["requested_grid_longitude"]))
    body = {
        "format_version": FORMAT_VERSION, "collection_dir": str(collection_dir),
        "as_of_date": AS_OF_DATE.isoformat(), "safe_era5_availability_end": AVAILABLE_END.isoformat(),
        "include_secondary": include_secondary, "model": "era5", "timezone": "UTC",
        "hourly_variables": list(HOURLY_VARIABLES), "requests": ordered,
        "site_request_mapping": sorted(mappings, key=lambda x: (x["source_id"], x["site_id"], x["year"], x["request_id"])),
        "excluded_sources": excluded, "input_hashes": inputs,
        "grid_assignment": "regular_0.25_degree_nearest_request_coordinates;returned_provider_coordinates_are_authoritative",
        "purpose": "retrospective_weather_reconstruction_not_historical_operational_availability",
        "previous_year_december": "not_added_unless_that_year_has_source_observations",
    }
    body["plan_sha256"] = _stable_hash(body)
    return body


def plan(collection_dir: Path, output_dir: Path, *, include_secondary: bool = False) -> dict:
    """Save one stable plan, refusing to replace an unrelated/existing version."""
    result = build_plan(collection_dir, include_secondary=include_secondary)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "weather_plan.json"
    if plan_path.is_file():
        saved = json.loads(plan_path.read_text())
        if saved["plan_sha256"] != result["plan_sha256"]:
            raise ValueError("Weather plan or inputs changed; choose a new output directory")
        return saved
    if any(output_dir.iterdir()):
        raise ValueError("Output directory contains unrelated files; refusing to overwrite")
    write_json(plan_path, result)
    pd.DataFrame(result["site_request_mapping"]).to_parquet(output_dir / "site_request_mapping.parquet", index=False)
    write_json(output_dir / "weather_manifest.json", {
        "format_version": FORMAT_VERSION, "status": "planned", "plan_sha256": result["plan_sha256"],
        "created_at_utc": utc_now(), "input_hashes": result["input_hashes"], "request_results": {},
    })
    return result


def _normalize_response(request: dict, response: dict, *, retrieved_at_utc: str | None = None) -> tuple[pd.DataFrame, dict]:
    if response.get("error"):
        raise ValueError(f"Weather provider error: {response.get('reason')}")
    if response.get("utc_offset_seconds", 0) != 0:
        raise ValueError("Weather response is not UTC")
    hourly = response.get("hourly", {})
    missing = {"time", *HOURLY_VARIABLES} - set(hourly)
    if missing:
        raise ValueError(f"Missing ERA5 hourly variables: {sorted(missing)}")
    times = pd.to_datetime(hourly["time"], utc=True, errors="raise")
    expected = pd.date_range(request["start_date"], pd.Timestamp(request["end_date"]) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    if len(times) != len(expected) or not times.equals(expected):
        raise ValueError("Weather timestamps do not exactly cover requested UTC hours")
    lat, lon = float(response["latitude"]), float(response["longitude"])
    if not math.isfinite(lat) or not math.isfinite(lon) or abs(lat - request["requested_grid_latitude"]) > .5 or abs(lon - request["requested_grid_longitude"]) > .5:
        raise ValueError("Weather provider returned unexpected coordinates")
    frame = pd.DataFrame({"request_id": request["request_id"], "valid_time_utc": times})
    coverage = {}
    for variable in HOURLY_VARIABLES:
        if len(hourly[variable]) != len(times):
            raise ValueError(f"ERA5 variable length mismatch: {variable}")
        frame[variable] = pd.to_numeric(pd.Series(hourly[variable]), errors="raise").astype(float)
        coverage[variable] = {"nonmissing_hours": int(frame[variable].notna().sum()), "total_hours": len(frame)}
    frame["provider_latitude"], frame["provider_longitude"] = lat, lon
    metadata = {
        "request_id": request["request_id"], "model_requested": "era5", "timezone": "UTC",
        "requested_grid_latitude": request["requested_grid_latitude"], "requested_grid_longitude": request["requested_grid_longitude"],
        "provider_latitude": lat, "provider_longitude": lon, "provider_elevation": response.get("elevation"),
        "retrieved_at_utc": retrieved_at_utc, "valid_time_start_utc": times[0].isoformat(), "valid_time_end_utc": times[-1].isoformat(),
        "hourly_units": response.get("hourly_units", {}), "hour_coverage": coverage,
        "initialization_time": None, "publication_time": None,
        "availability_semantics": "retrieval_time_is_known;historical_publication_times_are_not_reconstructed",
        "leaf_wetness": "not_observed_not_fabricated",
        "variable_time_semantics": "provider_valid_time;instantaneous_or_preceding_hour_aggregation_is_variable_specific",
    }
    return frame, metadata


def _current_inputs_match(inputs: dict[str, str]) -> list[str]:
    return [name for name, expected in inputs.items() if not Path(name).is_file() or digest_file(Path(name)) != expected]


def fetch(collection_dir: Path, output_dir: Path, *, max_requests: int = 250,
          include_secondary: bool = False, pause_seconds: float = 3.0, client: Any = None,
          retry_failed: bool = False) -> dict:
    """Fetch at most max_requests pending annual cell requests in this invocation."""
    if max_requests < 0:
        raise ValueError("max_requests must be nonnegative")
    output_dir = Path(output_dir).resolve()
    request_plan = plan(collection_dir, output_dir, include_secondary=include_secondary)
    if not request_plan["requests"]:
        raise ValueError("No dated georeferenced allowed cohorts available")
    lock_path = output_dir / ".weather.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another weather fetch is active in this directory") from exc
        manifest_path = output_dir / "weather_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete" or (manifest.get("status") == "complete_with_gaps" and not retry_failed):
            return check(output_dir)
        failures = _current_inputs_match(request_plan["input_hashes"])
        if failures:
            raise ValueError(f"External collection inputs changed: {failures}")
        sources = [Path(__file__), Path(__file__).with_name("external_validation_io.py"), Path(__file__).with_name("external_validation_collection.py")]
        source_hashes = {str(p): digest_file(p) for p in sources}
        if manifest.get("source_hashes") and manifest["source_hashes"] != source_hashes:
            raise ValueError("Weather collector code changed; use a new output directory")
        snapshot_dir = output_dir / "source_snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        for path in sources:
            dest = snapshot_dir / path.name
            if not dest.exists():
                dest.write_bytes(path.read_bytes())
        manifest.update(status="fetching", source_hashes=source_hashes, last_started_at_utc=utc_now())
        write_json(manifest_path, manifest)
        archive = client or ArchiveClient(output_dir, retries=2, pause=pause_seconds)
        outcomes = manifest.setdefault("request_results", {})
        pending = [r for r in request_plan["requests"] if r["request_id"] not in outcomes or
                   (retry_failed and outcomes[r["request_id"]].get("status") == "failed")]
        history = manifest.setdefault("attempt_history", [])
        manifest.pop("stop_reason", None)
        attempted = 0
        try:
            for request in pending[:max_requests]:
                rid = request["request_id"]
                stop_reason = None
                print(f"ERA5 {attempted+1}/{min(len(pending), max_requests)}: {request['year']} {request['requested_grid_latitude']}, {request['requested_grid_longitude']}", flush=True)
                try:
                    response = archive.get_json(request["url"], f"raw/{rid}.json", max_bytes=12_000_000)
                    receipt_path = output_dir / "raw" / f"{rid}.json.download.json"
                    retrieved = json.loads(receipt_path.read_text()).get("retrieved_at_utc") if receipt_path.is_file() else None
                    frame, metadata = _normalize_response(request, response, retrieved_at_utc=retrieved)
                    hourly_path = output_dir / "normalized" / f"{rid}.parquet"
                    hourly_path.parent.mkdir(exist_ok=True)
                    frame.to_parquet(hourly_path, index=False)
                    write_json(output_dir / "request_metadata" / f"{rid}.json", metadata)
                    outcomes[rid] = {"status": "complete", "hours": len(frame),
                                     "missing_variable_hours": sum(frame[v].isna().sum().item() for v in HOURLY_VARIABLES),
                                     "hourly_path": str(hourly_path.relative_to(output_dir)),
                                     "hourly_sha256": digest_file(hourly_path)}
                except (DownloadError, ValueError, KeyError, OSError) as error:
                    outcomes[rid] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    print(f"  Не получено: {outcomes[rid]['error']}", flush=True)
                    stop_reason = _provider_stop_reason(error)
                attempted += 1
                history.append({"request_id": rid, "at_utc": utc_now(), **outcomes[rid]})
                manifest.update(last_update_utc=utc_now(), request_results=outcomes)
                if stop_reason:
                    manifest["stop_reason"] = stop_reason
                write_json(manifest_path, manifest)
                if stop_reason:
                    print(f"  Запросы остановлены: {stop_reason}; оставшиеся ячейки сохранены pending.", flush=True)
                    break
        except BaseException as error:
            manifest.update(status="interrupted", error=f"{type(error).__name__}: {error}", last_update_utc=utc_now(), request_results=outcomes)
            write_json(manifest_path, manifest)
            raise
        failed = sum(r["status"] == "failed" for r in outcomes.values())
        remaining = len(request_plan["requests"]) - len(outcomes)
        if source_hashes != {str(p): digest_file(p) for p in sources}:
            raise ValueError("Collector source changed during fetch")
        manifest.update(status="partial" if remaining else "complete_with_gaps" if failed else "complete",
                        last_update_utc=utc_now(), total_requests=len(request_plan["requests"]),
                        attempted_this_invocation=attempted, complete_requests=len(outcomes)-failed,
                        failed_requests=failed, pending_requests=remaining,
                        hourly_rows=sum(x.get("hours", 0) for x in outcomes.values()))
        _write_report(output_dir, request_plan, manifest)
        manifest["file_inventory"] = {k: v for k, v in file_inventory(output_dir).items()
                                      if k not in {"weather_manifest.json", ".weather.lock"}}
        write_json(manifest_path, manifest)
        return {k: manifest[k] for k in ("status", "total_requests", "complete_requests", "failed_requests", "pending_requests", "hourly_rows")}


def _provider_stop_reason(error: Exception) -> str | None:
    message = str(error)
    if re.search(r"\bHTTP\s+429\b|\b429\s+(?:Client|Server)\s+Error\b", message, flags=re.IGNORECASE) or any(
        phrase in message.casefold() for phrase in ("api request limit", "rate limit")
    ):
        return "provider_rate_limit"
    if re.search(r"\bHTTP\s+(?:401|403)\b|\b(?:401|403)\s+Client\s+Error\b", message, flags=re.IGNORECASE):
        return "provider_authentication_or_access_block"
    return None


def _write_report(output_dir: Path, request_plan: dict, manifest: dict) -> None:
    lines = [
        "# ERA5 для внешних полевых наборов", "",
        "Погода предназначена для ретроспективной проверки опубликованных формул. Это не архив прогнозов и не доказательство доступности ERA5 на историческую дату выпуска предупреждения.", "",
        f"Срез доступности: {AS_OF_DATE.isoformat()}, безопасный последний день ERA5: {AVAILABLE_END.isoformat()}. Запрашивается только models=era5, UTC, ближайшая ячейка; IFS и Best Match не подставляются.", "",
        "Запросы охватывают полный календарный год, включая зиму; 2026 обрезан 4 ноября. Декабрь предыдущего года не добавлен автоматически, если такого года нет среди наблюдений. Измеренное увлажнение листа отсутствует и не создаётся из RH.", "",
        f"Статус: {manifest['status']}. Ячеек-лет: {manifest['total_requests']}; готово {manifest['complete_requests']}; ошибок {manifest['failed_requests']}; осталось {manifest['pending_requests']}. Почасовых строк: {manifest['hourly_rows']}.", "",
        "В site_request_mapping.parquet сохранены исходные публичные координаты участков и связь с запросом. Запросы объединяются по регулярной сетке 0,25° и году; фактические координаты, вернувшиеся у провайдера, сохранены в request_metadata и почасовых файлах. Округление исходных координат и реанализ ограничивают точность погоды у растений.", "",
        "raw содержит неизменяемые ответы с receipt и SHA-256; normalized — только полученные часы и значения, без заполнения пропусков. Единицы измерения и число доступных значений по переменной сохранены в request_metadata. Время получения известно; время первоначальной публикации и инициализации исторических оценок не восстанавливается.", "",
        "Основная коллекция исходов не изменялась. Погодные ответы не являются новыми наблюдениями болезни. Никакие модели не обучались, будущие состоявшиеся значения не выдаются за прогноз.", "",
        "```bash", f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_validation_weather plan --collection-dir {request_plan['collection_dir']} --output-dir {output_dir}",
        f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_validation_weather fetch --collection-dir {request_plan['collection_dir']} --output-dir {output_dir} --max-requests 250",
        f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_validation_weather check --output-dir {output_dir}", "```", "",
        "Повторный fetch продолжает ещё не обработанные запросы, максимум 250 за один вызов. Для явного повторения ошибочных запросов добавьте --retry-failed; история попыток сохраняется. Уже полученные raw-ответы не перезаписываются: повторный разбор читает тот же сохранённый ответ. HTTP429 или общий отказ доступа останавливает текущую пачку, оставляя остальные запросы pending. Ограничение числа запросов относится к ячейкам-годам, возможны HTTP-повторы. Для новых версий данных или кода требуется новый output-dir.",
    ]
    (output_dir / "README_ru.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(output_dir: Path) -> dict:
    output_dir = Path(output_dir).resolve()
    manifest = json.loads((output_dir / "weather_manifest.json").read_text())
    failures = []
    for name, expected in manifest.get("file_inventory", {}).items():
        path = output_dir / name
        if not path.is_file() or digest_file(path) != expected["sha256"]:
            failures.append(name)
    input_failures = _current_inputs_match(manifest.get("input_hashes", {}))
    return {"status": "passed" if manifest.get("status") in {"complete", "complete_with_gaps", "partial"} and not failures and not input_failures else "failed",
            "weather_status": manifest.get("status"), "checked_files": len(manifest.get("file_inventory", {})),
            "checked_inputs": len(manifest.get("input_hashes", {})), "mismatches": failures,
            "input_mismatches": input_failures, "pending_requests": manifest.get("pending_requests"),
            "failed_requests": manifest.get("failed_requests")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "fetch", "check"):
        cmd = sub.add_parser(command)
        if command in {"plan", "fetch"}:
            cmd.add_argument("--collection-dir", type=Path, required=True)
        cmd.add_argument("--output-dir", type=Path, required=True)
        if command in {"plan", "fetch"}:
            cmd.add_argument("--include-secondary", action="store_true")
        if command == "fetch":
            cmd.add_argument("--max-requests", type=int, default=250)
            cmd.add_argument("--pause-seconds", type=float, default=3.0)
            cmd.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan(args.collection_dir, args.output_dir, include_secondary=args.include_secondary)
        print(json.dumps({"planned_requests": len(result["requests"]), "site_year_mappings": len(result["site_request_mapping"]), "plan_sha256": result["plan_sha256"], "requests_by_source": {s: sum(s in r["source_ids"] for r in result["requests"]) for s in PRIMARY_SOURCES + SECONDARY_SOURCES}}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "fetch":
        result = fetch(args.collection_dir, args.output_dir, max_requests=args.max_requests, include_secondary=args.include_secondary, pause_seconds=args.pause_seconds, retry_failed=args.retry_failed)
    else:
        result = check(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("status") == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
