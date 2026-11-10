"""Collect external research data without fitting or evaluating any model.

Run ``collect --collection-id <new_version>``; an interrupted version can be
resumed. A completed collection is sealed and checked, never silently updated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import platform
import shutil

import pandas as pd

from .external_validation_io import ArchiveClient, DownloadError, digest_file, file_inventory, utc_now, write_json


PROVIDERS = {name: f"agro_phenology.external_validation_{name}" for name in ("zenodo", "insects", "russia", "gbif")}


def check(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    failures = []
    for relative, expected in manifest.get("file_inventory", {}).items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest_file(path) != expected["sha256"]:
            failures.append(relative)
    return {"status": "passed" if manifest.get("status") in {"complete", "complete_with_documented_gaps"} and not failures else "failed",
            "checked_files": len(manifest.get("file_inventory", {})), "mismatches": failures,
            "collection_status": manifest.get("status")}


def _validate_source(source: dict) -> None:
    name = source["source_id"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid source ID")
    names = set()
    for spec in source.get("files", []):
        name = spec["name"]
        if not name or Path(name).name != name or name in names:
            raise ValueError(f"Unsafe or duplicate file name: {name}")
        names.add(name)


def _download(client, source: dict, spec: dict) -> dict:
    relative = f"raw/{source['source_id']}/files/{spec['name']}"
    try:
        path = client.fetch_file(spec["url"], relative, expected_checksum=spec.get("expected_checksum"), max_bytes=int(spec.get("max_bytes", 64_000_000)))
        return {"path": relative, "status": "downloaded", "bytes": path.stat().st_size, "sha256": digest_file(path)}
    except OSError as error:
        return {"path": relative, "status": "unavailable", "error": str(error)}


def _write_report(root: Path, catalog: list[dict], results: dict) -> None:
    lines = ["# Внешние данные для проверки моделей AgroPhenology", "", "Сбор на 10 ноября 2026 года. Локальный исследовательский архив; модели не обучались и не оценивались.", "",
             "Исходные ответы сохранены в `raw/`, подготовленные таблицы — в `normalized/`. `catalog.json` содержит источники, лицензии, ограничения и ссылки; `manifest.json` — хеши файлов. Повторный запуск после остановки использует проверенный кеш и продолжает загрузки. Завершённая версия не обновляется: для новых данных нужен новый collection-id.", "",
             "## Состав коллекции", "", "| Набор | Статус | Файлы | Строки подготовленных данных | Тип свидетельства |", "|---|---|---:|---:|---|"]
    for source in catalog:
        result = results.get(source["source_id"], {})
        stats = result.get("normalization", {})
        title = source["title"].replace("|", "/").replace("\n", " ")
        lines.append(f"| [{title}]({source['landing_url']}) | {result.get('status', source['access_status'])} | {result.get('downloaded_files', 0)}/{len(source.get('files', []))} | {stats.get('rows', '—')} | {source.get('evidence_kind', 'unknown')} |")
    lines += ["", "Строки разных наборов имеют разный смысл и не складываются в число независимых случаев болезни. Уловы, стадии насекомого, поражённость делянки, региональное сообщение и фотографическая находка — разные исходы.", "",
              "## Что можно проверять", "",
              "Полевые испытания IPM Decisions: развитие поражения и первая зарегистрированная положительная оценка, с отдельным анализом контрольных/обработанных делянок. Делянки одного опыта не являются независимыми тестовыми годами. Untreated означает отсутствие обработки, а не доказанное отсутствие искусственной инокуляции.", "",
              "Названия авторов «18 опытов», «89 опытов», «36 опытов» сохранены. Они не приравниваются к автоматически построенным техническим группам: номера Trial повторяются между странами/годами/координатами. site_id теперь определяется местом, trial_id — местом, годом и исходным кодом опыта; plot_id ограничен соответствующим опытом. Соответствие авторским независимым опытам ещё требует предметного аудита.", "",
              "Насекомые: реальные уловы или учёты стадий, с сохранением нулей и неизвестных значений. Biofix, яйца, личинки и имаго различаются; улов взрослой особи не заменяет повреждение растения. Годовые суммы не проверяют дату первого лёта.", "",
              "Региональные сообщения Россельхозцентра и GBIF — вспомогательные источники присутствия/сезонности. Дата публикации не подменяет дату наблюдения. Отсутствие сообщения не является отрицательным осмотром. Записи GBIF могут дублировать iNaturalist и другие первоначальные источники; число наборов не является числом независимых внешних проверок.", "",
              "UC ANR degree-day challenge проверяет численную реализацию СЭТ на эталонных температурах. Это не полевая биологическая валидация.", "",
              "Синтетические и зашумлённые варианты AgriDataValue сохранены только внутри исходного архива и исключены из нормализованных исходов. Таблицы без точной даты не получают вымышленную дату 1 января.", "",
              "## Доступность и ограничения", ""]
    for source in catalog:
        result = results.get(source["source_id"], {})
        lines.append(f"### {source['source_id']}")
        license_info = source.get("license", {})
        lines += ["", f"Источник: {source['landing_url']}", "", f"Доступ: {source['access_status']}. Лицензия/условия: {json.dumps(license_info, ensure_ascii=False)}.", ""]
        for item in source.get("limitations", []) + result.get("normalization", {}).get("limitations", []):
            lines.append(f"- {item}")
        for failure in result.get("download_failures", []):
            lines.append(f"- Не получен `{failure['path']}`: {failure['error']}")
        if result.get("normalization_error"):
            lines.append(f"- Подготовка таблиц не завершена: {result['normalization_error']}")
        for error in source.get("discovery_errors", []):
            lines.append(f"- Ограничение обнаружения источников: {json.dumps(error, ensure_ascii=False)}")
        lines.append("")
    lines += ["## Погода и дальнейшая валидация", "",
              "Встроенная погода сохранена, где она присутствует в исходном наборе. Для остальных площадок отдельный сборщик может получить почасовую ERA5 по публичным координатам и периодам наблюдений. Реанализ нельзя выдавать за доступный в прошлом прогноз: для проверки предупреждения заранее потребуется отдельный архив прогнозных выпусков с временем доступности. Автоматически объединять источники погоды с замороженными моделями проекта нельзя.", "",
              "Лицензия данных сохраняется на уровне источника, а у GBIF — записи. CC BY-NC/NC-SA и отдельные требования авторов не трактуются как разрешение на любое коммерческое использование или публикацию. Сырые таблицы, координаты и тексты не отправлялись в репозиторий или внешние чаты.", "",
              "## Команды", "", "```bash", "PYTHONPATH=src .venv/bin/python -m agro_phenology.external_validation_collection collect --collection-id <new_collection_id>", f"PYTHONPATH=src .venv/bin/python -m agro_phenology.external_validation_collection check --collection-dir {root}", "```", "",
              "Для возобновления прерванного сбора повторить ту же команду с тем же collection-id. Закрытые записи и неудавшиеся запросы отражены в каталоге; синтетика и ссылки без скачанных файлов не считаются полученными полевыми данными.", ""]
    (root / "REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")


def collect(project_root: Path, collection_id: str, providers: list[str], workers: int = 3) -> Path:
    if not collection_id or Path(collection_id).name != collection_id or collection_id in {".", ".."}:
        raise ValueError("collection-id must be one new directory name")
    root = project_root / "data/external_validation" / collection_id
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if prior.get("status") in {"complete", "complete_with_documented_gaps"}:
        print(json.dumps(check(root), ensure_ascii=False, indent=2), flush=True)
        return root
    lock = root / ".collection.lock"
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f"Collection is running in process {pid}")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    manifest = {"collection_id": collection_id, "status": "collecting", "started_at_utc": prior.get("started_at_utc", utc_now()), "last_started_at_utc": utc_now(),
                "no_model_training": True, "no_publication": True, "providers": providers, "python_version": platform.python_version()}
    write_json(manifest_path, manifest)
    try:
        client = ArchiveClient(root)
        module_names = ["collection", "io", *providers]
        source_files = sorted(project_root / f"src/agro_phenology/external_validation_{name}.py" for name in module_names)
        source_hashes = {str(path.relative_to(project_root)): digest_file(path) for path in source_files}
        catalog, source_providers = [], {}
        for name in providers:
            print(f"Обнаружение источников: {name}", flush=True)
            module = importlib.import_module(PROVIDERS[name])
            try:
                discovered = module.discover(client)
            except (OSError, ValueError, KeyError) as error:
                discovered = [{"source_id": f"{name}_discovery_unavailable", "provider": name, "title": f"{name}: discovery unavailable", "landing_url": "", "access_status": "unavailable", "license": {}, "evidence_kind": "unavailable", "limitations": [str(error)], "files": []}]
            for source in discovered:
                _validate_source(source)
                if source["source_id"] in source_providers:
                    raise ValueError(f"Duplicate source: {source['source_id']}")
                source_providers[source["source_id"]] = module
                catalog.append(source)
            write_json(root / "catalog.json", catalog)
        jobs = [(s, f) for s in catalog for f in s.get("files", [])]
        outcomes = []
        print(f"Скачивание/проверка {len(jobs)} файлов из {len(catalog)} записей каталога", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(_download, client, source, spec): (source, spec) for source, spec in jobs}
            for index, future in enumerate(as_completed(pending), 1):
                outcome = future.result()
                outcomes.append(outcome)
                if index % 10 == 0 or index == len(jobs) or outcome["status"] != "downloaded":
                    print(f"  Файлы {index}/{len(jobs)}: {outcome['status']} {outcome['path']}", flush=True)
        results = {}
        for source in catalog:
            sid = source["source_id"]
            relevant = [r for r in outcomes if r["path"].startswith(f"raw/{sid}/files/")]
            failures = [r for r in relevant if r["status"] != "downloaded"]
            result = {"status": source["access_status"], "downloaded_files": len(relevant) - len(failures), "download_failures": failures}
            results[sid] = result
            if result["downloaded_files"]:
                print(f"Подготовка таблиц: {sid}", flush=True)
                try:
                    stats = source_providers[sid].normalize(source, root / "raw" / sid, root / "normalized" / sid)
                    result["normalization"] = stats
                    result["status"] = "partial" if failures else "collected"
                except Exception as error:
                    result["status"] = "raw_only_normalization_failed"
                    result["normalization_error"] = f"{type(error).__name__}: {error}"
                    print(f"  Подготовка не завершена: {result['normalization_error']}", flush=True)
            write_json(root / "collection_results.json", results)
        rows = [{"source_id": s["source_id"], "title": s["title"], "access_status": s["access_status"],
                 "collection_status": results[s["source_id"]]["status"], "downloaded_files": results[s["source_id"]]["downloaded_files"],
                 "normalized_rows": results[s["source_id"]].get("normalization", {}).get("rows"), "evidence_kind": s.get("evidence_kind"),
                 "license": json.dumps(s.get("license"), ensure_ascii=False), "landing_url": s["landing_url"]} for s in catalog]
        pd.DataFrame(rows).to_csv(root / "inventory.csv", index=False)
        _write_report(root, catalog, results)
        for path in source_files:
            if digest_file(path) != source_hashes[str(path.relative_to(project_root))]:
                raise RuntimeError(f"Source changed during collection: {path.name}; resume after completing edits")
            dest = root / "source_snapshot" / path.relative_to(project_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        for path in sorted((project_root / "tests").glob("test_external_validation_*.py")):
            dest = root / "source_snapshot" / path.relative_to(project_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        gap = any(r["status"] != "collected" for r in results.values())
        lock.unlink(missing_ok=True)
        manifest.update(status="complete_with_documented_gaps" if gap else "complete", completed_at_utc=utc_now(), source_hashes=source_hashes,
                        source_count=len(catalog), downloaded_files=sum(r["downloaded_files"] for r in results.values()),
                        normalized_sources=sum("normalization" in r for r in results.values()))
        manifest["file_inventory"] = file_inventory(root)
        write_json(manifest_path, manifest)
        print(json.dumps({k: manifest[k] for k in ("status", "source_count", "downloaded_files", "normalized_sources")}, ensure_ascii=False, indent=2), flush=True)
        print(root, flush=True)
        return root
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}", last_update_utc=utc_now())
        write_json(manifest_path, manifest)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("collect")
    start.add_argument("--collection-id", required=True)
    start.add_argument("--providers", default=",".join(PROVIDERS))
    start.add_argument("--workers", type=int, default=3)
    verify = commands.add_parser("check")
    verify.add_argument("--collection-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        providers = args.providers.split(",")
        if any(p not in PROVIDERS for p in providers) or not 1 <= args.workers <= 6:
            parser.error("Unknown provider or invalid worker count")
        collect(Path(__file__).resolve().parents[2], args.collection_id, providers, args.workers)
        return 0
    result = check(args.collection_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
