"""Auditable, resumable downloads for a local external validation collection.

Only public HTTP(S) resources are read. Existing verified responses are reused
within a collection version; a new response version requires a new collection.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse

import requests


class DownloadError(OSError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


class ArchiveClient:
    def __init__(self, root: Path, *, retries: int = 3, timeout: float = 45, pause: float = .15, session=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retries, self.timeout, self.pause = retries, timeout, pause
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "AgroPhenology-Research/1.0 (public-data-validation; no-model-training)", "Accept-Encoding": "identity"})
        self.log_path = self.root / "acquisition_log.jsonl"

    def _path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise DownloadError("Download path escapes the collection")
        return path

    def _log(self, **event) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at_utc": utc_now(), **event}, ensure_ascii=False) + "\n")

    def fetch_file(self, url: str, relative_path: str, *, expected_checksum: str | None = None, max_bytes: int = 64_000_000, accept_compression: bool = False) -> Path:
        if urlparse(url).scheme not in {"http", "https"}:
            raise DownloadError("Only public HTTP(S) download URLs are accepted")
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt = path.with_name(path.name + ".download.json")
        if path.is_file():
            if not receipt.is_file():
                raise DownloadError(f"Existing unverified file will not be overwritten: {relative_path}")
            saved = json.loads(receipt.read_text())
            if saved["requested_url"] != url or saved["sha256"] != digest_file(path):
                raise DownloadError(f"Cached response changed: {relative_path}")
            self._check_expected(path, expected_checksum)
            return path
        part = path.with_name(path.name + ".part")
        progress_path = path.with_name(path.name + ".resume.json")
        if part.is_file() and receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved.get("requested_url") == url and saved.get("sha256") == digest_file(part):
                self._check_expected(part, expected_checksum)
                os.replace(part, path)
                progress_path.unlink(missing_ok=True)
                self._log(status="recovered_atomic_publish", path=relative_path)
                return path
        last_error = None
        for attempt in range(self.retries):
            response = None
            try:
                progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
                if progress and progress.get("url") != url:
                    raise DownloadError("Partial response belongs to a different URL")
                offset = part.stat().st_size if part.is_file() else 0
                headers = {"Accept-Encoding": "gzip, deflate"} if accept_compression else {}
                if offset and (accept_compression or progress.get("content_encoding") not in {None, "", "identity"}):
                    # Range offsets refer to the encoded representation.
                    # Our saved HTTP payload is decoded, so never append a
                    # compressed range to an uncompressed partial response.
                    offset = 0
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                    if progress.get("validator"):
                        headers["If-Range"] = progress["validator"]
                    elif expected_checksum is None:
                        # Without a validator or source checksum a partial
                        # response cannot safely be joined to a fresh version.
                        offset = 0
                        headers = {}
                time.sleep(self.pause)
                response = self.session.get(url, headers=headers, stream=True, timeout=(15, self.timeout), allow_redirects=True)
                if response.status_code == 416 and offset:
                    # An interruption may have happened after the complete
                    # body was written but before the receipt. Restart only
                    # this downloader-owned partial, never a published file.
                    part.unlink(missing_ok=True)
                    progress_path.unlink(missing_ok=True)
                    raise DownloadError("Range already complete or stale; restarting partial response")
                if response.status_code in {401, 403, 404, 410}:
                    raise DownloadError(f"HTTP {response.status_code}: {url}")
                response.raise_for_status()
                total_bytes = None
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                    if not offset or not match or int(match[1]) != offset or int(match[2]) < offset:
                        raise DownloadError("Unexpected partial response range")
                    if match[3] != "*":
                        total_bytes = int(match[3])
                        if int(match[2]) >= total_bytes:
                            raise DownloadError("Invalid partial response total")
                    new_validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                    if progress.get("validator") and new_validator and new_validator != progress["validator"]:
                        part.unlink(missing_ok=True)
                        progress_path.unlink(missing_ok=True)
                        raise DownloadError("Partial response version changed; restarting")
                elif response.status_code == 200:
                    offset = 0  # server does not support a resume: restart .part
                else:
                    raise DownloadError(f"Unexpected HTTP status {response.status_code}")
                length = response.headers.get("Content-Length")
                content_encoding = response.headers.get("Content-Encoding", "identity").lower()
                decoded = content_encoding not in {"", "identity"}
                if length and not decoded and offset + int(length) > max_bytes:
                    raise DownloadError(f"Response exceeds {max_bytes} byte limit")
                write_json(progress_path, {"url": url, "validator": response.headers.get("ETag") or response.headers.get("Last-Modified"), "content_encoding": content_encoding})
                size = offset
                last_progress = time.monotonic()
                with part.open("ab" if offset else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            size += len(chunk)
                            if size > max_bytes:
                                raise DownloadError(f"Response exceeds {max_bytes} byte limit")
                            handle.write(chunk)
                            if size >= 5_000_000 and time.monotonic() - last_progress >= 10:
                                print(f"  Загрузка {relative_path}: {size / 1_000_000:.1f} MB", flush=True)
                                last_progress = time.monotonic()
                    handle.flush()
                    os.fsync(handle.fileno())
                if not size:
                    raise DownloadError("Empty response")
                if length and not decoded and size != offset + int(length):
                    raise DownloadError("Incomplete HTTP response body")
                if total_bytes is not None and size != total_bytes:
                    raise DownloadError("Partial response has not reached its declared total")
                self._check_expected(part, expected_checksum)
                result = {"requested_url": url, "resolved_url": response.url, "retrieved_at_utc": utc_now(), "bytes": size,
                          "sha256": digest_file(part), "expected_checksum": expected_checksum, "content_type": response.headers.get("Content-Type"), "http_content_encoding": content_encoding, "wire_content_length": length,
                          "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"), "resumed_from_bytes": offset}
                # Keep the receipt before atomically publishing the data path.
                write_json(receipt, result)
                os.replace(part, path)
                progress_path.unlink(missing_ok=True)
                self._log(status="downloaded", path=relative_path, **result)
                return path
            except (requests.RequestException, OSError, ValueError) as error:
                last_error = error
                if "Source checksum mismatch" in str(error):
                    part.unlink(missing_ok=True)
                    progress_path.unlink(missing_ok=True)
                self._log(status="failed_attempt", url=url, path=relative_path, attempt=attempt + 1, error=str(error))
                if any(code in str(error) for code in ("HTTP 401", "HTTP 403", "HTTP 404", "HTTP 410")) or "will not be overwritten" in str(error) or "exceeds" in str(error):
                    break
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 4))
            finally:
                if response is not None:
                    response.close()
        raise DownloadError(str(last_error)) from last_error

    @staticmethod
    def _check_expected(path: Path, expected: str | None) -> None:
        if expected:
            algorithm, value = expected.split(":", 1)
            if algorithm not in {"sha256", "md5"} or digest_file(path, algorithm) != value.lower():
                raise DownloadError(f"Source checksum mismatch: {path.name}")

    def get_json(self, url: str, relative_path: str, *, max_bytes: int = 8_000_000):
        path = self.fetch_file(url, relative_path, max_bytes=max_bytes, accept_compression=True)
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            raise DownloadError(f"Not a JSON response: {url}") from error

    def get_text(self, url: str, relative_path: str, *, max_bytes: int = 12_000_000, encoding: str | None = None) -> str:
        path = self.fetch_file(url, relative_path, max_bytes=max_bytes, accept_compression=True)
        raw = path.read_bytes()
        for candidate in ([encoding] if encoding else ["utf-8-sig", "cp1251"]):
            try:
                return raw.decode(candidate)
            except UnicodeError:
                continue
        raise DownloadError(f"Cannot decode response: {url}")


def file_inventory(root: Path) -> dict[str, dict]:
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": digest_file(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"
            and not path.name.endswith((".part", ".tmp", ".resume.json"))}
