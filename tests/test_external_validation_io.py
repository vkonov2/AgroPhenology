from __future__ import annotations

import hashlib
import json

import pytest
import requests

from agro_phenology.external_validation_io import ArchiveClient, DownloadError


class Response:
    def __init__(self, body=b"abcdef", status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.url = "https://example.org/data.csv"

    def iter_content(self, **kwargs):
        yield self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def close(self):
        pass


class Session:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = iter(responses), [], {}

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def test_verified_cache_never_refetches_and_detects_tampering(tmp_path):
    session = Session([Response()])
    client = ArchiveClient(tmp_path, session=session, pause=0)
    path = client.fetch_file("https://example.org/data.csv", "raw/source/data.csv", expected_checksum="md5:" + hashlib.md5(b"abcdef").hexdigest())
    assert client.fetch_file("https://example.org/data.csv", "raw/source/data.csv") == path
    assert len(session.calls) == 1
    path.write_bytes(b"changed")
    with pytest.raises(DownloadError, match="Cached response changed"):
        client.fetch_file("https://example.org/data.csv", "raw/source/data.csv")


def test_partial_download_resumes_exact_range(tmp_path):
    path = tmp_path / "data.csv"
    path.with_name("data.csv.part").write_bytes(b"abc")
    path.with_name("data.csv.resume.json").write_text(json.dumps({"url": "https://example.org/data.csv", "validator": '"v1"'}))
    session = Session([Response(b"def", 206, {"Content-Range": "bytes 3-5/6", "ETag": '"v1"'})])
    client = ArchiveClient(tmp_path, session=session, pause=0)
    result = client.fetch_file("https://example.org/data.csv", "data.csv")
    assert result.read_bytes() == b"abcdef"
    assert session.calls[0][1]["headers"] == {"Range": "bytes=3-", "If-Range": '"v1"'}


def test_server_ignoring_range_restarts_without_duplicate_prefix(tmp_path):
    (tmp_path / "data.csv.part").write_bytes(b"abc")
    checksum = "sha256:" + hashlib.sha256(b"abcdef").hexdigest()
    client = ArchiveClient(tmp_path, session=Session([Response()]), pause=0)
    assert client.fetch_file("https://example.org/data.csv", "data.csv", expected_checksum=checksum).read_bytes() == b"abcdef"


def test_complete_partial_after_crash_recovers_from_416(tmp_path):
    (tmp_path / "data.csv.part").write_bytes(b"abcdef")
    session = Session([Response(b"", 416), Response()])
    client = ArchiveClient(tmp_path, session=session, pause=0)
    expected = "sha256:" + hashlib.sha256(b"abcdef").hexdigest()
    assert client.fetch_file("https://example.org/data.csv", "data.csv", expected_checksum=expected).read_bytes() == b"abcdef"
    assert len(session.calls) == 2
    assert session.calls[1][1]["headers"] == {}


def test_checksum_mismatch_is_not_published(tmp_path):
    client = ArchiveClient(tmp_path, session=Session([Response()]), pause=0, retries=1)
    with pytest.raises(DownloadError, match="checksum mismatch"):
        client.fetch_file("https://example.org/data.csv", "data.csv", expected_checksum="md5:" + "0" * 32)
    assert not (tmp_path / "data.csv").exists()
    assert not (tmp_path / "data.csv.part").exists()


def test_paths_cannot_escape_collection(tmp_path):
    client = ArchiveClient(tmp_path, session=Session([]), pause=0)
    with pytest.raises(DownloadError, match="escapes"):
        client.fetch_file("https://example.org/data.csv", "../outside.csv")


def test_unverified_existing_file_is_preserved(tmp_path):
    (tmp_path / "data.csv").write_bytes(b"user-owned")
    client = ArchiveClient(tmp_path, session=Session([]), pause=0)
    with pytest.raises(DownloadError, match="will not be overwritten"):
        client.fetch_file("https://example.org/data.csv", "data.csv")
    assert (tmp_path / "data.csv").read_bytes() == b"user-owned"


def test_decoded_gzip_length_is_not_confused_with_wire_length(tmp_path):
    # requests.iter_content returns decoded bytes; Content-Length describes
    # the encoded HTTP representation and cannot verify decoded body size.
    session = Session([Response(b"abcdef", headers={"Content-Length": "3", "Content-Encoding": "gzip"})])
    client = ArchiveClient(tmp_path, session=session, pause=0)
    assert client.get_text("https://example.org/data.csv", "page.html") == "abcdef"
    assert session.calls[0][1]["headers"]["Accept-Encoding"] == "gzip, deflate"
