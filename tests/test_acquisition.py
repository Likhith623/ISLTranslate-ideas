import io
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading
from pathlib import Path

import pytest

from isl_slt.data.acquisition import MultipartReader, extract_selected, pose_index
from isl_slt.data.acquisition import plan_selection
from isl_slt.data.archives import ArchivePart, RangeFile, StagedParts


@contextmanager
def range_server(payloads, honor_ranges=True, *, log=None, redirect_to=None, failures=None):
    transferred = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *args):
            pass
        def do_GET(self):
            if log is not None:
                log.append({"peer": self.client_address, "authorization": self.headers.get("Authorization"),
                            "range": self.headers.get("Range")})
            if redirect_to or failures:
                code = 302 if redirect_to else failures.pop(0)
                self.send_response(code)
                if redirect_to:
                    self.send_header("Location", redirect_to + "/0")
                self.send_header("Content-Length", "0")
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            data = payloads[int(self.path.strip("/"))]
            header = self.headers.get("Range")
            if header and honor_ranges:
                start, end = map(int, header.removeprefix("bytes=").split("-"))
                body = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                body = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (ConnectionResetError, BrokenPipeError):
                pass
            transferred.append(len(body))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", transferred
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_remote_ranges_transfer_only_index_and_selected_member(tmp_path, record_factory):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("poses/wanted-1.pose", b"selected" * 200)
        archive.writestr("poses/unused-1.pose", os.urandom(2 * 1024**2))
    data = memory.getvalue()
    payloads = [data[start:end] for start, end in zip([0, 100, 400, 1000000], [100, 400, 1000000, len(data)])]
    with range_server(payloads) as (url, transferred):
        parts = [ArchivePart(f"part-{i}", len(payload), {"size": len(payload), "id": i},
                            lambda i=i, size=len(payload): RangeFile(f"{url}/{i}", size))
                 for i, payload in enumerate(payloads)]
        records, plan = plan_selection(parts, [record_factory("wanted-1")])
        extract_selected(parts, records, tmp_path / "out", inspect_sample=False)
        assert plan["compressed_bytes"] == 1600
        assert plan["uncompressed_bytes"] == 1600
        assert sum(transferred) < plan["compressed_bytes"] + 150000
        assert (tmp_path / "out/wanted-1.pose").read_bytes() == b"selected" * 200


def test_ranges_refuse_full_body_and_staged_fallback_is_bounded(tmp_path):
    payloads = [b"a" * 100, b"b" * 200, b"c" * 300, b"d" * 400]
    with range_server(payloads, honor_ranges=False) as (url, _):
        with pytest.raises(OSError, match="byte ranges"):
            RangeFile(f"{url}/0", 100).read(1)
        parts = [ArchivePart(str(i), len(data), {"size": len(data)}, None) for i, data in enumerate(payloads)]
        with StagedParts(parts, [f"{url}/{i}" for i in range(4)], tmp_path / "staging") as staging:
            for i in [3, 0, 1, 2, 3]:
                assert staging.open(i).read() == payloads[i]
                assert len(staging.cached) <= 2
            assert staging.peak_bytes <= 700
        assert not list((tmp_path / "staging").glob("*.part"))


def test_pretraining_plan_excludes_all_target_sources_and_budgets_groups(tmp_path, record_factory):
    target = [record_factory("target-1", text="  SAME Text  ", split="train")]
    pool = [record_factory("target-2", text="unique"), record_factory("other-1", text="same text"),
            record_factory("allowed-1", text="first unique"), record_factory("allowed-2", text="second unique")]
    archive = tmp_path / "selection.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for row in pool:
            stream.writestr(f"poses/{row.uid}.pose", b"x" * 10)
    rows, report = plan_selection([archive], pool, target_records=target, budget_bytes=20)
    assert {r.uid for r in rows} == {"allowed-1", "allowed-2"}
    assert report["compressed_bytes"] == 20
    assert {r["reason"] for r in report["excluded"]} == {"target_parent", "duplicate_normalized_text"}
    with pytest.raises(ValueError, match="authoritative target"):
        plan_selection([archive], pool, target_records=[], budget_bytes=20)


def test_split_zip_selective_extraction(tmp_path, record_factory):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("poses/wanted-1.pose", b"synthetic-pose")
        archive.writestr("poses/other-1.pose", b"not-requested")
    payload = memory.getvalue()
    parts = [tmp_path / "aa", tmp_path / "ab"]
    parts[0].write_bytes(payload[:17])
    parts[1].write_bytes(payload[17:])
    report = extract_selected(parts, [record_factory("wanted-1")], tmp_path / "output", inspect_sample=False)
    assert len(report["files"]) == 1
    assert (tmp_path / "output/wanted-1.pose").read_bytes() == b"synthetic-pose"
    assert not (tmp_path / "output/other-1.pose").exists()
    with MultipartReader(parts) as joined:
        joined.seek(-10, 2)
        assert joined.read() == payload[-10:]


def test_archive_traversal_rejected():
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("../outside.pose", b"bad")
    memory.seek(0)
    with zipfile.ZipFile(memory) as archive, pytest.raises(ValueError, match="Unsafe"):
        pose_index(archive)


def test_interrupted_extraction_resumes_and_rejects_corruption(tmp_path, record_factory, monkeypatch):
    from isl_slt.data import acquisition
    archive = tmp_path / "local.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("one-1.pose", b"first")
        stream.writestr("two-1.pose", b"second")
    records = [record_factory("one-1"), record_factory("two-1")]
    output = tmp_path / "out"
    original = acquisition.write_json
    def interrupt(path, value, *args, **kwargs):
        original(path, value, *args, **kwargs)
        if Path(path).name == "extraction.journal.json" and len(value["files"]) == 1:
            raise KeyboardInterrupt()
    monkeypatch.setattr(acquisition, "write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        extract_selected([archive], records, output, inspect_sample=False)
    assert not (output / "extraction.json").exists()
    first_time = (output / "one-1.pose").stat().st_mtime_ns
    monkeypatch.setattr(acquisition, "write_json", original)
    report = extract_selected([archive], records, output, inspect_sample=False, resume=True)
    assert report["reused_uids"] == ["one-1"]
    assert (output / "one-1.pose").stat().st_mtime_ns == first_time
    (output / "one-1.pose").write_bytes(b"bad!!")
    with pytest.raises(ValueError, match="corrupt"):
        extract_selected([archive], records, output, inspect_sample=False, resume=True)
    with pytest.raises(ValueError, match="selection differs"):
        extract_selected([archive], records[:1], output, inspect_sample=False, resume=True)


def test_staging_cleans_only_owned_stale_files_after_lock(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    stale = staging / ("a" * 64 + ".part")
    partial = staging / ("." + "b" * 64 + ".part.interrupted")
    keep = staging / "user-notes.txt"
    for path in (stale, partial, keep):
        path.write_bytes(b"existing")
    with StagedParts([], [], staging):
        assert not stale.exists() and not partial.exists()
        assert keep.exists()


def test_pool_reuses_connection_and_strips_cross_origin_token():
    from isl_slt.data.archives import ArchiveHTTP
    cdn_log, hub_log = [], []
    with range_server([b"abc"], log=cdn_log) as (cdn, _):
        with range_server([b""], log=hub_log, redirect_to=cdn) as (hub, _):
            with ArchiveHTTP("synthetic-token") as client:
                with RangeFile(hub + "/0", 3, client=client) as stream:
                    assert stream.read(1) == b"a"
                    assert stream.read(1) == b"b"
    assert len({row["peer"] for row in hub_log}) == 1
    assert len({row["peer"] for row in cdn_log}) == 1
    assert all(row["authorization"] == "Bearer synthetic-token" for row in hub_log)
    assert all(row["authorization"] is None for row in cdn_log)


def test_transient_retry_is_bounded_and_errors_are_sanitized(monkeypatch):
    from isl_slt.data import archives
    monkeypatch.setattr(archives.time, "sleep", lambda delay: None)
    log = []
    with range_server([b"abc"], log=log, failures=[429, 503]) as (url, _):
        with RangeFile(url + "/0", 3) as stream:
            assert stream.read(3) == b"abc"
    assert len(log) == 3
    log = []
    with range_server([b"abc"], log=log, failures=[503] * 4) as (url, _):
        with RangeFile(url + "/0?signature=private", 3) as stream:
            with pytest.raises(OSError, match="HTTP 503") as error:
                stream.read(1)
    assert len(log) == 3
    assert "signature" not in str(error.value)


def test_selected_windows_merge_reads_across_parts_without_unused_body(tmp_path, record_factory):
    memory = io.BytesIO()
    selected = [record_factory(f"wanted-{i}") for i in range(20)]
    with zipfile.ZipFile(memory, "w") as archive:
        for record in selected:
            archive.writestr(record.uid + ".pose", bytes([len(record.uid)]) * 5000)
        archive.writestr("unused.pose", b"x" * 2_000_000)
    payload = memory.getvalue()
    payloads = [payload[:43000], payload[43000:]]
    log = []
    from isl_slt.data.archives import ArchiveHTTP
    with range_server(payloads, log=log) as (url, transferred), ArchiveHTTP() as client:
        parts = [ArchivePart(str(i), len(data), {"size": len(data)},
                 lambda i=i: RangeFile(f"{url}/{i}", len(payloads[i]), client=client))
                 for i, data in enumerate(payloads)]
        result = extract_selected(parts, selected, tmp_path / "out", inspect_sample=False)
        assert len(result["files"]) == 20
        assert len(log) < 15  # Index plus two selected windows, rather than 3+ reads per clip.
        assert sum(transferred) < 250_000
    assert (tmp_path / "out/wanted-19.pose").read_bytes() == bytes([len("wanted-19")]) * 5000


def test_cache_stays_bounded_and_range_truncation_is_rejected(tmp_path):
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("large.pose", b"x" * (13 * 1024**2))
    path = tmp_path / "archive.zip"
    path.write_bytes(memory.getvalue())
    with MultipartReader([path]) as combined, zipfile.ZipFile(combined) as archive:
        combined.select_members(archive, ["large.pose"])
        with archive.open("large.pose") as source:
            while source.read(1024**2):
                assert sum(map(len, combined.cache.values())) <= 8 * 1024**2
        assert len(combined.cache) <= 2
    with range_server([b"short"]) as (url, _):
        with RangeFile(url + "/0", 10) as stream:
            with pytest.raises(OSError):
                stream.read(10)
