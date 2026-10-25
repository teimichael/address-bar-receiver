"""The weights fetcher downloads files concurrently, one download per blob, and still verifies and
links every file (serving/models.py)."""
import hashlib
import os
import threading

import pytest

from originblind.serving import models as M


def _rf(path: str, data: bytes) -> M.RemoteFile:
    return M.RemoteFile(path=path, size=len(data), git_oid="", sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    data = {"a.safetensors": b"A" * 300, "b.safetensors": b"B" * 100, "copy_of_b.bin": b"B" * 100,
            "config.json": b"{}"}
    files = [_rf(p, d) for p, d in data.items()]
    monkeypatch.setattr(M, "list_files", lambda repo, rev: files)
    calls, lock, active = [], threading.Lock(), {"now": 0, "max": 0}

    def fake_download(repo, rev, rf, dest, log=print, **kw):
        with lock:
            calls.append(rf.path)
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        if rf.path == "config.json" and os.environ.get("OB_TEST_FAIL_CONFIG"):
            with lock:
                active["now"] -= 1
            raise RuntimeError("hash mismatch")
        threading.Event().wait(0.05)
        dest.write_bytes(data[rf.path])
        with lock:
            active["now"] -= 1

    monkeypatch.setattr(M, "_download", fake_download)
    return {"files": files, "data": data, "calls": calls, "active": active}


def test_fetch_downloads_each_blob_once_largest_first_and_links_every_path(fake_hub):
    out = M.fetch("org/repo", "rev1", parallel=1, log=lambda *_: None)
    # b.safetensors and copy_of_b.bin share a blob: one download; largest first
    assert fake_hub["calls"] == ["a.safetensors", "b.safetensors", "config.json"]
    snap = M.repo_dir("org/repo") / "snapshots" / "rev1"
    for p, d in fake_hub["data"].items():
        assert (snap / p).is_symlink() and (snap / p).read_bytes() == d
    assert set(out["files"]) == set(fake_hub["data"])
    # a second fetch finds every blob verified and downloads nothing
    fake_hub["calls"].clear()
    M.fetch("org/repo", "rev1", parallel=4, log=lambda *_: None)
    assert fake_hub["calls"] == []


def test_fetch_runs_downloads_concurrently(fake_hub):
    M.fetch("org/repo", "rev1", parallel=3, log=lambda *_: None)
    assert fake_hub["active"]["max"] >= 2


def test_fetch_failure_lets_the_other_downloads_finish_then_raises(fake_hub, monkeypatch):
    monkeypatch.setenv("OB_TEST_FAIL_CONFIG", "1")
    with pytest.raises(RuntimeError, match="config.json"):
        M.fetch("org/repo", "rev1", parallel=2, log=lambda *_: None)
    blobs = M.repo_dir("org/repo") / "blobs"
    assert (blobs / fake_hub["files"][0].blob_id).read_bytes() == fake_hub["data"]["a.safetensors"]
    assert (blobs / fake_hub["files"][1].blob_id).exists()


class _FakeStream:
    """httpx.stream stand-in: each call serves the next scripted attempt from the requested offset."""

    def __init__(self, data: bytes, script):
        self.data, self.script, self.ranges = data, list(script), []

    def __call__(self, method, url, headers=None, **kw):
        start = int(headers["Range"][6:-1]) if headers and "Range" in headers else 0
        self.ranges.append(start)
        n_bytes, exc = self.script.pop(0) if self.script else (len(self.data), None)
        outer = self

        class Ctx:
            def __enter__(self):
                if exc is not None and n_bytes == 0:
                    raise exc
                r = type("R", (), {})()
                r.status_code, r.request = (206 if start else 200), None

                def iter_bytes(_n):
                    yield outer.data[start:start + n_bytes]
                    if exc is not None:
                        raise exc
                r.iter_bytes = iter_bytes
                return r

            def __exit__(self, *a):
                return False

        return Ctx()


def test_download_keeps_going_while_attempts_make_progress(tmp_path, monkeypatch):
    """Regression (2026-10-14): intermittent DNS and dropped connections exhausted a fixed attempt
    budget although every attempt added bytes. Progress now resets the budget."""
    import httpx

    data = bytes(range(100))
    rf = _rf("w.safetensors", data)
    fake = _FakeStream(data, [(10, httpx.ReadTimeout("slow"))] * 9)  # 9 partial attempts, then the rest
    monkeypatch.setattr(M.httpx, "stream", fake)
    sleeps = []
    M._download("org/repo", "rev", rf, tmp_path / "blob", retries=3, log=lambda *_: None, sleep=sleeps.append)
    assert (tmp_path / "blob").read_bytes() == data
    assert fake.ranges == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # each attempt resumes where the last stopped
    assert len(sleeps) == 9


def test_download_gives_up_after_consecutive_attempts_without_progress(tmp_path, monkeypatch):
    import httpx

    data = bytes(range(100))
    rf = _rf("w.safetensors", data)
    fake = _FakeStream(data, [(10, httpx.ReadTimeout("t"))] + [(0, httpx.ConnectError("dns"))] * 10)
    monkeypatch.setattr(M.httpx, "stream", fake)
    sleeps = []
    with pytest.raises(RuntimeError, match="3 in a row without progress"):
        M._download("org/repo", "rev", rf, tmp_path / "blob", retries=3, log=lambda *_: None, sleep=sleeps.append)
    assert len(fake.ranges) == 4 and fake.ranges[1:] == [10, 10, 10]
    assert (tmp_path / "blob.incomplete").stat().st_size == 10  # kept for the next run to resume


def test_list_files_caches_a_pinned_revision_and_retries_transient_errors(tmp_path, monkeypatch):
    """verify/materialize must not need the network once a pinned revision's listing is known
    (2026-10-14: materialize failed on a DNS error right after a successful verify)."""
    import httpx

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    rev = "a" * 40
    listing = [{"type": "file", "path": "w.safetensors", "oid": "g1", "size": 10,
                "lfs": {"oid": "s" * 64, "size": 10}}, {"type": "directory", "path": "d"}]
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            raise httpx.ConnectError("Temporary failure in name resolution")
        return httpx.Response(200, json=listing, request=httpx.Request("GET", url))

    monkeypatch.setattr(M.httpx, "get", fake_get)
    first = M.list_files("org/repo", rev, sleep=lambda s: None)
    assert [f.path for f in first] == ["w.safetensors"] and first[0].sha256 == "s" * 64 and len(calls) == 2

    def offline(url, **kw):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(M.httpx, "get", offline)
    assert M.list_files("org/repo", rev) == first  # served from the cache
    with pytest.raises(httpx.ConnectError):        # a branch name is mutable: never cached
        M.list_files("org/repo", "main", attempts=2, sleep=lambda s: None)


# ---------------------------------------------------------------- verify-volume: the copy vLLM actually reads

def _volume(monkeypatch, *, sha_out=None, size_out=None, timeout=False):
    from subprocess import CompletedProcess

    sha = hashlib.sha256(b"W").hexdigest()
    man = {"repo": "r/x", "revision": "abc", "files": {
        "model-00001-of-00001.safetensors": {"source": "model.safetensors-00001-of-00001.safetensors",
                                             "size": 1, "sha256": sha},
        "model.safetensors.index.json": {"size": 42, "sha256": "f" * 64},   # LFS source, rewritten on copy
        "config.json": {"size": 2, "sha256": None}}}
    monkeypatch.setattr(M, "volume_manifest", lambda key, volume=M.VOLUME: man)
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1] == "rm":
            return CompletedProcess(argv, 0, "", "")
        if timeout:
            raise M.subprocess.TimeoutExpired(argv, kw.get("timeout"))
        script, names = argv[argv.index("-c") + 1], argv[argv.index("-c") + 3:]
        if "sha256sum" in script:
            return CompletedProcess(argv, 0, sha_out or "".join(f"{sha}  {n}\n" for n in names), "")
        sizes = {"model.safetensors.index.json": 42, "config.json": 2}
        return CompletedProcess(argv, 0, size_out or "".join(f"{sizes[n]} {n}\n" for n in names), "")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    return calls, sha


def test_verify_volume_hashes_the_weights_and_size_checks_the_rewritten_index(monkeypatch):
    calls, sha = _volume(monkeypatch)
    res = M.verify_volume("x")
    assert res["ok"] and res["sha256_checked"] == 1 and res["size_checked"] == 2, res
    assert all(a[a.index("-v") + 1] == f"{M.VOLUME}:/m:ro" for a in calls)          # read-only mount
    assert calls[0][-1] == "model-00001-of-00001.safetensors"                         # names travel as argv


def test_verify_volume_reports_a_corrupt_shard_a_wrong_size_and_a_timeout(monkeypatch):
    _volume(monkeypatch, sha_out="0000  model-00001-of-00001.safetensors\n")
    res = M.verify_volume("x")
    assert not res["ok"] and "model-00001-of-00001.safetensors: sha256" in res["bad"][0]
    _volume(monkeypatch, size_out="41 model.safetensors.index.json\n2 config.json\n")
    assert not M.verify_volume("x")["ok"]
    calls, _ = _volume(monkeypatch, timeout=True)
    res = M.verify_volume("x", timeout=5)
    assert not res["ok"] and any("timed out" in b for b in res["bad"])
    assert any(a[1:3] == ["rm", "-f"] for a in calls)                                  # the reader is stopped
    monkeypatch.setattr(M, "volume_manifest", lambda key, volume=M.VOLUME: None)
    assert not M.verify_volume("x")["ok"]


def test_verify_volume_cli_requires_the_pinned_revision(monkeypatch, capsys):
    monkeypatch.setattr(M, "verify_volume", lambda key: {"key": key, "repo": "r/x", "revision": "old", "ok": True,
                                                         "bad": []})
    assert M.main(["verify-volume", "--repo", "r/x", "--revision", "abc", "--key", "x"]) == 1
    assert "volume holds r/x@old" in capsys.readouterr().out
    monkeypatch.setattr(M, "verify_volume", lambda key: {"key": key, "repo": "r/x", "revision": "abc", "ok": True,
                                                         "bad": []})
    assert M.main(["verify-volume", "--repo", "r/x", "--revision", "abc", "--key", "x"]) == 0
