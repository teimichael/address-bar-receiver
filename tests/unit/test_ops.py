"""ops/: the resource budget, probe parsers, store integrity (check, newline fix, manual repair, the lab
prefix rule), the monitor's guard and the pre-serve gate. No Docker, GPU or Windows host needed."""

import gzip
import io
import json
import os
import time
import zlib
from pathlib import Path

import pytest

from originblind.ops import budget, gate, probes, store
from originblind.ops.budget import GIB, Budget, load_budget
from originblind.ops.monitor import Guard


# ---------------------------------------------------------------- budget

def test_budget_defaults_and_env_overrides():
    b = load_budget({})
    assert b == Budget() and b.settle_s == 420 and b.drop_caches is True
    b = load_budget({"OB_SETTLE_S": "600", "OB_DROP_CACHES": "0", "OB_PIDS_LIMIT": "4096", "OB_VLLM_MEMORY": "24g"})
    assert (b.settle_s, b.drop_caches, b.pids_limit, b.vllm_memory) == (600.0, False, 4096, "24g")


@pytest.mark.parametrize("env", [{"OB_SETTLE_S": "soon"}, {"OB_PIDS_LIMIT": "-1"}, {"OB_DROP_CACHES": "maybe"},
                                 {"OB_VLLM_MEMORY": "lots"}])
def test_budget_rejects_invalid_values_naming_the_variable(env):
    with pytest.raises(ValueError, match=next(iter(env))):
        load_budget(env)


def test_container_memory_is_a_backstop_below_the_vm_total():
    total = int(47.04 * GIB)
    assert budget.container_memory("", total, 6) == "41g"
    assert budget.container_memory("24g", total, 6) == "24g"
    assert budget.container_memory("none", total, 6) is None
    assert budget.container_memory("", None, 6) is None
    assert budget.container_memory("", 8 * GIB, 6) == "4g"  # never below 4 GiB
    assert budget.parse_size("48GB") == 48 * GIB and budget.parse_size("512MB") == 512 * 1024 ** 2


# ---------------------------------------------------------------- probe parsers

def test_parsers_on_real_formats():
    m = probes.parse_meminfo("MemTotal:       49328256 kB\nMemAvailable:   45522376 kB\nHugePages_Total:       0\n")
    assert m["MemTotal"] == 49328256 * 1024 and m["HugePages_Total"] == 0
    psi = probes.parse_psi("some avg10=1.50 avg60=0.00 avg300=0.00 total=12\nfull avg10=0.25 avg60=0.00 avg300=0.00 total=3\n")
    assert psi["some"]["avg10"] == 1.5 and psi["full"]["total"] == 3
    assert probes.parse_vmstat("nr_free_pages 1\noom_kill 7\n") == 7
    g = probes.parse_gpu_csv("NVIDIA GeForce RTX 5090, 617.14, 1219, 30969, 32607, Yes, Disabled, 47, 0, 14.62")
    assert g["used_mib"] == 1219 and g["free_mib"] == 30969 and g["display_attached"] is True
    assert g["display_active"] is False and g["driver"] == "617.14"
    assert probes.parse_gpu_csv("NVIDIA GeForce RTX 5090, [N/A], [N/A]") is None
    w = probes.parse_wslconfig("[wsl2]\nnetworkingMode=mirrored\nmemory=48GB\nswap=16GB\n[experimental]\nx=1\n")
    assert w["memory"] == 48 * GIB and w["swap"] == 16 * GIB and "x" not in w["raw"]
    assert probes.parse_wslconfig({"value": "[wsl2]\nmemory=32GB\n"})["memory"] == 32 * GIB  # PowerShell note props


def test_host_json_and_counter_lines():
    h = probes.parse_host_json(json.dumps({
        "total_phys": 68133490688, "available_mb": 45000.0, "committed": 2.2e10, "commit_limit": 7.3e10,
        "gpu_dedicated": 1.29e9, "gpu_shared": 4.0e7, "build": "26300.9711", "display_version": "26H2",
        "pause_updates_expiry": None, "drives": [{"id": "C:", "free": 65e9, "size": 2e12}],
        "procs": [{"name": "vmmemWSL", "ws": 5.7e9}], "wslconfig": "[wsl2]\nmemory=48GB\n"}))
    assert h["available"] == int(45000 * 1024 * 1024) and h["drives"]["C:"]["free"] == 65e9
    assert h["procs"] == [("vmmemWSL", 5700000000)] and h["wslconfig"]["memory"] == 48 * GIB
    assert probes.parse_host_json("not json") is None
    c = probes.parse_counter_line('{"t":"2026-10-15T07:46:53Z","available_mb":45874.0,"gpu_shared":40599552.0,'
                                  '"drives_free_mb":{"C:":62428.0}}')
    assert c["available"] == int(45874 * 1024 * 1024) and c["drives_free"]["C:"] == int(62428 * 1024 * 1024)


def test_powershell_scripts_are_encoded_not_quoted(monkeypatch):
    monkeypatch.setenv("OB_POWERSHELL", "/x/powershell.exe")
    argv = probes.powershell_argv(probes.HOST_PROBE_PS)
    assert argv[0] == "/x/powershell.exe" and argv[-2] == "-EncodedCommand"
    import base64

    assert base64.b64decode(argv[-1]).decode("utf-16-le") == probes.HOST_PROBE_PS
    assert "-MaxSamples 15" in probes.HOST_COUNTERS_PS  # bounded: relaunched, never an endless process


# ---------------------------------------------------------------- store integrity

def _rec(i: int, status: str = "complete") -> bytes:
    return (json.dumps({"episode_id": f"e{i}", "attempt": 0, "status": status}) + "\n").encode()


def _member(i: int) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write((json.dumps({"episode_id": f"e{i}", "attempt": 0, "steps": []}) + "\n").encode())
    return buf.getvalue()


def _store(d: Path, episodes: bytes, members: bytes) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "episodes.jsonl").write_bytes(episodes)
    (d / "transcripts.jsonl.gz").write_bytes(members)
    (d / "manifest.json").write_text("{}")
    return d


def test_an_intact_store_is_ok_and_the_newline_fix_only_touches_torn_tails(tmp_path):
    d = _store(tmp_path / "x", _rec(1) + _rec(2), _member(1) + _member(2))
    assert store.check_store(d)["status"] == "ok"
    assert store.fix_tails(d) == [] and (d / "episodes.jsonl").read_bytes() == _rec(1) + _rec(2)
    (d / "episodes.jsonl").write_bytes(_rec(1) + _rec(2)[:20])            # torn by a crash...
    assert store.check_store(d)["status"] == "repair"                     # ...e2's transcript is now an orphan
    (d / "transcripts.jsonl.gz").write_bytes(_member(1))                 # (its transcript never landed)
    assert store.check_store(d)["status"] == "newline"
    assert store.fix_tails(d) == ["episodes.jsonl"]
    assert (d / "episodes.jsonl").read_bytes() == _rec(1) + _rec(2)[:20] + b"\n"
    assert store.fix_tails(d) == []                                       # idempotent


def test_without_the_newline_fix_the_next_record_would_be_lost(tmp_path):
    """The bug the fix closes: the writers append without checking the tail (orchestrator.py:426)."""
    from originblind.run.orchestrator import read_records

    p = tmp_path / "episodes.jsonl"
    p.write_bytes(_rec(1) + b"\x00" * 40)                                # NUL tail of a host hang
    with p.open("a") as fh:
        fh.write(_rec(2).decode())
    recs, corrupt = read_records(p)
    assert [r["episode_id"] for r in recs] == ["e1"] and corrupt == 1        # e2 glued onto the NULs: lost
    p.write_bytes(_rec(1) + b"\x00" * 40)
    store.newline_terminate(p)
    with p.open("a") as fh:
        fh.write(_rec(2).decode())
    recs, corrupt = read_records(p)
    assert [r["episode_id"] for r in recs] == ["e1", "e2"] and corrupt == 1  # torn bytes stay visible


@pytest.mark.parametrize("bad,kind", [(b"\xff\xfe garbage\n", "non_utf8"), (b"123\n", "non_object")])
def test_lines_that_crash_a_reader_need_a_repair(tmp_path, bad, kind):
    d = _store(tmp_path / "x", _rec(1) + bad + _rec(2), _member(1) + _member(2))
    chk = store.check_store(d)
    assert chk["status"] == "repair" and chk["files"]["episodes.jsonl"]["dangerous"][0]["kind"] == kind
    old = time.time() - 120
    for f in d.iterdir():
        os.utime(f, (old, old))
    rep = store.repair_store(d)
    assert (d / "episodes.jsonl").read_bytes() == _rec(1) + _rec(2)
    q = Path(rep["files"]["episodes.jsonl"]["quarantine"]["file"])
    assert q.read_bytes() == bad                                          # nothing deleted
    assert Path(rep["files"]["episodes.jsonl"]["backup"]).read_bytes() == _rec(1) + bad + _rec(2)
    assert rep["after"]["status"] == "ok"


def test_a_torn_gzip_member_and_an_orphan_transcript_are_quarantined_byte_for_byte(tmp_path):
    torn = _member(3)[:15]
    d = _store(tmp_path / "x", _rec(1) + _rec(2), _member(1) + _member(9) + _member(2) + torn)  # e9: no record
    chk = store.check_store(d)["files"]["transcripts.jsonl.gz"]
    assert chk["status"] == "repair" and chk["members"] == 3 and len(chk["corrupt_regions"]) == 1
    assert [o["key"] for o in chk["orphans"]] == [["e9", 0]]
    old = time.time() - 120
    for f in d.iterdir():
        os.utime(f, (old, old))
    rep = store.repair_store(d)
    assert (d / "transcripts.jsonl.gz").read_bytes() == _member(1) + _member(2)
    q = rep["files"]["transcripts.jsonl.gz"]["quarantine"]
    assert {s["kind"] for s in q["segments"]} == {"orphan", "corrupt"}
    assert Path(q["file"]).read_bytes() == _member(9) + torn
    with gzip.open(d / "transcripts.jsonl.gz", "rt") as fh:                # readable to the end again
        assert len(fh.read().splitlines()) == 2


def test_a_mid_file_corrupt_member_is_resynced(tmp_path):
    data = _member(1) + b"\x1f\x8b\x08garbage" + _member(2)
    sc = store.scan_gzip(data)
    assert len(sc["members"]) == 2 and len(sc["bad"]) == 1
    with pytest.raises((EOFError, OSError, zlib.error)):                   # what report would hit today
        with gzip.open(io.BytesIO(data), "rt") as fh:
            fh.read()


def test_repair_refuses_a_store_that_is_being_written(tmp_path):
    d = _store(tmp_path / "x", _rec(1) + b"123\n", _member(1))
    with pytest.raises(RuntimeError, match="changed less than"):
        store.repair_store(d)


def test_lab_prefix_rule(tmp_path):
    h = _store(tmp_path / "host", _rec(1) + _rec(2), _member(1) + _member(2))
    (h / "manifests").mkdir()
    (h / "manifests" / "a.json").write_text("{}")
    (h / "lab_runs").mkdir()
    (h / "lab_runs" / "x.json").write_text("{}")                          # host-only provenance is fine
    c = _store(tmp_path / "vol", _rec(1) + _rec(2) + _rec(3), _member(1) + _member(2) + _member(3))
    (c / "manifests").mkdir()
    (c / "manifests" / "a.json").write_text("{}")
    assert store.prefix_check(h, c)["ok"]
    (h / "episodes.jsonl").write_bytes(_rec(1) + _rec(2) + b"\x00" * 30)   # a torn host tail is ignored
    assert store.prefix_check(h, c)["ok"]
    short = _store(tmp_path / "short", _rec(1), _member(1))               # a volume that lost records
    (short / "manifests").mkdir()
    (short / "manifests" / "a.json").write_text("{}")
    res = store.prefix_check(h, short)
    assert not res["ok"] and any("episodes.jsonl" in p for p in res["problems"])
    diverged = _store(tmp_path / "div", _rec(1) + _rec(7) + _rec(3), _member(1) + _member(2) + _member(3))
    (diverged / "manifests").mkdir()
    (diverged / "manifests" / "a.json").write_text("{}")
    assert not store.prefix_check(h, diverged)["ok"]


# ---------------------------------------------------------------- guard

def _sample(**kw):
    s = {"vm": {"available": 30 * GIB, "swap_used": 0}, "psi": {"memory": {"some": {"avg10": 0}, "full": {"avg10": 0}}},
         "oom_kill": 0, "results_free": 500e9, "gpu": {"temp_c": 50},
         "host": {"available": 30e9, "gpu_shared": 50 * 1024 ** 2, "drives_free": {"C:": 60e9}}, "host_age_s": 1.0}
    for k, v in kw.items():
        s[k] = v
    return s


def test_guard_needs_consecutive_samples_and_fires_at_once_on_an_oom_kill():
    g = Guard(Budget())
    assert g.evaluate(_sample()) == ([], None)
    low = _sample(host={"available": 1e9, "gpu_shared": 0, "drives_free": {}})
    assert g.evaluate(low)[1] is None and g.evaluate(low)[1] is None
    warn, crit = g.evaluate(low)
    assert crit and "Windows available memory" in crit and any("Windows available" in w for w in warn)
    g = Guard(Budget())
    g.evaluate(_sample())
    assert "OOM killer" in g.evaluate(_sample(oom_kill=1))[1]


def test_guard_ignores_stale_host_samples_and_keeps_gpu_shared_a_warning_until_calibrated():
    g = Guard(Budget())
    stale = _sample(host={"available": 1e9, "gpu_shared": 0, "drives_free": {}}, host_age_s=300.0)
    for _ in range(5):
        assert g.evaluate(stale)[1] is None
    spill = _sample(host={"available": 30e9, "gpu_shared": 6 * 1024 ** 3, "drives_free": {}})
    for _ in range(5):
        warn, crit = g.evaluate(spill)
    assert crit is None and any("GPU shared" in w for w in warn)
    g = Guard(Budget(guard_gpu_shared_mib=4096))
    for _ in range(3):
        warn, crit = g.evaluate(spill)
    assert crit and "GPU shared" in crit


# ---------------------------------------------------------------- the pre-serve gate

class _Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _gpu(used, free=None, total=32607.0):
    return {"used_mib": used, "free_mib": total - used - 400 if free is None else free, "total_mib": total}


def _gate(tmp_path, gpu_seq, *, b=None, vm=None, host=None, stop=None):
    c = _Clock()
    seq = list(gpu_seq)
    calls = {"drop": 0}

    def gpu():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    def drop():
        calls["drop"] += 1
        return True, ""

    g = gate.Gate(b or Budget(settle_s=0, gpu_quiesce_stable_s=4, gpu_quiesce_timeout_s=60, gpu_poll_s=2),
                  state_dir=tmp_path, stop_file=stop, gpu=gpu, vm=lambda: vm or {"available": 30 * GIB, "cached": 0},
                  host=lambda: host, drop=drop, sleep=c.sleep, clock=c.now, now=c.now, log=lambda m: None)
    return g, c, calls


def test_gate_waits_for_a_steadily_quiet_gpu(tmp_path):
    busy, quiet = _gpu(29000), _gpu(1200)
    g, c, calls = _gate(tmp_path, [busy, busy, quiet, busy, quiet])
    res = g.run(util=0.9, key="k")
    assert res.ok and res.record["steps"]["quiesce"]["waited_s"] >= 4 and calls["drop"] == 1


def test_gate_refuses_a_gpu_that_never_quiets_or_cannot_be_measured(tmp_path):
    g, _, calls = _gate(tmp_path, [_gpu(29000)])
    res = g.run(util=0.9, key="k")
    assert not res.ok and "did not become quiet" in res.reason and calls["drop"] == 0
    g, _, _ = _gate(tmp_path, [None])
    res = g.run(util=0.9, key="k")
    assert not res.ok and "cannot be measured" in res.reason


def test_gate_requires_what_vllm_will_request(tmp_path):
    """0.92 x 32607 = 29998 MiB: free VRAM below that is not quiet, whatever `used` says."""
    g, _, _ = _gate(tmp_path, [_gpu(1200, free=29900)])
    assert not g.run(util=0.92, key="k").ok
    g, _, _ = _gate(tmp_path, [_gpu(1200, free=30969)])
    res = g.run(util=0.92, key="k")
    assert res.ok and res.record["steps"]["quiesce"]["slack_mib"] == 971


def test_gate_settles_after_the_previous_stop(tmp_path):
    b = Budget(settle_s=420, gpu_quiesce_stable_s=0, gpu_poll_s=2)
    g, c, _ = _gate(tmp_path, [_gpu(1200)], b=b)
    gate.write_stop_state(tmp_path, key="prev", container="ob-vllm", epoch=c.now() - 100)
    res = g.run(util=0.9, key="k")
    assert res.ok and res.record["steps"]["settle"]["waited_s"] == 320.0
    g, c, _ = _gate(tmp_path, [_gpu(1200)], b=b)
    gate.write_stop_state(tmp_path, key="prev", container="ob-vllm", epoch=c.now() - 3600)
    assert g.run(util=0.9, key="k").record["steps"]["settle"]["waited_s"] == 0.0


def test_gate_refuses_without_memory_headroom_and_on_a_stop_file(tmp_path):
    g, _, _ = _gate(tmp_path, [_gpu(1200)], vm={"available": 4 * GIB, "cached": 0})
    res = g.run(util=0.9, key="k")
    assert not res.ok and "memory headroom" in res.reason
    g, _, _ = _gate(tmp_path, [_gpu(1200)], host={"available": 5e9})
    assert not g.run(util=0.9, key="k").ok
    stop = tmp_path / "STOP"
    stop.write_text('{"reason": "host memory"}')
    g, _, calls = _gate(tmp_path, [_gpu(1200)], stop=stop)
    res = g.run(util=0.9, key="k")
    assert not res.ok and "host memory" in res.reason and calls["drop"] == 0


# ---------------------------------------------------------------- Windows commit headroom (measured 2026-10-15)

def test_commit_need_matches_the_measured_4b_start():
    """A fresh 4B start took 46.9 GB of Windows commit (24.9 -> 71.8 GB of a 73.8 GB limit)."""
    need = probes.commit_needed(0.9, 32607 * 1024 ** 2, 9_342_898_613, 8.0)
    assert 46.9e9 <= need <= 49e9
    assert probes.commit_headroom({"commit_limit": 73.77e9, "committed": 24.9e9}) == pytest.approx(48.87e9)
    assert probes.commit_headroom({"committed": 1}) is None
    man = {"files": {"a.safetensors": {"size": 10}, "b.safetensors": {"size": 5}, "tokenizer.json": {"size": 99}}}
    assert probes.weights_bytes(man) == 15 and probes.weights_bytes({}) is None


def test_gate_refuses_when_the_commit_headroom_cannot_hold_the_server(tmp_path):
    host = {"available": 40e9, "commit_limit": 73.77e9, "committed": 30.4e9}       # this host, 2026-10-15
    g, _, _ = _gate(tmp_path, [_gpu(1200)], host=host, b=Budget(settle_s=0, gpu_quiesce_stable_s=0,
                                                                    gpu_quiesce_timeout_s=30))
    res = g.run(util=0.9, key="qwen3.5-9b", weights_bytes=19_329_384_361)
    assert not res.ok and "commit headroom" in res.reason and "pagefile" in res.reason
    big = dict(host, commit_limit=137.8e9)                                         # + a 64 GB pagefile
    g, _, _ = _gate(tmp_path, [_gpu(1200)], host=big, b=Budget(settle_s=0, gpu_quiesce_stable_s=0))
    res = g.run(util=0.9, key="qwen3.5-9b", weights_bytes=19_329_384_361)
    assert res.ok and res.record["steps"]["memory"]["commit_needed"] > 57e9


def test_guard_watches_the_commit_headroom():
    g = Guard(Budget())
    tight = _sample(host={"available": 30e9, "gpu_shared": 0, "drives_free": {}, "commit_limit": 73.8e9,
                          "committed": 73.0e9})
    assert g.evaluate(tight)[1] is None and g.evaluate(tight)[1] is None
    warn, crit = g.evaluate(tight)
    assert crit and "commit headroom" in crit
    ok = _sample(host={"available": 30e9, "gpu_shared": 0, "drives_free": {}, "commit_limit": 73.8e9,
                       "committed": 66.0e9})
    warn, crit = Guard(Budget()).evaluate(ok)
    assert crit is None and any("commit headroom" in w for w in warn)              # 7.8 GB: a warning only


def test_gate_requires_the_windows_probe_when_asked(tmp_path):
    """Without the probe the gate cannot see the commit headroom: skipped and noted by default, refused
    under require_host_probe (run_full.sh sets OB_REQUIRE_HOST_PROBE=1 for `full`)."""
    b = Budget(settle_s=0, gpu_quiesce_stable_s=0, gpu_quiesce_timeout_s=30)
    g, _, _ = _gate(tmp_path, [_gpu(1200)], host=None, b=b)
    res = g.run(util=0.9, key="k")
    assert res.ok and res.record["steps"]["memory"]["host_probe"] is False
    strict = Budget(settle_s=0, gpu_quiesce_stable_s=0, gpu_quiesce_timeout_s=30, require_host_probe=True)
    g, _, _ = _gate(tmp_path, [_gpu(1200)], host=None, b=strict)
    res = g.run(util=0.9, key="k")
    assert not res.ok and "host probe" in res.reason
    g, _, _ = _gate(tmp_path, [_gpu(1200)], host={"available": 40e9}, b=strict)   # answers, but no commit counters
    assert not g.run(util=0.9, key="k").ok
    big = {"available": 40e9, "commit_limit": 137.8e9, "committed": 30.4e9}
    g, _, _ = _gate(tmp_path, [_gpu(1200)], host=big, b=strict)
    res = g.run(util=0.9, key="k", weights_bytes=19_329_384_361)
    assert res.ok and res.record["steps"]["memory"]["host_probe_required"] is True
    assert load_budget({"OB_REQUIRE_HOST_PROBE": "1"}).require_host_probe is True
    assert load_budget({}).require_host_probe is False


# ---------------------------------------------------------------- one lab environment per run (ops/labenv.py)

def _lab_env(pkgs, iid="sha256:img1", label="src1"):
    import hashlib

    pkgs = sorted(pkgs)
    return lambda image: {"image": image, "image_id": iid, "src_sha256": label, "packages": pkgs,
                          "dpkg_sha256": hashlib.sha256(("\n".join(pkgs) + "\n").encode()).hexdigest(),
                          "n_packages": len(pkgs), "utc": "2026-10-16T00:00:00Z"}


def test_lab_env_is_recorded_once_and_must_match_afterwards(tmp_path):
    from originblind.ops import labenv

    rec = tmp_path / "ns" / "lab_env.json"
    base = ["libc6\t2.39-0ubuntu8.6", "xvfb\t2:21.1.12-1ubuntu1.4"]
    rc, res = labenv.check("lab", rec, tmp_path / "logs" / "a.json", describe_fn=_lab_env(base))
    assert rc == 0 and res["status"] == "recorded" and rec.exists() and (tmp_path / "logs" / "a.json").exists()
    rc, res = labenv.check("lab", rec, None, describe_fn=_lab_env(base, iid="sha256:img2", label="src2"))
    assert rc == 0 and res["status"] == "match" and res["image_rebuilt"]            # only the code layers changed
    rc, res = labenv.check("lab", rec, None, describe_fn=_lab_env(["libc6\t2.39-0ubuntu8.6", "xvfb\t2:21.1.12-1ubuntu1.5"]))
    assert rc == 1 and res["status"] == "mismatch" and res["only_now"] == ["xvfb\t2:21.1.12-1ubuntu1.5"]

    def missing(image):
        raise RuntimeError("lab image lab not found")

    rc, res = labenv.check("lab", rec, None, describe_fn=missing)
    assert rc == 1 and "not found" in res["error"]
    assert json.loads(rec.read_text())["image_id"] == "sha256:img1"                 # the first record stays


def test_lab_env_reads_the_packages_offline_without_pulling(monkeypatch):
    from subprocess import CompletedProcess

    from originblind.ops import labenv

    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            return CompletedProcess(argv, 0, "sha256:abc src123\n", "")
        return CompletedProcess(argv, 0, "xvfb\t1\nlibc6\t2\n", "")

    monkeypatch.setattr(labenv.subprocess, "run", fake_run)
    monkeypatch.setattr(labenv.probes, "docker_bin", lambda: "docker")
    d = labenv.describe("originblind-lab:latest")
    assert d["image_id"] == "sha256:abc" and d["src_sha256"] == "src123" and d["packages"] == ["libc6\t2", "xvfb\t1"]
    run = calls[1]
    assert run[:2] == ["docker", "run"] and run[run.index("--pull") + 1] == "never"
    assert run[run.index("--network") + 1] == "none" and run[run.index("--entrypoint") + 1] == "dpkg-query"
    assert run[-2:] == ["sha256:abc", "-W"]                                          # the inspected id, not the tag


# ---------------------------------------------------------------- readiness verdict (ops/summary.py)

def _ready_run(d: Path, *, head_gb=60.0, gate_ok=True, host_probe=True, critical=False, host=True) -> Path:
    (d / "logs").mkdir(parents=True)
    (d / "RUN_STATUS.json").write_text(json.dumps({
        "state": "finished", "exit_code": 0, "overrides": ["partial_selection"],
        "steps": [{"step": "serve:ministral-3-14b", "status": "ok"}, {"step": "report", "status": "skipped"},
                  {"step": "finish", "status": "finished"}]}))
    for key in ("ministral-3-14b", "qwen3.5-35b-a3b"):
        (d / "logs" / f"gate_{key}_20261016T000000Z.json").write_text(json.dumps({
            "key": key, "ok": gate_ok, "reason": "ok" if gate_ok else "the GPU did not become quiet",
            "steps": {"settle": {"waited_s": 420.0}, "quiesce": {"waited_s": 20.0},
                      "memory": {"host_probe": host_probe, "commit_headroom": 110e9, "commit_needed": 70e9}}}))
    rows = []
    for i in range(5):
        s = {"t": f"2026-10-16T00:00:0{i}Z", "vm": {"available": 20 * GIB, "swap_used": 0},
             "psi": {"memory": {"some": {"avg10": 0.0}, "full": {"avg10": 0.0}}}, "host_age_s": 1.0,
             "host": ({"available": 40e9, "commit_limit": 137.8e9, "gpu_shared": 1e8,
                       "committed": 137.8e9 - (head_gb if i == 3 else 90.0) * 1e9} if host else None)}
        if critical and i == 4:
            s["critical"] = "Windows commit headroom 1.0 GB"
        rows.append(json.dumps(s))
    # a stale Windows sample (the relay lagged) is ignored even when it is low
    rows.append(json.dumps({"t": "late", "host_age_s": 120.0, "host": {"commit_limit": 10e9, "committed": 9.9e9}}))
    (d / "logs" / "monitor_20261016T000000Z.jsonl").write_text("\n".join(rows) + "\n")
    return d


def test_summarize_passes_a_clean_switch_and_names_each_failure(tmp_path):
    from originblind.ops.summary import summarize

    ok = summarize(_ready_run(tmp_path / "ok"), min_commit_headroom_gb=40)
    assert ok["ok"], ok["problems"]
    assert ok["min_commit_headroom_gb"] == 60.0 and len(ok["gates"]) == 2 and ok["windows_samples"] == 5
    for name, kw, why in (("low", dict(head_gb=30.0), "commit headroom fell to 30.0 GB"),
                          ("refused", dict(gate_ok=False), "refused"),
                          ("noprobe", dict(host_probe=False), "without the Windows probe"),
                          ("crit", dict(critical=True), "critical"),
                          ("nohost", dict(host=False), "no fresh Windows sample")):
        res = summarize(_ready_run(tmp_path / name, **kw), min_commit_headroom_gb=40)
        assert not res["ok"] and any(why in p for p in res["problems"]), (name, res["problems"])
    assert not summarize(tmp_path / "missing")["ok"]
