"""`serve start`/`stop` lifecycle with a fake Docker: the pre-serve gate, container limits, the settle
reference written at stop, OOM reporting and the run lock (README.md)."""

import fcntl
import json
import os
import subprocess
import time

import pytest

from originblind.config import ModelSpec
from originblind.ops.budget import GIB, Budget
from originblind.serving import vllm


def _plan(key="qwen3.5-9b", args=()):
    m = ModelSpec(key=key, hf_repo="Qwen/Qwen3.5-9B", revision="c2022", served_name=key, pipeline="both",
                  max_model_len=16384, vllm_args=list(args))
    return vllm.serve_plan(key, [m])


class FakeDocker:
    """Enough of `docker` for the serving lifecycle: one container, its state and labels."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.c: dict | None = None

    def __call__(self, args, *, timeout=120, merge=False):
        self.calls.append(list(args))
        cmd = args[0]
        ok = lambda out="": subprocess.CompletedProcess(args, 0, out, "")  # noqa: E731
        if cmd == "inspect":
            if self.c is None:
                return subprocess.CompletedProcess(args, 1, "", "Error: No such object")
            st = {"Status": self.c["status"], "Running": self.c["status"] == "running", "ExitCode": 0,
                  "StartedAt": "2026-10-15T00:00:00Z", "FinishedAt": self.c.get("finished", "0001-01-01T00:00:00Z"),
                  "OOMKilled": self.c.get("oom", False)}
            return ok(json.dumps([{"Id": "abc123", "State": st, "Config": {"Labels": self.c["labels"], "Image": "x"},
                                   "Image": "sha256:img"}]))
        if cmd == "run":
            labels = {}
            for i, a in enumerate(args):
                if a == "--label":
                    k, _, v = args[i + 1].partition("=")
                    labels[k] = v
            self.c = {"status": "running", "labels": labels}
            return ok("abc123\n")
        if cmd == "stop" and self.c:
            self.c["status"] = "exited"
            return ok()
        if cmd == "rm":
            self.c = None
            return ok()
        if cmd == "logs":
            return ok("log\n")
        if cmd == "exec":
            return ok("1234567\n42\nmax\n")
        if cmd == "image":
            return ok('["vllm/vllm-openai@sha256:abc"]')
        return ok()  # network inspect / create

    def ran(self):
        return [c for c in self.calls if c[0] == "run"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = FakeDocker()
    clock = {"t": 10_000.0}
    monkeypatch.setattr(vllm, "_docker", d)
    monkeypatch.setattr(vllm, "_weights_manifest", lambda k: {"repo": "Qwen/Qwen3.5-9B", "revision": "c2022", "files": {}})
    monkeypatch.setattr(vllm, "_http_get", lambda url, timeout=5.0: (200, b""))
    monkeypatch.setattr(vllm, "served_payload", lambda port: {"data": [
        {"id": d.c["labels"].get("org.originblind.key") if d.c else "?", "root": "/models/qwen3.5-9b",
         "max_model_len": 16384}]})
    monkeypatch.setattr(vllm, "_gpu_probe", lambda: {"used_mib": 1200.0, "free_mib": 30969.0, "total_mib": 32607.0})
    monkeypatch.setattr(vllm, "_vm_probe", lambda: {"available": 30 * GIB, "cached": 0, "total": 47 * GIB})
    monkeypatch.setattr(vllm, "_host_probe", lambda: None)
    monkeypatch.setattr(vllm, "_drop_caches", lambda: (True, ""))
    monkeypatch.setattr(vllm, "_sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(vllm, "_clock", lambda: clock["t"])
    monkeypatch.setattr(vllm, "_now", lambda: clock["t"])
    monkeypatch.setattr(vllm.time, "time", lambda: clock["t"])
    monkeypatch.setenv("OB_OPS_DIR", str(tmp_path / "ops"))
    monkeypatch.setenv("OB_RUN_LOCK", str(tmp_path / "ops" / "run_full.lock"))
    monkeypatch.delenv("OB_RUN_LOCK_HELD", raising=False)
    return d, clock, tmp_path


def _start(tmp_path, plan=None, budget=None, **kw):
    return vllm.start(plan or _plan(), logdir=tmp_path / "logs", budget=budget or Budget(settle_s=420,
                      gpu_quiesce_stable_s=0), limits={"memory": "41g", "pids": 32768}, **kw)


def test_limits_sit_after_ipc_and_the_label_hash_is_the_reuse_key():
    plan = _plan()
    lim = {"memory": "41g", "pids": 32768}
    want = vllm.spec_sha256(plan, port=8000, gpu_mem=0.9, network=vllm.NETWORK, image=vllm.IMAGE,
                            volume=vllm.VOLUME, limits=lim)
    argv = vllm.docker_run_argv(plan, limits=lim)
    i = argv.index("--ipc=host")
    assert argv[i + 1:i + 7] == ["--memory", "41g", "--memory-swap", "41g", "--pids-limit", "32768"]
    assert f"{vllm.LABEL}.spec_sha256={want}" in argv
    assert want != vllm.spec_sha256(plan, port=8000, gpu_mem=0.9, network=vllm.NETWORK, image=vllm.IMAGE,
                                    volume=vllm.VOLUME)  # an unlimited container is never reused
    assert argv[argv.index("--model") - 1] == vllm.IMAGE
    assert vllm.docker_run_argv(plan) == vllm.docker_run_argv(plan, limits={})  # no limits: argv unchanged


def test_the_last_gpu_memory_utilization_wins():
    plan = _plan("qwen3.5-35b-a3b", ["--quantization", "moe_wna16", "--gpu-memory-utilization", "0.92"])
    assert vllm.gpu_utilization(vllm.docker_run_argv(plan)) == 0.92
    assert vllm.gpu_utilization(vllm.docker_run_argv(_plan())) == 0.9


def test_a_cold_start_passes_the_gate_and_records_it(env):
    d, _, tmp = env
    assert _start(tmp) == vllm.EXIT_OK
    assert len(d.ran()) == 1
    rec = json.loads(next((tmp / "logs").glob("serve_*.json")).read_text())
    assert rec["gate"]["ok"] and rec["limits"] == {"memory": "41g", "pids": 32768}
    assert rec["cgroup_after_ready"]["memory_peak"] == 1234567
    assert json.loads(next((tmp / "logs").glob("gate_*.json")).read_text())["ok"]


def test_a_switch_settles_after_the_previous_server_stopped(env):
    d, clock, tmp = env
    assert _start(tmp) == vllm.EXIT_OK
    t_stop = clock["t"]
    assert vllm.stop(logdir=tmp / "logs") == vllm.EXIT_OK          # writes the settle reference
    assert json.loads((tmp / "ops" / "last_server_stop.json").read_text())["epoch"] == t_stop
    assert _start(tmp, plan=_plan("qwen3.5-9b")) == vllm.EXIT_OK
    gates = sorted((tmp / "logs").glob("gate_*.json"))
    assert json.loads(gates[-1].read_text())["steps"]["settle"]["waited_s"] == 420.0
    assert clock["t"] - t_stop >= 420


def test_the_reuse_path_skips_the_gate(env):
    d, _, tmp = env
    assert _start(tmp) == vllm.EXIT_OK
    n = len(list((tmp / "logs").glob("gate_*.json")))
    assert _start(tmp) == vllm.EXIT_OK                              # same spec, running, identity ok
    assert len(d.ran()) == 1 and len(list((tmp / "logs").glob("gate_*.json"))) == n


def test_a_refusal_starts_nothing(env, monkeypatch):
    d, _, tmp = env
    monkeypatch.setattr(vllm, "_gpu_probe", lambda: {"used_mib": 29000.0, "free_mib": 3000.0, "total_mib": 32607.0})
    rc = _start(tmp, budget=Budget(gpu_quiesce_timeout_s=30, gpu_quiesce_stable_s=0))
    assert rc == vllm.EXIT_GATE and d.ran() == []
    rec = json.loads(next((tmp / "logs").glob("gate_*.json")).read_text())
    assert not rec["ok"] and "did not become quiet" in rec["reason"]


def test_a_stop_file_refuses_before_anything_starts(env):
    d, _, tmp = env
    stop = tmp / "STOP"
    stop.write_text('{"reason": "critical host memory"}')
    assert _start(tmp, stop_file=stop) == vllm.EXIT_GATE and d.ran() == []


def test_an_oom_killed_container_is_reported_at_removal(env, capsys):
    d, _, tmp = env
    d.c = {"status": "exited", "labels": {f"{vllm.LABEL}.key": "qwen3.5-9b"}, "oom": True,
           "finished": "2026-10-15T00:00:00Z"}
    res = vllm._remove(vllm.CONTAINER, tmp / "logs", "x")
    assert res["oom_killed"] is True and "OOM-killed" in capsys.readouterr().err


def test_a_long_dead_leftover_needs_no_settle(env):
    """The crash-time 9B container (finished hours earlier) must not delay the resumed run."""
    d, clock, tmp = env
    finished = time.strftime("%Y-%m-%dT%H:%M:%S.123456789Z", time.gmtime(clock["t"] - 36_000))
    d.c = {"status": "exited", "labels": {f"{vllm.LABEL}.key": "qwen3.5-9b"}, "finished": finished}
    vllm._remove(vllm.CONTAINER, tmp / "logs", "x")
    assert clock["t"] - json.loads((tmp / "ops" / "last_server_stop.json").read_text())["epoch"] >= 35_999


def test_serving_is_refused_while_a_run_holds_the_lock(env, monkeypatch):
    d, _, tmp = env
    lock = tmp / "ops" / "run_full.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert _start(tmp) == vllm.EXIT_FAIL and d.ran() == []
        assert vllm.stop(logdir=tmp / "logs") == vllm.EXIT_FAIL
        monkeypatch.setenv("OB_RUN_LOCK_HELD", "1")                   # the holder's own children may serve
        assert _start(tmp) == vllm.EXIT_OK
    finally:
        os.close(fd)
