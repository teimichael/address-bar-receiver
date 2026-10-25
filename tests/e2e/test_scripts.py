"""The shell entrypoints: syntax, the GPU-free dry run of the full pipeline, the model-major loop and
resource guards, resume, and asserted smoke."""

import json
import re
import subprocess
import shlex
import sys
from pathlib import Path

import pytest
import yaml

from originblind.config import load_spec
from originblind.run.plan import ordered

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPO / "scripts").glob("*.sh"))


def _default_configs() -> list[str]:
    """The configs scripts/run_full.sh runs when given no options."""
    m = re.search(r'^CONFIGS="([^"]+)"', (REPO / "scripts/run_full.sh").read_text(), re.M)
    return m.group(1).split()


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_bash_syntax(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_run_full_dry_run_prints_the_model_major_schedule_without_docker(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "OB_PY": sys.executable, "PYTHONPATH": str(REPO / "src"), "OB_RESULTS": str(tmp_path), "HOME": str(Path.home())}
    r = subprocess.run(["bash", "scripts/run_full.sh", "--dry-run"], cwd=REPO,
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout
    total = sum(len(ordered(load_spec(str(REPO / c)))) for c in _default_configs())
    assert "dry run: namespace full" in out  # the default namespace
    assert "model-major schedule" in out and f"{total:,} episodes planned" in out
    assert "e2s_screenshot" in out and "lab" in out
    assert "native browser-use reproduction excluded from public release" in out


@pytest.mark.browser
def test_smoke_script_passes(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "OB_PY": sys.executable, "PYTHONPATH": str(REPO / "src"), "OB_RESULTS": str(tmp_path), "HOME": str(Path.home())}
    r = subprocess.run(["bash", "scripts/15_smoke.sh"], cwd=REPO, capture_output=True, text=True, env=env,
                       timeout=900)
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:])
    assert "0 violations" in r.stdout


def _multi_model_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "model_groups.yaml"
    cfg.write_text(yaml.safe_dump({
        "name": "model_groups", "master_seed_hex": "b0" * 16,
        "roster": str(REPO / "configs/roster.yaml"), "brands": ["northwell"],
        "models": ["qwen3.5-9b", "gemma-4-12b", "qwen3.5-4b"],
        "origin_classes": ["exact"], "flows": ["login"], "observation_arms": ["T1"], "defenses": ["D1"],
        "replicates_per_brand": 1, "driver": "chromium", "environment": "realistic", "step_budget": 12}))
    return cfg


def test_run_full_model_loop_survives_children_that_read_stdin(tmp_path):
    """Regression (end-to-end V6, 2026-10-13): the model-major loop read the schedule from stdin and
    `docker run` (inside `serve start`) consumed the remaining rows, so every experiment run was
    silently skipped. With a stub CLI whose `serve start` reads all of stdin, every scheduled
    (config, model) pair must still run, in model-major order, with one server per group."""
    log = tmp_path / "calls.log"
    fake = _stub_cli(tmp_path, log)
    env = {"PATH": "/usr/bin:/bin", "OB_PY": sys.executable, "PYTHONPATH": str(REPO / "src"), "OB_RESULTS": str(tmp_path / "results"), "HOME": str(Path.home()),
           "OB_CLI": str(fake), "OB_BU_LLM_MODELS": "none"}
    r = subprocess.run(["bash", "scripts/run_full.sh", "--namespace", "loop", "--configs", "configs/e2e/e2e_text.yaml",
                        "--skip-setup", "--skip-fetch", "--skip-smoke", "--skip-audits", "--no-report",
                        "--skip-timestamps", "--allow-dirty"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=600, stdin=subprocess.DEVNULL)
    calls = log.read_text().splitlines()
    starts = [c for c in calls if c.startswith("serve start")]
    runs = [c for c in calls if c.startswith("run ")]
    assert len(starts) == 2, (calls, r.stdout[-2000:], r.stderr[-2000:])
    assert [c.split("--model ")[1].split()[0] for c in runs] == ["qwen3.5-4b", "qwen2.5-7b"], calls
    # model-major: the second server starts only after the first model's runs
    assert calls.index(starts[1]) > calls.index(runs[0])


def _stub_cli(tmp_path: Path, log: Path) -> Path:
    """A stand-in for `originblind`: logs every call; the pure subcommands (schedule, plan, ops budget,
    ops summarize) are the real CLI; the rest is scripted through OB_STUB_* variables (sleep, exit code,
    STOP alarm, the Windows commit headroom the gate and the monitor report), and the long-running ones
    record their PID so the tests can check that nothing outlives a stop. `serve start` writes a gate
    record into --logdir and `ops monitor` one sample into --out, as the real ones do."""
    real = f"{shlex.quote(sys.executable)} -m originblind.cli"
    fake = tmp_path / "fake_ob.sh"
    fake.write_text(f"""#!/usr/bin/env bash
echo "$*" >> "{log}"
arg() {{ local want="$1"; shift; while [ $# -gt 1 ]; do [ "$1" = "$want" ] && {{ echo "$2"; return; }}; shift; done; }}
case "$1 $2" in
  "serve schedule") exec {real} "$@" ;;
  "serve start") cat >/dev/null; echo $$ > "{tmp_path}/serve_start.pid"
                 ld="$(arg --logdir "$@")"; k="$(arg --key "$@")"
                 [ -d "$ld" ] && printf '{{"key": "%s", "ok": true, "reason": "ok", "steps": {{"memory": {{"host_probe": true, "commit_headroom": %s, "commit_needed": 7e10}}}}}}\\n' "$k" "${{OB_STUB_GATE_HEADROOM:-1.1e11}}" > "$ld/gate_${{k}}_$$.json"
                 sleep "${{OB_STUB_SERVE_START_SLEEP:-0}}"; exit "${{OB_STUB_SERVE_START_RC:-0}}" ;;
  "serve stop"|"serve status") exit 0 ;;
  "ops budget"|"ops summarize") exec {real} "$@" ;;
  "ops monitor") echo $$ > "{tmp_path}/monitor.pid"
                 o="$(arg --out "$@")"
                 [ -n "$o" ] && printf '{{"t": "2026-10-16T00:00:00Z", "host_age_s": 1.0, "vm": {{"available": 2e10}}, "host": {{"available": 4e10, "commit_limit": 1.378e11, "committed": %s}}}}\\n' "${{OB_STUB_MONITOR_COMMITTED:-5e10}}" >> "$o"
                 exec sleep 300 ;;
  "ops oom") echo 0; exit 0 ;;
  "ops probe") echo '{{}}'; exit 0 ;;
  "ops lab-env") exit "${{OB_STUB_LABENV_RC:-0}}" ;;
  "ops hostcheck"|"ops sweep"|"ops store") exit 0 ;;
esac
case "$1" in
  plan) exec {real} "$@" ;;
  preflight) echo '{{"blockers": {{}}, "mode": "pilot", "ready": true, "n_blocked": 0, "n_warn": 0}}'; exit 0 ;;
  run) echo $$ > "{tmp_path}/run.pid"
       [ -n "${{OB_STUB_RUN_WRITE_STOP:-}}" ] && echo '{{"reason": "stub alarm"}}' > "$OB_STUB_RUN_WRITE_STOP"
       if [ -n "${{OB_STUB_RUN_TOUCH_OUT:-}}" ]; then o="$(arg --out "$@")"; mkdir -p "$(dirname "$o")"; touch "$o"; fi
       sleep "${{OB_STUB_RUN_SLEEP:-0}}"; exit "${{OB_STUB_RUN_RC:-0}}" ;;
  validate) exit "${{OB_STUB_VALIDATE_RC:-0}}" ;;
esac
exit 0
""")
    fake.chmod(0o755)
    return fake


def _stub_audits(tmp_path: Path, log: Path) -> Path:
    d = tmp_path / "audit"
    for a in ("exposure",):
        (d / a).mkdir(parents=True, exist_ok=True)
        (d / a / "run.sh").write_text(f'#!/usr/bin/env bash\necho "audit {a} $*" >> "{log}"\n')
        (d / a / "run.sh").chmod(0o755)
    return d


def _run_full_stubbed(tmp_path: Path, cfg: Path, ns: str, extra_env: dict | None = None, report: bool = False):
    log = tmp_path / "calls.log"
    env = {"PATH": "/usr/bin:/bin", "OB_PY": sys.executable, "PYTHONPATH": str(REPO / "src"), "OB_RESULTS": str(tmp_path / "results"), "HOME": str(Path.home()),
           "OB_CLI": str(_stub_cli(tmp_path, log)), "OB_AUDIT_DIR": str(_stub_audits(tmp_path, log)),
           "OB_POLL_S": "1", **(extra_env or {})}
    r = subprocess.run(["bash", "scripts/run_full.sh", "--namespace", ns, "--configs", str(cfg),
                        "--skip-setup", "--skip-fetch", "--skip-smoke", *([] if report else ["--no-report"]),
                        "--skip-timestamps", "--allow-dirty"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=600, stdin=subprocess.DEVNULL)
    calls = log.read_text().splitlines() if log.exists() else []
    return r, calls


# ---------------------------------------------------------------- unattended-run safety (README.md)

import fcntl  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402

_FAST = ["--skip-setup", "--skip-fetch", "--skip-smoke", "--no-report", "--skip-timestamps", "--allow-dirty"]


def _one_model_config(tmp_path: Path, model: str = "qwen3.5-4b") -> Path:
    cfg = tmp_path / "one_model.yaml"
    cfg.write_text(yaml.safe_dump({
        "name": "one_model_check", "master_seed_hex": "c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0",
        "roster": str(REPO / "configs/roster.yaml"), "brands": ["northwell"], "models": [model],
        "origin_classes": ["exact"], "flows": ["login"], "observation_arms": ["T1"], "defenses": ["D1"],
        "replicates_per_brand": 1, "driver": "chromium", "environment": "realistic", "step_budget": 12}))
    return cfg


def _start_run_full(tmp_path: Path, cfg: Path, ns: str, extra_env: dict | None = None):
    """run_full in the background (output to a file, never a pipe that could fill up)."""
    log = tmp_path / "calls.log"
    env = {"PATH": "/usr/bin:/bin", "OB_PY": sys.executable, "PYTHONPATH": str(REPO / "src"), "OB_RESULTS": str(tmp_path / "results"), "HOME": str(Path.home()),
           "OB_CLI": str(_stub_cli(tmp_path, log)), "OB_AUDIT_DIR": str(_stub_audits(tmp_path, log)),
           "OB_POLL_S": "1", "OB_BU_LLM_MODELS": "none", **(extra_env or {})}
    out = open(tmp_path / "run_full.out", "w")
    p = subprocess.Popen(["bash", "scripts/run_full.sh", "--namespace", ns, "--configs", str(cfg), *_FAST],
                         cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    return p, log


def _wait_for(path: Path, timeout: float = 120) -> None:
    t0 = time.time()
    while not path.exists():
        assert time.time() - t0 < timeout, f"{path} never appeared"
        time.sleep(0.2)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _lock_free(lock: Path) -> bool:
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def _status(tmp_path: Path, ns: str) -> dict:
    return json.loads((tmp_path / "results" / ns / "RUN_STATUS.json").read_text())


def _steps(tmp_path: Path, ns: str) -> list[dict]:
    lines = []
    for f in sorted((tmp_path / "results" / ns / "logs").glob("run_full_*.steps.jsonl")):
        lines += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return lines


def test_run_full_refuses_while_another_run_holds_the_lock(tmp_path):
    """One run at a time (ob-vllm and port 8000 are singletons); the holder's status is untouched."""
    lock = tmp_path / "results/.ops/run_full.lock"
    lock.parent.mkdir(parents=True)
    status = tmp_path / "results/lockns/RUN_STATUS.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"state": "running", "pid": 1}')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        r, calls = _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "lockns")
    finally:
        os.close(fd)
    assert r.returncode == 1 and "another scripts/run_full.sh is active" in r.stderr, r.stderr[-1000:]
    assert status.read_text() == '{"state": "running", "pid": 1}'
    assert not [c for c in calls if c.startswith(("run ", "serve start", "ops monitor"))], calls


def test_run_full_sigterm_mid_run_stops_promptly_and_cleans_up(tmp_path):
    p, log = _start_run_full(tmp_path, _one_model_config(tmp_path), "term", {"OB_STUB_RUN_SLEEP": "300"})
    _wait_for(tmp_path / "run.pid")
    time.sleep(0.5)
    run_pid = int((tmp_path / "run.pid").read_text())
    st = _status(tmp_path, "term")
    assert st["state"] == "running" and st["current_step"].startswith("run:"), st
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=120)
    assert rc == 143 and time.time() - t0 < 60, (rc, (tmp_path / "run_full.out").read_text()[-2000:])
    st = _status(tmp_path, "term")
    assert st["state"] == "interrupted" and st["exit_code"] == 143
    calls = log.read_text().splitlines()
    first_run = next(i for i, c in enumerate(calls) if c.startswith("run "))
    assert any(c.startswith("serve stop") for c in calls[first_run:]), calls
    assert _gone(run_pid)
    assert _gone(int((tmp_path / "monitor.pid").read_text()))
    assert _lock_free(tmp_path / "results/.ops/run_full.lock")
    steps = _steps(tmp_path, "term")
    assert steps[-1]["step"] == "finish" and steps[-1]["status"] == "interrupted"


def test_run_full_sigterm_during_serve_start_stops_the_server(tmp_path):
    p, log = _start_run_full(tmp_path, _one_model_config(tmp_path), "termstart", {"OB_STUB_SERVE_START_SLEEP": "300"})
    _wait_for(tmp_path / "serve_start.pid")
    time.sleep(0.5)
    pid = int((tmp_path / "serve_start.pid").read_text())
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=120) == 143
    calls = log.read_text().splitlines()
    start = next(i for i, c in enumerate(calls) if c.startswith("serve start"))
    assert any(c.startswith("serve stop") for c in calls[start:]) and not any(c.startswith("run ") for c in calls)
    assert _gone(pid)


def test_run_full_aborts_on_a_stop_alarm_and_refuses_to_resume_until_cleared(tmp_path):
    stop = tmp_path / "results/alarm/STOP"
    p, log = _start_run_full(tmp_path, _one_model_config(tmp_path), "alarm",
                             {"OB_STUB_RUN_SLEEP": "300", "OB_STUB_RUN_WRITE_STOP": str(stop)})
    rc = p.wait(timeout=120)
    assert rc == 6, (tmp_path / "run_full.out").read_text()[-2000:]
    assert _status(tmp_path, "alarm")["state"] == "aborted_resource"
    calls = log.read_text().splitlines()
    first_run = next(i for i, c in enumerate(calls) if c.startswith("run "))
    assert any(c.startswith("serve stop") for c in calls[first_run:])
    assert _gone(int((tmp_path / "run.pid").read_text()))
    r, _ = _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "alarm")
    assert r.returncode == 1 and "STOP alarm is pending" in r.stderr and "stub alarm" in r.stderr


def test_run_full_stops_when_the_pre_serve_gate_refuses(tmp_path):
    r, calls = _run_full_stubbed(tmp_path, _multi_model_config(tmp_path), "gate",
                                 {"OB_STUB_SERVE_START_RC": "6", "OB_BU_LLM_MODELS": "none"})
    assert r.returncode == 6, r.stdout[-2000:]
    assert sum(c.startswith("serve start") for c in calls) == 1 and not any(c.startswith("run ") for c in calls)
    assert _status(tmp_path, "gate")["state"] == "aborted_resource"


def test_run_full_breaker_stops_before_a_third_server_switch(tmp_path):
    """Every server switch is the known host-hang trigger: two failed groups in a row end the loop."""
    r, calls = _run_full_stubbed(tmp_path, _multi_model_config(tmp_path), "brk",
                                 {"OB_STUB_SERVE_START_RC": "1", "OB_BU_LLM_MODELS": "none"})
    assert r.returncode == 1
    assert sum(c.startswith("serve start") for c in calls) == 2, calls
    assert any(s["step"] == "breaker" and s["status"] == "failed" for s in _steps(tmp_path, "brk"))


def test_run_full_kills_a_stalled_pair_and_retries_it_once(tmp_path):
    r, calls = _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "stall",
                                 {"OB_STUB_RUN_SLEEP": "300", "OB_STALL_TIMEOUT_S": "3", "OB_STALL_WARN_S": "1"})
    assert r.returncode == 1, r.stdout[-2000:]
    assert sum(c.startswith("run ") for c in calls) == 2, calls
    steps = _steps(tmp_path, "stall")
    assert any(s["step"].startswith("stall_warning") for s in steps)
    assert [s["rc"] for s in steps if s["step"].startswith("run:")] == [124]
    assert _gone(int((tmp_path / "run.pid").read_text()))


def test_full_namespace_refuses_a_partial_selection(tmp_path):
    r, calls = _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "full")
    assert r.returncode == 1 and "OB_ALLOW_PARTIAL_FULL" in r.stderr
    assert not [c for c in calls if not c.startswith("ops budget")], calls


def test_full_report_is_deferred_until_every_confirmatory_experiment_is_complete(tmp_path):
    """No interim looks: on `full` the report runs only when all seven experiments validate complete."""
    r, calls = _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "full", {"OB_ALLOW_PARTIAL_FULL": "1"},
                                 report=True)
    assert r.returncode == 2, r.stdout[-2000:]
    assert not any(c.startswith("report ") for c in calls)
    steps = _steps(tmp_path, "full")
    assert any(s["step"] == "report" and s["status"] == "deferred" for s in steps)
    validated = {s["step"] for s in steps if s["step"].startswith("validate:")}
    assert len(validated) == len(_default_configs())  # every confirmatory config, not just --configs
    # a pilot namespace keeps reporting incomplete results
    r, calls = _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "pilotx", report=True)
    assert any(c.startswith("report ") for c in calls)


def test_run_full_rejects_an_invalid_worker_count_before_any_step(tmp_path):
    r, calls = _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "badw", {"OB_WORKERS": "six"})
    assert r.returncode == 1 and "positive integer" in r.stderr and calls == []


def test_steps_file_keeps_its_line_format(tmp_path):
    """Flight-recorder steps preserve a stable JSONL format."""
    _run_full_stubbed(tmp_path, _one_model_config(tmp_path), "fmt")
    f = next((tmp_path / "results/fmt/logs").glob("run_full_*.steps.jsonl"))
    lines = f.read_text().splitlines()
    pat = re.compile(r'^\{"step": "[^"]*", "status": "[a-z_]+", "rc": -?\d+, "detail": ".*", '
                     r'"utc": "\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"\}$')
    assert lines and all(pat.match(x) for x in lines), lines
    assert json.loads(lines[-1])["step"] == "finish"


# ---------------------------------------------------------------- rerun readiness (2026-10-15 Docker purge)

def _stub_setup_and_lab(tmp_path: Path, log: Path) -> dict:
    """Stand-ins for scripts/00_setup.sh and scripts/lab_run.sh: they log their arguments (the lab one also
    the OB_LAB_NO_BUILD it sees) and the lab one leaves an episodes.jsonl where lab_run.sh exports."""
    setup = tmp_path / "setup_stub.sh"
    setup.write_text(f'#!/usr/bin/env bash\necho "setup $*" >> "{log}"\n')
    lab = tmp_path / "lab_stub.sh"
    lab.write_text(f'#!/usr/bin/env bash\necho "lab $1 $2 OB_LAB_NO_BUILD=${{OB_LAB_NO_BUILD:-}}" >> "{log}"\n'
                   'e="$(basename "$1" .yaml)"; mkdir -p "$OB_RESULTS_DIR/$2/$e"; touch "$OB_RESULTS_DIR/$2/$e/episodes.jsonl"\n'
                   'exit "${OB_STUB_LAB_RC:-0}"\n')
    for f in (setup, lab):
        f.chmod(0o755)
    return {"OB_SETUP": str(setup), "OB_LAB_RUN": str(lab)}


def test_run_full_records_the_lab_environment_then_freezes_the_lab_image(tmp_path):
    """One lab environment per run: right after setup, `ops lab-env` records (first attempt) or compares the
    lab image's packages, and every lab pair then runs with OB_LAB_NO_BUILD=1, so a mid-run source edit
    fails the lab instead of rebuilding its image. A changed environment stops the run before any server."""
    log = tmp_path / "calls.log"
    env = {"PATH": "/usr/bin:/bin", "OB_PY": sys.executable, "PYTHONPATH": str(REPO / "src"), "OB_RESULTS": str(tmp_path / "results"), "HOME": str(Path.home()),
           "OB_CLI": str(_stub_cli(tmp_path, log)), "OB_POLL_S": "1", "OB_BU_LLM_MODELS": "none",
           **_stub_setup_and_lab(tmp_path, log)}
    args = ["bash", "scripts/run_full.sh", "--namespace", "labenv", "--configs", "configs/ops/ready_check_lab.yaml",
            "--skip-fetch", "--skip-smoke", "--skip-audits", "--no-report", "--skip-timestamps"]
    r = subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)
    calls = log.read_text().splitlines()
    i_setup = next(i for i, c in enumerate(calls) if c.startswith("setup "))
    i_env = next(i for i, c in enumerate(calls) if c.startswith("ops lab-env"))
    labs = [c for c in calls if c.startswith("lab ")]
    assert labs and i_setup < i_env < calls.index(labs[0]), (calls, r.stdout[-2000:])
    assert f"--record {tmp_path}/results/labenv/lab_env.json" in calls[i_env]
    assert len(labs) == 2 and all(c.endswith("OB_LAB_NO_BUILD=1") for c in labs), calls
    assert r.returncode == 0, r.stdout[-2000:]
    log.write_text("")
    r = subprocess.run(args, cwd=REPO, env={**env, "OB_STUB_LABENV_RC": "1"}, capture_output=True, text=True,
                       timeout=600, stdin=subprocess.DEVNULL)
    calls = log.read_text().splitlines()
    assert r.returncode == 1 and not any(c.startswith("serve start") for c in calls), calls
    assert any(s["step"] == "lab_env" and s["status"] == "failed" for s in _steps(tmp_path, "labenv"))


@pytest.mark.parametrize("committed,passes", [("5e10", True), ("1.2e11", False)])
def test_ready_check_passes_only_with_commit_headroom_to_spare(tmp_path, committed, passes):
    """scripts/06_ready_check.sh: preflight --full on the confirmatory configs, the 14B -> 35B switch through
    run_full itself (setup, lab env, weights, gate), then the flight-record verdict from the real
    `ops summarize`: 87.8 GB of Windows commit headroom passes, 17.8 GB fails."""
    log = tmp_path / "calls.log"
    env = {"PATH": "/usr/bin:/bin", "OB_PY": sys.executable, "PYTHONPATH": str(REPO / "src"), "OB_RESULTS": str(tmp_path / "results"), "HOME": str(Path.home()),
           "OB_CLI": str(_stub_cli(tmp_path, log)), "OB_POLL_S": "1", "OB_STUB_RUN_TOUCH_OUT": "1",
           "OB_STUB_MONITOR_COMMITTED": committed, **_stub_setup_and_lab(tmp_path, log)}
    r = subprocess.run(["bash", "scripts/06_ready_check.sh"], cwd=REPO, env=env, capture_output=True, text=True,
                       timeout=900, stdin=subprocess.DEVNULL)
    calls = log.read_text().splitlines()
    assert calls[0].startswith("preflight ") and "--full --namespace full" in calls[0], calls[:3]
    assert all(c in calls[0] for c in _default_configs())
    assert [c.split("--key ")[1].split()[0] for c in calls if c.startswith("serve start")] == \
        ["ministral-3-14b", "qwen3.5-35b-a3b"], calls
    assert any(c.startswith("setup ") for c in calls) and any(c.startswith("models fetch") for c in calls)
    assert ("READY CHECK: PASS" in r.stdout) is passes and r.returncode == (0 if passes else 1), r.stdout[-3000:]
    if not passes:
        assert "commit headroom fell to 17.8 GB" in r.stdout and "READY CHECK: FAIL" in r.stdout
