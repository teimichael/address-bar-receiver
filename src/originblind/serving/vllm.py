"""vLLM serving for the model-major pipeline: one model resident at a time, provably the right one.

Why this shape (measurement rationale):
  - Identity. An episode is attributable to a model only if the endpoint provably serves the pinned
    weights. `start` therefore checks /v1/models after warm-up: every expected served name must be
    listed with root == /models/<weights key> (the revision-pinned, materialized weights), else exit
    4. A stale server, a wrong mount or a lookalike served name can never answer for another model.
  - Weights come from the `ob-models` named volume, materialized at the pinned revision with a
    `.ob_manifest.json` (serving/models.py). Bind mounts of WSL paths go stale under Docker Desktop
    and HF offline resolution rejects Qwen3.5's shard names; the named volume avoids both.
    `start` refuses weights whose manifest revision is not the pinned one.
  - Shared weights, one server. ModelSpecs with the same (hf_repo, revision) share one materialized
    key and one container: the key of the first such model in roster order (qwen3.5-9b-think serves
    the qwen3.5-9b weights; thinking is a per-request chat-template switch). Every served_name of the
    group is registered, so the same server answers for each member.
  - Forensics. No --rm: a container that dies keeps its logs; `start` fails fast when it exits during
    warm-up (printing the last 80 log lines) and `stop` saves the full log to
    <logdir>/serve_<key>_<utc>.log before removing it. Each successful start writes
    serve_<key>_<utc>.json (docker argv, image digest, weight manifest, /v1/models payload).
  - --limit-mm-per-prompt '{"image":1}' only for vision-capable models (the screenshot arms send one
    image per prompt); text-only models never get it.
  - Host safety (README.md). Two host hangs began during the weight load of a server
    started seconds after the previous one stopped. Before every `docker run`, the pre-serve gate
    (ops/gate.py) waits for a quiet GPU, a settle period after the last stop, drops the VM page
    cache and requires memory headroom; it refuses with exit 6 instead of starting. The container
    gets --memory/--memory-swap/--pids-limit backstops (ops/budget.py: VM MemTotal minus a reserve,
    so they never bind before the VM itself would). Each gate decision is written to
    <logdir>/gate_<key>_<utc>.json. `start`/`stop` refuse while scripts/run_full.sh holds the run
    lock, unless called by that run (OB_RUN_LOCK_HELD=1).

Subcommands (`originblind serve ...`; the docker argv is built by pure, unit-tested functions):
  start --key K [--configs ...] [--port 8000] [--gpu-mem 0.90] [--network ob-net] [--logdir DIR]
        [--timeout 900] [--print-argv] [--stop-file F] [--no-gate] [--no-limits]
  stop [--logdir DIR]            save the container log, then remove the container
  status                         JSON: container state and served models (exit 0 iff up and healthy)
  identity --key K [--configs]   check /v1/models against the expected served names and root
  logs [--tail N]                print the container log
  schedule --configs ... [--models ...] [--namespace NS] [--results DIR] [--format json|tsv|text]
                                 the model-major schedule consumed by scripts/run_full.sh

Exit codes (as in cli.py): 0 ok; 1 serving failed (weights not ready, docker error, container
exited, health timeout, run lock held by another process); 3 endpoint unreachable; 4 served-model
identity mismatch; 5 invalid configuration (unknown model key, conflicting model definitions, invalid
OB_* budget value); 6 refused by the pre-serve gate (GPU not quiet, no memory headroom, STOP file):
nothing was started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from originblind.config import ModelSpec, load_yaml_spec
from originblind.ops import budget as _budget
from originblind.ops import gate as _gate
from originblind.ops import lock as _lock
from originblind.ops import probes as _probes
from originblind.serving import models as _models

IMAGE = "vllm/vllm-openai:v0.31.0"
# the multi-arch index every pilot and the first confirmatory attempt ran (their manifests record it);
# scripts/00_setup.sh pulls this digest and tags it as IMAGE, preflight B08 refuses anything else
IMAGE_DIGEST = "sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
CONTAINER = "ob-vllm"
NETWORK = "ob-net"
VOLUME = _models.VOLUME
MODEL_ROOT = "/models"
CONTAINER_PORT = 8000
LABEL = "org.originblind"
READY_TIMEOUT_S = 900.0
POLL_S = 5.0
LOG_TAIL = 80
STOP_GRACE_S = 60

EXIT_OK, EXIT_FAIL, EXIT_UNREACHABLE, EXIT_MISMATCH, EXIT_CONFIG, EXIT_GATE = 0, 1, 3, 4, 5, 6
# Fallback for the `plan --cost` projection when cli's table is unavailable (eps per GPU-hour).
_FALLBACK_THROUGHPUT = {"inprocess": 3000.0, "chromium": 900.0, "chromium-headful": 300.0}


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- model resolution (pure)

def default_roster() -> Path:
    """configs/roster.yaml of the working tree, else of the repository this package lives in."""
    cwd = Path.cwd() / "configs" / "roster.yaml"
    if cwd.exists():
        return cwd
    return Path(__file__).resolve().parents[3] / "configs" / "roster.yaml"


def model_sources(configs: Sequence[str | Path] | None, roster: str | Path | None = None) -> list[Path]:
    """The roster first (it fixes the canonical weights key of a group), then the given configs."""
    srcs: list[Path] = []
    r = Path(roster) if roster else default_roster()
    if r.exists():
        srcs.append(r)
    for c in configs or []:
        p = Path(c)
        if p not in srcs:
            srcs.append(p)
    return srcs


def load_models(paths: Sequence[str | Path]) -> list[ModelSpec]:
    """Every ModelSpec defined in `paths` (rosters or experiment configs), in first-seen order. A key
    defined twice must name the same weights; a later definition may refine serving fields."""
    out: dict[str, ModelSpec] = {}
    for p in paths:
        for raw in load_yaml_spec(p).get("models", []) or []:
            if not isinstance(raw, dict):
                raise ValueError(f"{p}: model {raw!r} is given by key but the file names no roster")
            m = ModelSpec(**raw)
            prev = out.get(m.key)
            if prev is not None and (prev.hf_repo, prev.revision) != (m.hf_repo, m.revision):
                raise ValueError(f"model {m.key!r} has conflicting weights: {prev.hf_repo}@{prev.revision} "
                                 f"vs {m.hf_repo}@{m.revision} ({p})")
            out[m.key] = m  # dict keeps the first insertion position (roster order)
    return list(out.values())


def servable(m: ModelSpec) -> bool:
    """`none/...` repos are scripted stand-ins (the GPU-free smoke); they are never served."""
    return not m.hf_repo.startswith("none/")


@dataclass(frozen=True)
class ServePlan:
    key: str  # the requested model key
    weights_key: str  # materialized key in the volume: /models/<weights_key>
    hf_repo: str
    revision: str | None
    served_names: tuple[str, ...]  # the requested model's served_name first
    members: tuple[str, ...]  # every model key that shares these weights
    max_model_len: int
    vision: bool
    vllm_args: tuple[str, ...]

    @property
    def model_path(self) -> str:
        return f"{MODEL_ROOT}/{self.weights_key}"


def serve_plan(key: str, models: Sequence[ModelSpec]) -> ServePlan:
    """Resolve what one server must look like to serve `key` and every model sharing its weights."""
    by_key = {m.key: m for m in models}
    if key not in by_key:
        raise KeyError(f"unknown model key {key!r} (known: {', '.join(sorted(by_key))})")
    m = by_key[key]
    if not servable(m):
        raise ValueError(f"model {key!r} ({m.hf_repo}) is a scripted stand-in and cannot be served")
    group = [g for g in models if (g.hf_repo, g.revision) == (m.hf_repo, m.revision)]
    args = {tuple(g.vllm_args) for g in group}
    if len(args) > 1:
        raise ValueError(f"models sharing {m.hf_repo}@{m.revision} disagree on vllm_args: "
                         + "; ".join(f"{g.key}={g.vllm_args}" for g in group))
    names = [m.served_name] + [g.served_name for g in group]
    return ServePlan(key=key, weights_key=group[0].key, hf_repo=m.hf_repo, revision=m.revision,
                     served_names=tuple(dict.fromkeys(names)), members=tuple(g.key for g in group),
                     max_model_len=max(g.max_model_len for g in group), vision=any(g.vision for g in group),
                     vllm_args=tuple(m.vllm_args))


def vllm_argv(plan: ServePlan, *, gpu_mem: float = 0.90) -> list[str]:
    """Arguments after the image name (the vllm serve command line)."""
    a = ["--model", plan.model_path, "--served-model-name", *plan.served_names,
         "--max-model-len", str(plan.max_model_len), "--gpu-memory-utilization", f"{gpu_mem:g}",
         "--enable-prefix-caching"]
    if plan.vision:
        a += ["--limit-mm-per-prompt", json.dumps({"image": 1}, separators=(",", ":"))]
    return a + list(plan.vllm_args)


def spec_sha256(plan: ServePlan, *, port: int, gpu_mem: float, network: str, image: str, volume: str,
                limits: dict | None = None) -> str:
    """Hash of everything that defines the server; a running container is reused only on equality
    (the container limits count: a container started without them is never reused)."""
    payload = {"image": image, "vllm": vllm_argv(plan, gpu_mem=gpu_mem), "port": port, "network": network,
               "volume": volume, "revision": plan.revision}
    if limits:
        payload["limits"] = limits
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def limit_argv(limits: dict | None) -> list[str]:
    """docker flags for the runaway backstops: memory (no swap beyond it) and the task count."""
    out: list[str] = []
    if limits and limits.get("memory"):
        out += ["--memory", str(limits["memory"]), "--memory-swap", str(limits["memory"])]
    if limits and limits.get("pids"):
        out += ["--pids-limit", str(limits["pids"])]
    return out


def default_limits(b: _budget.Budget | None = None, mem_total: int | None = None) -> dict:
    """The budget's backstops for the vLLM container (memory auto = VM MemTotal - reserve)."""
    b = b or _budget.load_budget()
    if mem_total is None:
        mem_total = (_probes.vm_memory() or {}).get("total")
    out: dict = {}
    mem = _budget.container_memory(b.vllm_memory, mem_total, b.container_reserve_gib)
    if mem:
        out["memory"] = mem
    if b.pids_limit:
        out["pids"] = b.pids_limit
    return out


def gpu_utilization(argv: Sequence[str]) -> float:
    """The --gpu-memory-utilization vLLM will apply: the last occurrence wins (the 35B's roster
    vllm_args append 0.92 after the 0.90 default)."""
    val = 0.90
    for i, a in enumerate(argv[:-1]):
        if a == "--gpu-memory-utilization":
            try:
                val = float(argv[i + 1])
            except ValueError:
                pass
    return val


def docker_run_argv(plan: ServePlan, *, port: int = 8000, gpu_mem: float = 0.90, network: str = NETWORK,
                    image: str = IMAGE, container: str = CONTAINER, volume: str = VOLUME,
                    limits: dict | None = None, spec: str | None = None) -> list[str]:
    labels = {
        "key": plan.key, "weights_key": plan.weights_key, "hf_repo": plan.hf_repo,
        "revision": plan.revision or "", "served": ",".join(plan.served_names),
        "spec_sha256": spec or spec_sha256(plan, port=port, gpu_mem=gpu_mem, network=network, image=image,
                                           volume=volume, limits=limits),
    }
    argv = ["docker", "run", "-d", "--gpus", "all", "--ipc=host", *limit_argv(limits), "--name", container,
            "--network", network, "-p", f"{port}:{CONTAINER_PORT}", "-v", f"{volume}:{MODEL_ROOT}:ro",
            # weights are a local path; offline mode forbids any silent Hub download at load time
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"]
    for k, v in labels.items():
        argv += ["--label", f"{LABEL}.{k}={v}"]
    return argv + [image, *vllm_argv(plan, gpu_mem=gpu_mem)]


def check_identity(payload: dict | list, plan: ServePlan) -> dict:
    """Compare a /v1/models payload with the plan: each served name listed, rooted at the weights."""
    data = payload.get("data", []) if isinstance(payload, dict) else list(payload)
    by_id = {d.get("id"): d for d in data if isinstance(d, dict)}
    problems = []
    for name in plan.served_names:
        d = by_id.get(name)
        if d is None:
            problems.append(f"served model {name!r} is not listed (listed: {sorted(map(str, by_id))})")
            continue
        if d.get("root") != plan.model_path:
            problems.append(f"{name!r} is rooted at {d.get('root')!r}, expected {plan.model_path!r}")
        mml = d.get("max_model_len")
        if mml is not None and int(mml) != plan.max_model_len:
            problems.append(f"{name!r} has max_model_len {mml}, expected {plan.max_model_len}")
    return {"ok": not problems, "problems": problems, "expected_root": plan.model_path,
            "expected_names": list(plan.served_names),
            "listed": [{"id": d.get("id"), "root": d.get("root"), "max_model_len": d.get("max_model_len")}
                       for d in by_id.values()]}


def weights_status(manifest: dict | None, plan: ServePlan) -> tuple[bool, str]:
    """Is the volume's .ob_manifest.json the pinned revision of the plan's repo?"""
    where = f"volume {VOLUME}:/{plan.weights_key}"
    if not manifest:
        return False, f"{where} has no .ob_manifest.json (not materialized)"
    if manifest.get("repo") != plan.hf_repo or manifest.get("revision") != plan.revision:
        return False, (f"{where} holds {manifest.get('repo')}@{str(manifest.get('revision'))[:12]}, "
                       f"expected {plan.hf_repo}@{str(plan.revision)[:12]}")
    return True, f"{where} = {plan.hf_repo}@{str(plan.revision)[:12]} ({len(manifest.get('files', {}))} files)"


def fix_hint(plan: ServePlan) -> str:
    return (f"originblind models fetch --configs configs/roster.yaml --only {plan.weights_key} && "
            f"originblind models materialize --configs configs/roster.yaml --only {plan.weights_key}")


# ---------------------------------------------------------------- side effects (patched in tests)

def _docker(args: Sequence[str], *, timeout: float | None = 120, merge: bool = False) -> subprocess.CompletedProcess:
    """Run `docker <args>`; never raises for a non-zero exit (callers inspect returncode)."""
    streams = ({"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT} if merge
               else {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})
    try:
        return subprocess.run(["docker", *args], text=True, timeout=timeout, errors="replace", **streams)
    except FileNotFoundError:
        return subprocess.CompletedProcess(["docker", *args], 127, "", "docker: command not found")
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(["docker", *args], 124, "", f"docker {args[0]} timed out after {e.timeout}s")


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    """(status, body); status 0 when unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception:  # noqa: BLE001 - connection refused/reset, timeout, DNS: all "unreachable"
        return 0, b""


def _weights_manifest(key: str) -> dict | None:
    return _models.volume_manifest(key)


_sleep: Callable[[float], None] = time.sleep
_clock: Callable[[], float] = time.monotonic
_now: Callable[[], float] = time.time
_gpu_probe: Callable[[], dict | None] = _probes.gpu_query
_vm_probe: Callable[[], dict | None] = _probes.vm_memory
_host_probe: Callable[[], dict | None] = _probes.host_probe
_drop_caches: Callable[[], tuple[bool, str]] = _gate.drop_page_cache


# ---------------------------------------------------------------- docker helpers

def container_state(container: str = CONTAINER) -> dict | None:
    """None if no such container; raises RuntimeError if docker itself is unusable."""
    r = _docker(["inspect", "--type", "container", container], timeout=30)
    if r.returncode != 0:
        if r.returncode == 127 or "Cannot connect" in (r.stderr or ""):
            raise RuntimeError(f"docker unavailable: {(r.stderr or '').strip()[:200]}")
        return None
    try:
        info = json.loads(r.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None
    st = info.get("State") or {}
    cfg = info.get("Config") or {}
    labels = cfg.get("Labels") or {}
    return {"id": (info.get("Id") or "")[:12], "status": st.get("Status"), "running": bool(st.get("Running")),
            "exit_code": st.get("ExitCode"), "started_at": st.get("StartedAt"), "finished_at": st.get("FinishedAt"),
            "oom_killed": bool(st.get("OOMKilled")),
            "image": cfg.get("Image"), "image_id": info.get("Image"),
            "labels": {k[len(LABEL) + 1:]: v for k, v in labels.items() if k.startswith(LABEL + ".")}}


def ensure_network(network: str) -> None:
    if _docker(["network", "inspect", network], timeout=30).returncode == 0:
        return
    r = _docker(["network", "create", network], timeout=30)
    if r.returncode != 0 and _docker(["network", "inspect", network], timeout=30).returncode != 0:
        raise RuntimeError(f"cannot create docker network {network}: {(r.stderr or '').strip()[:200]}")


def logs_tail(container: str, n: int = LOG_TAIL) -> str:
    r = _docker(["logs", "--tail", str(n), container], timeout=60, merge=True)
    return r.stdout or ""


def save_logs(container: str, logdir: Path, key: str) -> Path | None:
    """Write the container's full log (timestamped) to <logdir>/serve_<key>_<utc>.log."""
    r = _docker(["logs", "--timestamps", container], timeout=300, merge=True)
    if r.returncode != 0:
        return None
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / f"serve_{key}_{_stamp()}.log"
    path.write_text(r.stdout or "")
    return path


def _endpoint(port: int) -> str:
    return f"http://localhost:{port}"


def served_payload(port: int) -> dict | None:
    code, body = _http_get(f"{_endpoint(port)}/v1/models", timeout=10)
    if code != 200:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def wait_ready(port: int, container: str, timeout: float, poll: float = POLL_S) -> tuple[bool, str]:
    """Poll /health until 200; fail fast if the container stops running."""
    t0 = _clock()
    last_note = t0
    while True:
        st = container_state(container)
        if st is None:
            return False, "the container disappeared during warm-up"
        if not st["running"]:
            return False, f"the container exited during warm-up (status={st['status']}, exit code {st['exit_code']})"
        if _http_get(f"{_endpoint(port)}/health")[0] == 200:
            return True, f"ready after {_clock() - t0:.0f} s"
        if _clock() - t0 > timeout:
            return False, f"not ready after {timeout:.0f} s"
        if _clock() - last_note >= 60:
            _err(f"  waiting for {_endpoint(port)}/health ({_clock() - t0:.0f} s elapsed, container running)")
            last_note = _clock()
        _sleep(poll)


def cgroup_peaks(container: str) -> dict | None:
    """memory.peak / pids.peak of a running container's cgroup (evidence that the limits never
    bound); None when unavailable."""
    r = _docker(["exec", container, "cat", "/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/pids.peak",
                 "/sys/fs/cgroup/memory.max"], timeout=20)
    if r.returncode != 0:
        return None
    vals = (r.stdout or "").split()
    out: dict = {}
    for name, v in zip(("memory_peak", "pids_peak", "memory_max"), vals):
        out[name] = int(v) if v.isdigit() else v
    return out or None


def _finished_epoch(st: dict) -> float | None:
    """Docker's FinishedAt (RFC 3339 with nanoseconds) as epoch seconds; None if unset/unparseable."""
    raw = st.get("finished_at") or ""
    if not raw or raw.startswith("0001-"):
        return None
    try:
        return datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _remove(container: str, logdir: Path, key: str, *, graceful: bool = True) -> dict:
    """Stop (gracefully), save the full log, remove, and record when the GPU was released (the
    settle reference of the next start's gate). Returns what happened."""
    st = container_state(container)
    if st is None:
        return {"removed": False, "reason": f"no container named {container}"}
    key = st["labels"].get("key") or key
    was_running = st["running"]
    peaks = cgroup_peaks(container) if was_running else None
    if st["running"] and graceful:
        _docker(["stop", "-t", str(STOP_GRACE_S), container], timeout=STOP_GRACE_S + 60)
        st = container_state(container) or st
    log = save_logs(container, logdir, key)  # after the stop: the log includes vLLM's shutdown
    r = _docker(["rm", "-f", container], timeout=120)
    # a server that was running released the GPU just now; one that had exited released it when it
    # finished (a crash-time leftover, removed long after, needs no settle); a container that never
    # ran (status "created") never held it
    released = time.time() if was_running else _finished_epoch(st)
    if released is not None:
        _gate.write_stop_state(_lock.ops_dir(), key=key, container=container, epoch=released)
    if st.get("oom_killed"):
        _err(f"{container}: the container was OOM-killed (docker --memory backstop or VM memory); "
             "raise OB_VLLM_MEMORY or the WSL memory only after reading README.md")
    return {"removed": r.returncode == 0, "key": key, "log": str(log) if log else None,
            "status": st.get("status"), "exit_code": st.get("exit_code"), "oom_killed": st.get("oom_killed"),
            "cgroup": peaks, "error": (r.stderr or "").strip()[:200] if r.returncode else None}


# ---------------------------------------------------------------- lifecycle

def start(plan: ServePlan, *, port: int = 8000, gpu_mem: float = 0.90, network: str = NETWORK,
          logdir: Path = Path("results/logs"), timeout: float = READY_TIMEOUT_S, image: str = IMAGE,
          container: str = CONTAINER, keep_on_failure: bool = False, limits: dict | None = None,
          gate: bool = True, stop_file: Path | None = None, budget: _budget.Budget | None = None) -> int:
    """Serve `plan` and verify its identity. Exit code as documented in the module docstring.
    `limits=None` applies the budget's container backstops; pass {} for none."""
    locked = _lock.refuse_if_locked()
    if locked:
        _err(f"{plan.key}: {locked}")
        return EXIT_FAIL
    try:
        b = budget or _budget.load_budget()
    except ValueError as e:
        _err(f"{plan.key}: invalid resource budget: {e}")
        return EXIT_CONFIG
    if limits is None:
        limits = default_limits(b)
    try:
        return _start(plan, port=port, gpu_mem=gpu_mem, network=network, logdir=logdir, timeout=timeout,
                      image=image, container=container, keep_on_failure=keep_on_failure, limits=limits,
                      gate=gate, stop_file=stop_file, budget=b)
    except RuntimeError as e:  # docker became unusable mid-way (daemon restart, WSL hiccup)
        _err(f"{plan.key}: serving failed: {e}")
        return EXIT_FAIL


def _run_gate(plan: ServePlan, argv: list[str], b: _budget.Budget, stop_file: Path | None,
              logdir: Path, weights_bytes: int | None = None) -> _gate.GateResult:
    """The pre-serve gate (ops/gate.py); its decision is written to <logdir>/gate_<key>_<utc>.json."""
    g = _gate.Gate(b, state_dir=_lock.ops_dir(), stop_file=stop_file, gpu=_gpu_probe, vm=_vm_probe,
                   host=_host_probe, drop=_drop_caches, sleep=_sleep, clock=_clock, now=_now, log=_err)
    res = g.run(util=gpu_utilization(argv), key=plan.key, weights_bytes=weights_bytes)
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        path = logdir / f"gate_{plan.key}_{_stamp()}.json"
        path.write_text(json.dumps(res.record, indent=2, sort_keys=True))
        res.record["record"] = str(path)
    except OSError as e:
        _err(f"warning: cannot write the gate record: {e}")
    return res


def _start(plan: ServePlan, *, port: int, gpu_mem: float, network: str, logdir: Path, timeout: float,
           image: str, container: str, keep_on_failure: bool, limits: dict | None = None, gate: bool = False,
           stop_file: Path | None = None, budget: _budget.Budget | None = None) -> int:
    if not plan.revision:
        _err(f"{plan.key}: no pinned revision in its ModelSpec; refusing to serve unpinned weights")
        return EXIT_CONFIG
    manifest = None
    try:
        manifest = _weights_manifest(plan.weights_key)
        ok, detail = weights_status(manifest, plan)
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"cannot read the volume manifest: {e!r}"
    if not ok:
        _err(f"{plan.key}: weights not ready: {detail}.\n  fix: {fix_hint(plan)}")
        return EXIT_FAIL
    _err(f"{plan.key}: {detail}")

    want = spec_sha256(plan, port=port, gpu_mem=gpu_mem, network=network, image=image, volume=VOLUME,
                       limits=limits)
    argv = docker_run_argv(plan, port=port, gpu_mem=gpu_mem, network=network, image=image, container=container,
                           limits=limits, spec=want)
    st = container_state(container)
    if st is not None:
        if st["running"] and st["labels"].get("spec_sha256") == want:
            payload = served_payload(port)
            if payload is not None and check_identity(payload, plan)["ok"]:
                _err(f"{container}: already serving {plan.weights_key} with the identical spec; reusing it")
                rec = _record(plan, argv, st, payload, logdir, ready_s=0.0, reused=True)
                print(json.dumps(rec, indent=2))
                return EXIT_OK
        _err(f"{container}: replacing the existing container ({st['status']}, "
             f"serving {st['labels'].get('weights_key', '?')}); its log is saved first")
        _remove(container, logdir, plan.key)

    gate_rec = None
    if gate:
        res = _run_gate(plan, argv, budget or _budget.load_budget(), stop_file, logdir,
                        weights_bytes=_probes.weights_bytes(manifest))
        gate_rec = res.record
        if not res.ok:
            _err(f"{plan.key}: refused by the pre-serve gate: {res.reason} (record: {gate_rec.get('record')})")
            return EXIT_GATE
    ensure_network(network)
    _err("+ " + shlex.join(argv))
    t0 = _clock()
    r = _docker(argv[1:], timeout=600)
    if r.returncode != 0:
        _err(f"docker run failed (exit {r.returncode}): {(r.stderr or r.stdout or '').strip()[:1000]}")
        if container_state(container) is not None:
            _remove(container, logdir, plan.key, graceful=False)
        return EXIT_FAIL

    ready, why = wait_ready(port, container, timeout)
    if not ready:
        _err(f"{plan.key}: vLLM {why}. Last {LOG_TAIL} log lines:\n{logs_tail(container)}")
        if keep_on_failure:
            _err(f"{container} kept for inspection (--keep-on-failure); remove it with `originblind serve stop`")
        else:
            res = _remove(container, logdir, plan.key, graceful=False)
            _err(f"  full log saved to {res.get('log')}")
        return EXIT_FAIL

    payload = served_payload(port)
    ident = check_identity(payload, plan) if payload is not None else {
        "ok": False, "problems": [f"{_endpoint(port)}/v1/models unreachable after /health succeeded"]}
    if not ident["ok"]:
        _err(f"{plan.key}: served-model identity mismatch: " + "; ".join(ident["problems"]))
        if not keep_on_failure:
            _remove(container, logdir, plan.key, graceful=False)
        return EXIT_MISMATCH
    rec = _record(plan, argv, container_state(container) or {}, payload, logdir,
                  ready_s=round(_clock() - t0, 1), reused=False,
                  extra={"gate": gate_rec, "limits": limits, "cgroup_after_ready": cgroup_peaks(container)})
    _err(f"{plan.key}: serving {', '.join(plan.served_names)} from {plan.model_path} ({why})")
    print(json.dumps(rec, indent=2))
    return EXIT_OK


def _record(plan: ServePlan, argv: list[str], st: dict, payload: dict, logdir: Path, *,
            ready_s: float, reused: bool, extra: dict | None = None) -> dict:
    """Provenance of one serving session: written to <logdir>/serve_<key>_<utc>.json."""
    digest = _docker(["image", "inspect", "--format", "{{json .RepoDigests}}", _image_of(argv)], timeout=30)
    try:
        repo_digests = json.loads(digest.stdout) if digest.returncode == 0 else []
    except json.JSONDecodeError:
        repo_digests = []
    manifest = None
    try:
        manifest = _weights_manifest(plan.weights_key)
    except Exception:  # noqa: BLE001
        pass
    rec = {"utc": _utc(), "reused": reused, "ready_s": ready_s, "plan": asdict(plan),
           "model_path": plan.model_path, "docker_argv": argv, "container": st,
           "image_repo_digests": repo_digests, "weights_manifest": manifest,
           "served_models": (payload or {}).get("data", payload), **(extra or {})}
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        path = logdir / f"serve_{plan.key}_{_stamp()}.json"
        path.write_text(json.dumps(rec, indent=2, sort_keys=True))
        rec["record"] = str(path)
    except OSError as e:
        _err(f"warning: cannot write the serve record: {e}")
    return {k: rec[k] for k in ("utc", "reused", "ready_s", "model_path", "served_models", "record") if k in rec} | {
        "key": plan.key, "weights_key": plan.weights_key, "served_names": list(plan.served_names)}


def _image_of(argv: list[str]) -> str:
    """The image name in a docker_run_argv (the first argument after the docker options)."""
    i = argv.index("--model") - 1
    return argv[i]


def stop(*, container: str = CONTAINER, logdir: Path = Path("results/logs")) -> int:
    locked = _lock.refuse_if_locked()
    if locked:
        _err(locked)
        return EXIT_FAIL
    try:
        res = _remove(container, logdir, "unknown")
    except RuntimeError as e:
        _err(str(e))
        return EXIT_FAIL
    print(json.dumps(res, indent=2))
    if res.get("reason"):
        return EXIT_OK  # nothing to stop
    return EXIT_OK if res.get("removed") else EXIT_FAIL


def status(*, container: str = CONTAINER, port: int = 8000) -> tuple[int, dict]:
    try:
        st = container_state(container)
    except RuntimeError as e:
        return EXIT_FAIL, {"container": container, "error": str(e)}
    out: dict = {"container": container, "port": port, "state": st, "healthy": False, "served_models": None}
    if st and st["running"]:
        out["healthy"] = _http_get(f"{_endpoint(port)}/health")[0] == 200
        payload = served_payload(port)
        out["served_models"] = (payload or {}).get("data") if payload else None
    return (EXIT_OK if out["healthy"] else EXIT_FAIL), out


def identity(plan: ServePlan, *, port: int = 8000) -> tuple[int, dict]:
    payload = served_payload(port)
    if payload is None:
        return EXIT_UNREACHABLE, {"ok": False, "problems": [f"{_endpoint(port)}/v1/models unreachable"]}
    res = check_identity(payload, plan)
    return (EXIT_OK if res["ok"] else EXIT_MISMATCH), res


# ---------------------------------------------------------------- model-major schedule

def _throughput() -> dict[str, float]:
    try:
        from originblind.cli import _DEFAULT_THROUGHPUT

        return dict(_DEFAULT_THROUGHPUT)
    except Exception:  # noqa: BLE001
        return dict(_FALLBACK_THROUGHPUT)


def build_schedule(configs: Sequence[str | Path], *, models: Sequence[str] | None = None, namespace: str = "full",
                   results: str | Path = "results", roster: str | Path | None = None) -> dict:
    """Group every (config, model) pair of the selection by served weights, in roster order.

    Within a group, host-driver pairs come first (config order), then lab (chromium-headful) pairs.
    `complete`/`missing` are read from results/<namespace>/<config name>/episodes.jsonl, so a resumed
    pipeline can skip finished pairs without starting a server. GPU-hours use the same throughput
    table as `originblind plan --cost`.
    """
    from originblind.config import load_spec
    from originblind.run.orchestrator import completeness
    from originblind.run.plan import ordered

    tp = _throughput()
    known = load_models(model_sources(configs, roster))
    want = list(dict.fromkeys(models)) if models else None
    pairs: list[dict] = []
    cfg_rows: list[dict] = []
    unservable: list[dict] = []
    seen: set[str] = set()
    names: dict[str, str] = {}
    for c in configs:
        spec = load_spec(c)
        if spec.name in names:
            raise ValueError(f"configs {names[spec.name]} and {c} share the name {spec.name!r} "
                             "(they would write the same results directory)")
        names[spec.name] = str(c)
        items = ordered(spec)
        by_model: dict[str, list] = {}
        for it in items:
            by_model.setdefault(it.model_key, []).append(it)
        rate = tp.get(spec.driver, _FALLBACK_THROUGHPUT["chromium"])
        cfg_rows.append({"config": str(c), "name": spec.name, "driver": spec.driver, "n_episodes": len(items),
                         "models": [m.key for m in spec.models], "eps_per_gpu_hour": rate})
        out = Path(results) / namespace / spec.name / "episodes.jsonl"
        for m in spec.models:
            seen.add(m.key)
            if want is not None and m.key not in want:
                continue
            mitems = by_model.get(m.key, [])
            if not servable(m):
                unservable.append({"config": str(c), "name": spec.name, "model": m.key,
                                   "reason": f"{m.hf_repo} is a scripted stand-in"})
                continue
            comp = completeness(spec, mitems, out) if out.exists() else {
                "planned": len(mitems), "complete": 0, "missing": len(mitems)}
            pairs.append({"config": str(c), "name": spec.name, "model": m.key, "served_name": m.served_name,
                          "driver": spec.driver, "runner": "lab" if spec.driver == "chromium-headful" else "host",
                          "weights_key": serve_plan(m.key, known).weights_key, "planned": comp["planned"],
                          "complete": comp["complete"], "missing": comp["missing"],
                          "gpu_h": round(len(mitems) / rate, 3), "gpu_h_remaining": round(comp["missing"] / rate, 3),
                          "out": str(out)})
    if want is not None:
        unknown = [k for k in want if k not in seen]
        if unknown:
            raise KeyError(f"--models {unknown} not in any selected config (models there: {sorted(seen)})")

    order = {m.key: i for i, m in enumerate(known)}
    by_weights: dict[str, list[dict]] = {}
    for p in pairs:
        by_weights.setdefault(p["weights_key"], []).append(p)
    served_by: dict[str, str] = {}
    groups = []
    for gi, wk in enumerate(sorted(by_weights, key=lambda k: order.get(k, len(order))), 1):
        gp = sorted(by_weights[wk], key=lambda p: p["runner"] == "lab")  # stable: host first
        plan = serve_plan(wk, known)
        for name in plan.served_names:
            if served_by.setdefault(name, wk) != wk:
                raise ValueError(f"served name {name!r} would be served from two different weights "
                                 f"({served_by[name]} and {wk}); records could not be attributed")
        groups.append({"index": gi, "weights_key": wk, "start_key": wk, "hf_repo": plan.hf_repo,
                       "revision": plan.revision, "served_names": list(plan.served_names), "vision": plan.vision,
                       "max_model_len": plan.max_model_len, "members": list(plan.members), "pairs": gp,
                       "planned": sum(p["planned"] for p in gp), "missing": sum(p["missing"] for p in gp),
                       "gpu_h": round(sum(p["gpu_h"] for p in gp), 2),
                       "gpu_h_remaining": round(sum(p["gpu_h_remaining"] for p in gp), 2)})
    per_model: dict[str, dict] = {}
    for p in pairs:
        d = per_model.setdefault(p["model"], {"planned": 0, "missing": 0, "gpu_h": 0.0, "gpu_h_remaining": 0.0})
        for k in d:
            d[k] = round(d[k] + p[k], 3)
    return {"namespace": namespace, "results": str(Path(results) / namespace), "configs": cfg_rows,
            "models_filter": want, "groups": groups, "unservable": unservable, "per_model": per_model,
            "throughput_eps_per_gpu_hour": tp,
            "totals": {"servers": len(groups), "pairs": len(pairs),
                       "planned": sum(p["planned"] for p in pairs), "missing": sum(p["missing"] for p in pairs),
                       "gpu_h": round(sum(p["gpu_h"] for p in pairs), 2),
                       "gpu_h_remaining": round(sum(p["gpu_h_remaining"] for p in pairs), 2)}}


def schedule_tsv(s: dict) -> str:
    """Tab-separated rows for bash (scripts/run_full.sh):
    G <i> <weights_key> <served,names> <vision 0|1> <n_pairs> <planned> <missing> <gpu_h_remaining>
    P <i> <runner> <config> <name> <model> <driver> <planned> <complete> <missing> <gpu_h_remaining>
    U <config> <model> <reason>"""
    rows = []
    for g in s["groups"]:
        rows.append(["G", g["index"], g["weights_key"], ",".join(g["served_names"]), int(g["vision"]),
                     len(g["pairs"]), g["planned"], g["missing"], g["gpu_h_remaining"]])
        for p in g["pairs"]:
            rows.append(["P", g["index"], p["runner"], p["config"], p["name"], p["model"], p["driver"],
                         p["planned"], p["complete"], p["missing"], p["gpu_h_remaining"]])
    for u in s["unservable"]:
        rows.append(["U", u["config"], u["model"], u["reason"]])
    return "".join("\t".join(str(x) for x in r) + "\n" for r in rows)


def schedule_text(s: dict) -> str:
    t = s["totals"]
    lines = [f"model-major schedule: namespace={s['namespace']}  results={s['results']}",
             f"  {t['servers']} server session(s), {t['pairs']} (config, model) pairs, "
             f"{t['planned']:,} episodes planned, {t['missing']:,} to run, "
             f"~{t['gpu_h']:.2f} GPU-h projected (~{t['gpu_h_remaining']:.2f} remaining)"]
    hdr = f"      {'config':<20} {'model':<18} {'driver':<17} {'runner':<6} {'planned':>8} {'to run':>8} {'GPU-h':>7}"
    for g in s["groups"]:
        lines.append(f"  [{g['index']}/{len(s['groups'])}] weights {g['weights_key']} -> {MODEL_ROOT}/{g['weights_key']}"
                     f"  served as {', '.join(g['served_names'])}  ({'vision' if g['vision'] else 'text'};"
                     f" max_model_len {g['max_model_len']}; {str(g['revision'])[:12]})")
        lines.append(hdr)
        for p in g["pairs"]:
            lines.append(f"      {p['name']:<20} {p['model']:<18} {p['driver']:<17} {p['runner']:<6} "
                         f"{p['planned']:>8,} {p['missing']:>8,} {p['gpu_h_remaining']:>7.2f}")
        lines.append(f"      {'group total':<64} {g['planned']:>8,} {g['missing']:>8,} {g['gpu_h_remaining']:>7.2f}")
    if s["per_model"]:
        lines.append("  per model: " + "; ".join(
            f"{k} {v['planned']:,} eps ({v['missing']:,} to run) ~{v['gpu_h_remaining']:.2f} GPU-h"
            for k, v in s["per_model"].items()))
    for u in s["unservable"]:
        lines.append(f"  not served: {u['name']} x {u['model']} ({u['reason']})")
    tp = s["throughput_eps_per_gpu_hour"]
    lines.append("  throughput assumed (eps/GPU-h, as `originblind plan --cost`): "
                 + ", ".join(f"{k} {v:g}" for k, v in tp.items()))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI

def _common(p: argparse.ArgumentParser, *, key: bool = False) -> None:
    if key:
        p.add_argument("--key", required=True, help="model key (roster or --configs)")
        p.add_argument("--configs", nargs="*", default=[], help="experiment configs defining extra models")
        p.add_argument("--roster", default=None, help="roster file (default configs/roster.yaml)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--container", default=CONTAINER)


def _plan_from_args(a) -> ServePlan:
    return serve_plan(a.key, load_models(model_sources(a.configs, a.roster)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="originblind serve", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("start", help="start the vLLM container for one model and verify its identity")
    _common(p, key=True)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--network", default=NETWORK)
    p.add_argument("--image", default=IMAGE)
    p.add_argument("--logdir", default="results/logs")
    p.add_argument("--timeout", type=float, default=READY_TIMEOUT_S, help="seconds to wait for /health")
    p.add_argument("--keep-on-failure", action="store_true", help="keep a failed container for inspection")
    p.add_argument("--print-argv", action="store_true", help="print the docker command and exit (no Docker)")
    p.add_argument("--stop-file", default=None, help="refuse (exit 6) while this file exists (the monitor's alarm)")
    p.add_argument("--no-gate", action="store_true", help="skip the pre-serve gate (manual diagnosis only)")
    p.add_argument("--no-limits", action="store_true", help="start the container without --memory/--pids-limit")
    p = sub.add_parser("stop", help="save the container log to --logdir, then remove the container")
    p.add_argument("--container", default=CONTAINER)
    p.add_argument("--logdir", default="results/logs")
    p = sub.add_parser("status", help="container state and served models as JSON")
    _common(p)
    p = sub.add_parser("identity", help="check /v1/models against the expected served names and root")
    _common(p, key=True)
    p = sub.add_parser("logs", help="print the container log")
    p.add_argument("--container", default=CONTAINER)
    p.add_argument("--tail", type=int, default=200)
    p = sub.add_parser("schedule", help="the model-major schedule for scripts/run_full.sh")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--namespace", default="full")
    p.add_argument("--results", default="results")
    p.add_argument("--roster", default=None)
    p.add_argument("--format", choices=["json", "tsv", "text"], default="json")
    p.add_argument("--json-out", default=None)
    p.add_argument("--tsv-out", default=None)
    a = ap.parse_args(argv)

    if a.action == "schedule":
        try:
            s = build_schedule(a.configs, models=a.models or None, namespace=a.namespace, results=a.results,
                               roster=a.roster)
        except Exception as e:  # noqa: BLE001 - invalid configuration of any kind
            _err(f"invalid schedule: {e}")
            return EXIT_CONFIG
        for path, text in ((a.json_out, json.dumps(s, indent=2)), (a.tsv_out, schedule_tsv(s))):
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(text)
        print({"json": json.dumps(s, indent=2), "tsv": schedule_tsv(s), "text": schedule_text(s)}[a.format],
              end="" if a.format != "json" else "\n")
        return EXIT_OK
    if a.action == "stop":
        return stop(container=a.container, logdir=Path(a.logdir))
    if a.action == "logs":
        r = _docker(["logs", "--tail", str(a.tail), a.container], timeout=120, merge=True)
        print(r.stdout or "", end="")
        return EXIT_OK if r.returncode == 0 else EXIT_FAIL
    if a.action == "status":
        rc, out = status(container=a.container, port=a.port)
        print(json.dumps(out, indent=2))
        return rc
    try:
        plan = _plan_from_args(a)
    except Exception as e:  # noqa: BLE001
        _err(f"invalid model selection: {e}")
        return EXIT_CONFIG
    if a.action == "identity":
        rc, out = identity(plan, port=a.port)
        print(json.dumps(out, indent=2))
        return rc
    try:
        limits = {} if a.no_limits else default_limits()
    except ValueError as e:
        _err(f"invalid resource budget: {e}")
        return EXIT_CONFIG
    if a.print_argv:
        print(shlex.join(docker_run_argv(plan, port=a.port, gpu_mem=a.gpu_mem, network=a.network,
                                         image=a.image, container=a.container, limits=limits)))
        return EXIT_OK
    return start(plan, port=a.port, gpu_mem=a.gpu_mem, network=a.network, logdir=Path(a.logdir),
                 timeout=a.timeout, image=a.image, container=a.container, keep_on_failure=a.keep_on_failure,
                 limits=limits, gate=not a.no_gate, stop_file=Path(a.stop_file) if a.stop_file else None)


if __name__ == "__main__":
    raise SystemExit(main())
