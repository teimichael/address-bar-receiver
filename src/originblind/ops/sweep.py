"""Leftover containers of a crashed or killed run.

`scripts/run_full.sh` calls `ops sweep` (under its run lock) before preflight:
  - an exited/created `ob-lab-<ns>-*` container is removed after its log is saved to
    <logdir>/orphan_<name>_<utc>.log; a *running* one is refused (a second writer on the namespace's
    results volume): the operator stops it after checking that no other run is active;
  - a leftover `ob-vllm` goes through serving.vllm._remove, which saves its full log first (after
    the host hang of 2026-10-14 this preserves the crash-time 9B log as serve_qwen3.5-9b_<utc>.log).
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from originblind.ops import lock as _lock
from originblind.ops import probes


def lab_prefix(ns: str) -> str:
    """The container-name prefix lab_run.sh gives a namespace (`tr -c 'a-zA-Z0-9_.-' '-'`)."""
    return "ob-lab-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", ns) + "-"


def _docker(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([probes.docker_bin(), *args], capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, errors="replace")
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(args, 124, "", repr(e))


def sweep(ns: str, logdir: Path) -> dict:
    locked = _lock.refuse_if_locked()
    if locked:
        return {"ok": False, "error": locked}
    logdir.mkdir(parents=True, exist_ok=True)
    out: dict = {"ok": True, "lab_removed": [], "lab_running": [], "vllm": None}
    r = _docker(["ps", "-a", "--filter", "name=^ob-lab-", "--format", "{{.Names}}\t{{.State}}"], timeout=60)
    if r.returncode != 0:
        return {"ok": False, "error": f"docker ps failed: {(r.stderr or '').strip()[:200]}"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for row in (r.stdout or "").splitlines():
        name, _, state = row.partition("\t")
        if not name.startswith(lab_prefix(ns)):
            continue
        if state == "running":
            out["lab_running"].append(name)
            continue
        logs = _docker(["logs", "--timestamps", name], timeout=300)
        (logdir / f"orphan_{name}_{stamp}.log").write_text((logs.stdout or "") + (logs.stderr or ""))
        rm = _docker(["rm", "-f", name], timeout=120)
        out["lab_removed"].append({"name": name, "state": state, "removed": rm.returncode == 0})
    if out["lab_running"]:
        out["ok"] = False
        out["error"] = (f"lab container(s) still running: {out['lab_running']}. Another attempt may still be "
                        "writing this namespace; if no run is active, stop them (`docker stop <name>`) and rerun.")
        return out
    from originblind.serving import vllm

    st = vllm.container_state(vllm.CONTAINER)
    if st is not None:
        res = vllm._remove(vllm.CONTAINER, logdir, "unknown")
        out["vllm"] = res
        if not res.get("removed"):
            out["ok"] = False
            out["error"] = f"could not remove the leftover {vllm.CONTAINER}: {res.get('error')}"
    return out
