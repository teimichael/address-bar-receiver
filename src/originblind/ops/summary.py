"""`originblind ops summarize <rundir>`: the outcome-blind verdict on a supervised check run.

It reads only operational records -- RUN_STATUS.json, logs/gate_*.json, logs/monitor_*.jsonl and
logs/serve_*.json -- never an episode store. scripts/06_ready_check.sh uses it to decide whether
the host held the largest server switch (Ministral-3-14B -> Qwen3.5-35B-A3B) under the real run's
concurrency with headroom to spare (README.md). The run passes only if:
  - it finished with exit 0 and no step failed;
  - every gate passed with the Windows probe working and commit headroom >= what its server needed;
  - no monitor sample was critical;
  - fresh Windows samples exist and their minimum commit headroom stayed >= the threshold.
Warnings, the minimum available memory, NVIDIA shared usage, PSI and container peaks are reported.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def _gb(v) -> float | None:
    return None if v is None else round(float(v) / 1e9, 1)


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def summarize(rundir: Path, *, min_commit_headroom_gb: float = 40.0, max_host_age_s: float = 30.0) -> dict:
    rundir = Path(rundir)
    logs = rundir / "logs"
    problems: list[str] = []

    st = _load(rundir / "RUN_STATUS.json")
    if not isinstance(st, dict):
        problems.append("no readable RUN_STATUS.json")
        st = {}
    else:
        if st.get("state") != "finished" or st.get("exit_code") != 0:
            problems.append(f"the run ended {st.get('state')} (exit {st.get('exit_code')})")
        bad = [f"{s.get('step')}={s.get('status')}" for s in st.get("steps") or []
               if s.get("status") not in ("ok", "skipped") and s.get("step") != "finish"]
        if bad:
            problems.append("steps not ok: " + ", ".join(bad))

    gates = []
    for p in sorted(logs.glob("gate_*.json")):
        g = _load(p)
        if not isinstance(g, dict):
            problems.append(f"unreadable {p.name}")
            continue
        steps = g.get("steps") or {}
        mem = steps.get("memory") or {}
        head, need = mem.get("commit_headroom"), mem.get("commit_needed")
        gates.append({"key": g.get("key"), "ok": g.get("ok"), "reason": g.get("reason"),
                      "settle_waited_s": (steps.get("settle") or {}).get("waited_s"),
                      "quiesce_waited_s": (steps.get("quiesce") or {}).get("waited_s"),
                      "commit_headroom_gb": _gb(head), "commit_needed_gb": _gb(need),
                      "host_probe": mem.get("host_probe")})
        if not g.get("ok"):
            problems.append(f"gate {g.get('key')} refused: {g.get('reason')}")
        elif not mem.get("host_probe"):
            problems.append(f"gate {g.get('key')} passed without the Windows probe (commit headroom unchecked)")
        elif head is None or need is None:
            problems.append(f"gate {g.get('key')} recorded no commit-headroom check")
        elif head < need:
            problems.append(f"gate {g.get('key')}: commit headroom {_gb(head)} GB < {_gb(need)} GB needed")
    if not gates:
        problems.append("no gate record: no server was started")

    n = 0
    heads: list[tuple[float, str]] = []
    avail, shared, vm_avail, psi_full, psi_some, swap = [], [], [], [], [], []
    crit: list[str] = []
    warns: Counter = Counter()
    for p in sorted(logs.glob("monitor_*.jsonl")):
        with p.open(errors="replace") as fh:
            for line in fh:
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(s, dict):
                    continue
                n += 1
                if s.get("critical"):
                    crit.append(f"{s.get('t')}: {s['critical']}")
                for w in s.get("warn") or []:
                    warns[re.sub(r"\d+(\.\d+)?", "#", str(w))] += 1
                h, age = s.get("host"), s.get("host_age_s")
                if isinstance(h, dict) and (age is None or age <= max_host_age_s):
                    if h.get("commit_limit") is not None and h.get("committed") is not None:
                        heads.append((float(h["commit_limit"]) - float(h["committed"]), str(s.get("t"))))
                    if h.get("available") is not None:
                        avail.append(float(h["available"]))
                    if h.get("gpu_shared") is not None:
                        shared.append(float(h["gpu_shared"]))
                vm = s.get("vm") or {}
                if vm.get("available") is not None:
                    vm_avail.append(float(vm["available"]))
                if vm.get("swap_used") is not None:
                    swap.append(float(vm["swap_used"]))
                pm = (s.get("psi") or {}).get("memory") or {}
                if (pm.get("full") or {}).get("avg10") is not None:
                    psi_full.append(float(pm["full"]["avg10"]))
                if (pm.get("some") or {}).get("avg10") is not None:
                    psi_some.append(float(pm["some"]["avg10"]))
    if crit:
        problems.append(f"{len(crit)} critical monitor sample(s), first {crit[0]}")
    low = min(heads) if heads else None
    if low is None:
        problems.append("no fresh Windows sample with commit counters (powershell.exe interop?)")
    elif low[0] < min_commit_headroom_gb * 1e9:
        problems.append(f"Windows commit headroom fell to {low[0] / 1e9:.1f} GB at {low[1]} "
                        f"(< {min_commit_headroom_gb:g} GB)")

    servers = []
    for p in sorted(logs.glob("serve_*.json")):
        d = _load(p)
        if isinstance(d, dict):
            cg = d.get("cgroup_after_ready") or {}
            servers.append({"key": (d.get("plan") or {}).get("key") or p.stem.split("_")[1],
                            "ready_s": d.get("ready_s"), "memory_peak_gib": None if cg.get("memory_peak") is None
                            else round(cg["memory_peak"] / 2**30, 1), "memory_limit": (d.get("limits") or {}).get("memory")})

    return {"rundir": str(rundir), "ok": not problems, "problems": problems,
            "state": st.get("state"), "exit_code": st.get("exit_code"), "overrides": st.get("overrides"),
            "gates": gates, "servers": servers, "monitor_samples": n, "windows_samples": len(heads),
            "min_commit_headroom_gb": None if low is None else round(low[0] / 1e9, 1),
            "min_commit_headroom_at": None if low is None else low[1],
            "threshold_commit_headroom_gb": min_commit_headroom_gb,
            "min_windows_available_gb": round(min(avail) / 1e9, 1) if avail else None,
            "max_gpu_shared_mib": round(max(shared) / 2**20) if shared else None,
            "min_vm_available_gib": round(min(vm_avail) / 2**30, 1) if vm_avail else None,
            "max_vm_swap_used_gib": round(max(swap) / 2**30, 2) if swap else None,
            "max_psi_memory_full_avg10": max(psi_full) if psi_full else None,
            "max_psi_memory_some_avg10": max(psi_some) if psi_some else None,
            "critical": crit[:5], "warnings": dict(warns.most_common(10))}
