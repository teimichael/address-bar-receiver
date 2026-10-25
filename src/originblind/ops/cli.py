"""`originblind ops ...`: the operational commands used by scripts/run_full.sh and scripts/lab_run.sh.

  budget [--shell]                 the effective resource budget (OB_* overrides applied); exit 5 if invalid
  probe [--host]                   one JSON snapshot: VM memory/PSI, OOM kills, GPU (+ Windows host)
  oom                              the kernel's OOM-kill counter (one integer; -1 if unknown)
  hostcheck --namespace NS [--full] [--results R] [--configs ...]
                                   the fast host/STOP subset of preflight (run before setup); exit 1 if blocked
  store check DIR                  read-only integrity of an experiment or a namespace; exit 2 if a repair is needed
  store repair EXPDIR              manual: quarantine what the readers cannot survive (never deletes bytes)
  store fix-tail EXPDIR            newline-terminate torn JSONL tails (the only automatic fix)
  store prefix-check --host H --candidate C
                                   may C (the lab volume copy) replace H without losing host data? exit 1 if not
  monitor --rundir R --pid P --out F
                                   the flight recorder and guard (ops/monitor.py)
  sweep --ns NS --logdir D         leftover containers of a crashed run (ops/sweep.py); exit 1 if refused
  lab-env --record F [--out F] [--image I]
                                   one lab environment per run: record the lab image's packages the
                                   first time, exit 1 when a later attempt's differ (ops/labenv.py)
  summarize RUNDIR [--min-commit-headroom-gb 40]
                                   outcome-blind verdict on a supervised check run (ops/summary.py);
                                   exit 1 unless it passed
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="originblind ops", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("budget")
    p.add_argument("--shell", action="store_true", help="print BUDGET_<FIELD>=value lines for bash")
    p = sub.add_parser("probe")
    p.add_argument("--host", action="store_true", help="include the Windows host (about 2 s)")
    sub.add_parser("oom")
    p = sub.add_parser("hostcheck")
    p.add_argument("--namespace", required=True)
    p.add_argument("--results", default=None)
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--full", action="store_true")
    p = sub.add_parser("store")
    ss = p.add_subparsers(dest="action", required=True)
    q = ss.add_parser("check")
    q.add_argument("dir")
    q = ss.add_parser("repair")
    q.add_argument("dir")
    q.add_argument("--min-age-s", type=float, default=60.0)
    q = ss.add_parser("fix-tail")
    q.add_argument("dir")
    q = ss.add_parser("prefix-check")
    q.add_argument("--host", required=True)
    q.add_argument("--candidate", required=True)
    p = sub.add_parser("monitor", add_help=False)
    p = sub.add_parser("sweep")
    p.add_argument("--ns", required=True)
    p.add_argument("--logdir", required=True)
    p = sub.add_parser("lab-env")
    p.add_argument("--image", default="originblind-lab:latest")
    p.add_argument("--record", required=True, help="<ns>/lab_env.json: written once, compared afterwards")
    p.add_argument("--out", default=None, help="this attempt's record (with the package list)")
    p = sub.add_parser("summarize")
    p.add_argument("rundir")
    p.add_argument("--min-commit-headroom-gb", type=float, default=40.0)

    if argv and argv[0] == "monitor":
        from originblind.ops import monitor

        return monitor.main(argv[1:])
    a = ap.parse_args(argv)

    if a.cmd == "budget":
        from originblind.ops.budget import as_dict, load_budget

        try:
            b = as_dict(load_budget())
        except ValueError as e:
            print(f"invalid resource budget: {e}", file=sys.stderr)
            return 5
        if a.shell:
            for k, v in b.items():
                print(f"BUDGET_{k.upper()}={shlex.quote(str(int(v) if isinstance(v, bool) else v))}")
        else:
            _print(b)
        return 0
    if a.cmd == "probe":
        from originblind.ops import probes

        snap = {"vm": probes.vm_memory(), "psi": probes.vm_psi(), "oom_kill": probes.oom_kills(),
                "load": probes.loadavg(), "gpu": probes.gpu_query()}
        if a.host:
            snap["host"] = probes.host_probe()
        _print(snap)
        return 0
    if a.cmd == "oom":
        from originblind.ops import probes

        v = probes.oom_kills()
        print(-1 if v is None else v)
        return 0
    if a.cmd == "hostcheck":
        from originblind import preflight
        from originblind.config import load_spec

        models = {}
        for c in a.configs or []:
            try:
                for m in load_spec(c).models:
                    if not m.hf_repo.startswith("none/"):
                        models.setdefault(m.key, m)
            except Exception:  # noqa: BLE001 - preflight's B07 reports invalid configs
                continue
        hc = preflight.host_safety(models.values(), full=a.full)
        res = dict(hc["blockers"])
        res.update({k: v for k, v in preflight.namespace_checks([], a.namespace,
                                                                 Path(a.results or "results"), full=a.full).items()
                    if k == "B22_stop_alarm"})
        n_blocked = sum(1 for v in res.values() if v["status"] == preflight.BLOCKED)
        _print({"blockers": res, "n_blocked": n_blocked, "ready": n_blocked == 0, "overrides": hc["overrides"],
                "host": hc["host"]})
        return 0 if n_blocked == 0 else 1
    if a.cmd == "store":
        from originblind.ops import store

        d = Path(a.dir) if a.action != "prefix-check" else None
        if a.action == "check":
            res = store.check_store(d) if (d / "episodes.jsonl").exists() else store.check_namespace(d)
            _print(res)
            return 2 if res["status"] == "repair" else 0
        if a.action == "repair":
            try:
                _print(store.repair_store(d, min_age_s=a.min_age_s))
            except RuntimeError as e:
                print(f"refused: {e}", file=sys.stderr)
                return 1
            return 0
        if a.action == "fix-tail":
            _print({"dir": str(d), "newline_terminated": store.fix_tails(d)})
            return 0
        res = store.prefix_check(Path(a.host), Path(a.candidate))
        _print(res)
        return 0 if res["ok"] else 1
    if a.cmd == "sweep":
        from originblind.ops.sweep import sweep

        res = sweep(a.ns, Path(a.logdir))
        _print(res)
        return 0 if res.get("ok") else 1
    if a.cmd == "lab-env":
        from originblind.ops import labenv

        rc, res = labenv.check(a.image, Path(a.record), Path(a.out) if a.out else None)
        _print(res)
        return rc
    if a.cmd == "summarize":
        from originblind.ops.summary import summarize

        res = summarize(Path(a.rundir), min_commit_headroom_gb=a.min_commit_headroom_gb)
        _print(res)
        return 0 if res["ok"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
