"""originblind command-line entry point.

Exit codes (used by scripts/run_full.sh):
  0  done (for `run`: every planned item of the selection has a complete record)
  2  incomplete (some items still need a run; resumable)
  3  aborted by the infrastructure circuit breaker
  4  served-model identity mismatch (the vLLM endpoint does not serve the expected model)
  5  invalid configuration / refused resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import yaml

from originblind.config import ExperimentSpec
from originblind.controls import always_fill
from originblind.llm.client import InfraError, OpenAIBackend
from originblind.llm.mock import ScriptedBackend
from originblind.run.orchestrator import ResumeMismatch, completeness, run_plan
from originblind.run.plan import ordered, plan_hash, run_identity

# Episodes per GPU-hour used only for the --cost / schedule projection. Pilot v2 (RTX 5090, probes
# included) measured ~5,900-6,500 eps/h in real Chromium for 2B-12B models (Qwen3.5-9B ~3,400) and
# ~14,000-16,500 eps/h in-process (README.md); the headful lab with 4 private displays ran
# 108-episode VLM pilots at ~1,750-3,100 eps/h incl. container start. The table is conservative.
_DEFAULT_THROUGHPUT = {"inprocess": 7000.0, "chromium": 3000.0, "chromium-headful": 1500.0}


def load_spec(p: str) -> ExperimentSpec:
    from originblind.config import load_spec as _load

    return _load(p)


def _openai_factory(_served: str) -> OpenAIBackend:
    return OpenAIBackend()


def _mock_factory(_served: str) -> ScriptedBackend:
    # deterministic always-fill control, for the GPU-free plumbing smoke
    return ScriptedBackend(always_fill())


def _base_url() -> str:
    return os.environ.get("ORIGIN_BLIND_VLLM_URL", "http://localhost:8000/v1")


def _health() -> bool:
    url = _base_url().rsplit("/v1", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def check_served(spec: ExperimentSpec, model_keys: set[str]) -> dict:
    """Verify the endpoint serves every selected model under its served_name; return the payload."""
    served = OpenAIBackend().served_models()
    ids = {m.get("id") for m in served}
    want = {m.served_name for m in spec.models if m.key in model_keys}
    missing = sorted(want - ids)
    return {"served": served, "missing": missing}


def cmd_plan(args):
    spec = load_spec(args.config)
    items = ordered(spec, model=args.model)
    by_model: dict[str, int] = {}
    for it in items:
        by_model[it.model_key] = by_model.get(it.model_key, 0) + 1
    out = {"name": spec.name, **run_identity(spec), "n_episodes": len(items), "by_model": by_model,
           "driver": spec.driver, "models": [m.key for m in spec.models]}
    if args.cost:
        tp = _DEFAULT_THROUGHPUT.get(spec.driver, 900.0)
        if args.throughput:
            tp = float(args.throughput)
        out["projected_gpu_hours"] = {k: round(v / tp, 2) for k, v in by_model.items()}
        out["projected_gpu_hours_total"] = round(len(items) / tp, 2)
        out["throughput_eps_per_gpu_hour_assumed"] = tp
    print(json.dumps(out, indent=2))


def cmd_run(args):
    try:
        spec = load_spec(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"invalid config: {e}", file=sys.stderr)
        raise SystemExit(5)
    items = ordered(spec, model=args.model)
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    mock = args.backend == "mock"
    factory = _mock_factory if mock else _openai_factory
    if mock:
        os.environ["ORIGIN_BLIND_BACKEND"] = "mock"  # inherited by browser worker processes
    probe = spec.recognition_probe and not mock
    # a crash can leave a torn last line; the writers append without checking, which would glue the
    # next record onto it and lose it (README.md). Lazy import: workers never load ops.
    from originblind.ops.store import newline_terminate

    for store in (out, out.parent / "probes.jsonl"):
        if newline_terminate(store):
            print(f"store: newline-terminated the torn last line of {store}", file=sys.stderr)
    extra: dict = {"backend": args.backend, "namespace": args.namespace, "model_filter": args.model,
                   "runtime": {"workers": args.workers}}  # throughput only; not part of any identity
    if not mock and items:
        try:
            chk = check_served(spec, {it.model_key for it in items})
        except InfraError as e:
            print(f"endpoint unreachable: {e}", file=sys.stderr)
            raise SystemExit(3)
        extra["served_models"] = chk["served"]
        if chk["missing"]:
            print(f"served-model mismatch: endpoint does not serve {chk['missing']}", file=sys.stderr)
            raise SystemExit(4)
    try:
        res = run_plan(spec, items, out, factory, probe=probe, namespace=args.namespace,
                       workers=args.workers, health_check=None if mock else _health,
                       allow_code_change=args.allow_code_change, manifest_extra=extra)
    except ResumeMismatch as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(5)
    res["completeness"] = completeness(spec, items, out)
    print(json.dumps(res, indent=2))
    if res.get("aborted"):
        raise SystemExit(3)
    if res["completeness"]["missing"]:
        raise SystemExit(2)


def cmd_preflight(args):
    from originblind import preflight

    raise SystemExit(preflight.main(Path(args.repo_root) if args.repo_root else None,
                                    configs=args.configs, smoke_only=args.smoke_only, full=args.full,
                                    namespace=args.namespace, results=args.results, host_checks=args.host_checks))


def cmd_analyze(args):
    from originblind.analysis.aggregate import write_outputs

    res = write_outputs(Path(args.results), Path(args.out_dir))
    print(json.dumps(res, indent=2))


# Subcommands owned by their modules (each exposes main(argv) -> int).
_DELEGATED = {
    "models": "originblind.serving.models",   # fetch | verify | materialize | status
    "serve": "originblind.serving.vllm",      # start | stop | status | identity
    "validate": "originblind.analysis.validate",
    "report": "originblind.analysis.report",
    "lab": "originblind.lab.serve",           # lab serve: twins + proxy for external pipelines
    "ops": "originblind.ops.cli",             # budget | probe | store | monitor | sweep (README.md)
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in _DELEGATED:
        import importlib

        mod = importlib.import_module(_DELEGATED[argv[0]])
        raise SystemExit(mod.main(argv[1:]))
    ap = argparse.ArgumentParser(prog="originblind")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("config"); p.add_argument("--model", default=None)
    p.add_argument("--cost", action="store_true"); p.add_argument("--throughput", default=None)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run")
    p.add_argument("config"); p.add_argument("--out", required=True)
    p.add_argument("--namespace", default="pilot"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default=None, help="run only this model's items (model-major serving)")
    p.add_argument("--backend", choices=["openai", "mock"], default="openai")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--allow-code-change", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("analyze")
    p.add_argument("results"); p.add_argument("--out-dir", required=True)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("preflight")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--namespace", default=None, help="also check this namespace (STOP, resume identity, stores)")
    p.add_argument("--results", default=None, help="results root of --namespace (default <repo>/results)")
    p.add_argument("--host-checks", action="store_true", help="host safety checks outside --full (warnings only)")
    p.set_defaults(fn=cmd_preflight)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    main()
