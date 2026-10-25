"""Recompute released evidence and reconstruct retained browser-use summaries."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from verify import ROOT, integrity, numeric_outputs, safe_path

RUN_OUTPUT: Path | None = None


def main() -> int:
    global RUN_OUTPUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-only", action="store_true", help="skip empirical figure rendering")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT.is_relative_to(out) or (out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "outputs")):
        raise ValueError("output must be under outputs/ or outside the release")
    out.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in out.rglob("*")):
        raise ValueError("output directory contains links; select a fresh --out directory")
    RUN_OUTPUT = out
    (out / "validation.json").write_text(json.dumps({"status": "running"}) + "\n")
    integrity()
    evidence = out / "input/full"
    manifest = json.loads((ROOT / "provenance.json").read_text())
    for name in ("input", "report", "generated", "figures", ".cache"):
        if not (out / name).resolve().is_relative_to(out):
            raise ValueError("output directory contains a link outside its root")
    wanted = {item["expanded"] for item in manifest["evidence_files"]}
    if evidence.exists():
        existing = {str(p.relative_to(evidence)) for p in evidence.rglob("*") if p.is_file()}
        if existing - wanted or any(p.is_symlink() for p in evidence.rglob("*")):
            raise ValueError("expanded input contains unexpected files or links; select a fresh --out directory")
    for item in manifest["evidence_files"]:
        data = safe_path(ROOT, item["stored"]).read_bytes()
        if item["compression"] == "gzip":
            data = gzip.decompress(data)
        if hashlib.sha256(data).hexdigest() != item["expanded_sha256"]:
            raise ValueError("evidence expansion hash mismatch")
        dest = safe_path(evidence, item["expanded"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    env = dict(os.environ)
    env.update(OB_OUTPUT_ROOT=str(out), OB_EVIDENCE_ROOT=str(evidence), OB_REPORT_ROOT=str(out / "report"),
               MPLCONFIGDIR=str(out / ".cache/matplotlib"), PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    spec = importlib.util.spec_from_file_location("artifact_exposure_score", ROOT / "audit/exposure/score.py")
    score = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = score
    spec.loader.exec_module(score)
    rows, errors = score.score_tree(evidence / "exposure/raw")
    retained_rows = json.loads((evidence / "exposure/exposure.json").read_text())
    if errors or len(rows) != 312 or rows != retained_rows:
        raise ValueError("exposure re-scoring differs from the retained 312 observation records")
    (out / "exposure_validation.json").write_text(json.dumps({"status": "passed", "raw_dumps": len(rows),
                                                           "errors": len(errors)}, indent=2) + "\n")
    print("Re-scored all 312 raw exposure observations: identical results.", flush=True)
    print("Recomputing the core report; withheld browser-use mechanism audit remains not_run.", flush=True)
    subprocess.run([sys.executable, "-m", "originblind.cli", "report", str(evidence), "--out", str(out / "report"),
                    "--prereg", str(ROOT / "configs/prereg.yaml"), "--no-figures"], cwd=ROOT, env=env, check=True)
    scripts = ["derived.py", "crosscheck_raw.py", "make_numbers.py", "make_tables.py", "verify_values.py"]
    if not args.data_only:
        scripts += ["fig_binding_plane.py", "fig_defense_heatmap.py"]
    for name in scripts:
        print(f"Running analysis/{name}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "analysis" / name)], cwd=ROOT, env=env, check=True)
    result = numeric_outputs(out)
    result["figures"] = "skipped_by_request" if args.data_only else ["fig_binding_plane.pdf", "fig_defense_heatmap.pdf"]
    result["fresh_model_experiments"] = "not_run"
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        if RUN_OUTPUT is not None:
            (RUN_OUTPUT / "validation.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        print(f"reproduction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
