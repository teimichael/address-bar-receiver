"""The orchestrator + aggregation run end to end with scripted backends and produce a
coherent claims file. Proves the record schema, resume, and analysis wire together."""

import json
from pathlib import Path

from originblind.analysis.aggregate import load_records, write_outputs
from originblind.config import ExperimentSpec, ModelSpec
from originblind.controls import always_fill
from originblind.llm.mock import ScriptedBackend
from originblind.run.orchestrator import run_plan
from originblind.run.plan import ordered


def _spec():
    return ExperimentSpec(
        name="e2e", master_seed_hex="03" * 16,
        brands=["acme", "vantafin"],
        models=[ModelSpec(key="mock", hf_repo="x", served_name="mock", pipeline="text")],
        origin_classes=["exact", "typosquat", "a2_action"],
        secrets=["password"], flows=["login"], observation_arms=["T1"],
        defenses=["D0"], replicates=2, recognition_probe=False,
    )


def test_run_and_resume_and_analyze(tmp_path: Path):
    spec = _spec()
    items = ordered(spec)
    out = tmp_path / "episodes.jsonl"

    def factory(_served):
        return ScriptedBackend(always_fill())

    r1 = run_plan(spec, items, out, factory, probe=False, namespace="test")
    assert r1["run"] == len(items) and r1["infra_error"] == 0 and r1["harness_error"] == 0

    # resume: nothing new runs
    r2 = run_plan(spec, items, out, factory, probe=False, namespace="test")
    assert r2["run"] == 0 and r2["skipped"] == len(items)

    recs = load_records(out)
    assert len(recs) == len(items)
    # always-fill: leak on typosquat and a2_action, release on exact
    by_oc = {}
    for rec in recs:
        by_oc.setdefault(rec["factors"]["origin_class"], []).append(rec["outcome"])
    assert all(o["r_bound"] and not o["leak"] for o in by_oc["exact"])
    assert all(o["leak"] for o in by_oc["typosquat"])
    assert all(o["leak"] for o in by_oc["a2_action"])

    summ = write_outputs(out, tmp_path / "derived")
    assert summ["n_records"] == len(items)
    claims = json.loads((tmp_path / "derived" / "claims.json").read_text())
    # H1 A2 OSI for the always-fill control should read as blind (~0)
    key = next(k for k in claims if k.startswith("H1_A2_OSI"))
    assert claims[key]["osi"] < 0.2
