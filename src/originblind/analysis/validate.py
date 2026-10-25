"""`originblind validate <exp_dir> --config <cfg> [--allow-partial]`: is an experiment's episode
store a complete, uncontaminated realisation of its plan?

Compares the deterministic plan (run.plan.ordered on the config) with the store:
  - per (model, cell) planned / complete / missing (cell = every factor except brand/replicate), and
    per model (complete | partial | not_started: model-major runs add models one at a time);
  - attempts per status (infra_error, harness_error, excluded) mapped to their plan cell;
  - the pre-registered per-cell exclusion rate excluded / (complete + excluded), flagged above the
    prereg threshold (prereg.yaml `exclusions.instrument`, "> 5% exclusions flagged");
  - duplicate complete records, records whose episode_id is not in the plan, corrupt lines;
  - oracle channel agreement (twin log vs browser log; real-browser drivers only);
  - run identity: config_hash / plan_sha256 / code_sha256 must agree across manifest.json and every
    per-invocation manifests/*.json, and with the config being validated; every code_sha256 change
    across invocations is listed with the invocations (manifests/<utc>.json) that recorded it;
  - stimulus environment: the configuration's environment (lab | realistic) against the manifests'
    and every complete record's (`factors.environment`; records predating the factor inherit the
    manifest's). A store mixing environments is a contaminated realisation of its plan.

Writes validation.json next to episodes.jsonl. Exit 0 if every planned episode has a complete
record (or --allow-partial), 2 if not, 5 if the config cannot be loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from originblind.analysis.load import CELL_COLUMNS, ExperimentData, load_experiment
from originblind.analysis.prereg import DEFAULT_PREREG, Prereg, load_prereg

IDENTITY_KEYS = ("config_hash", "plan_sha256", "code_sha256")
STATUSES = ("complete", "infra_error", "harness_error", "excluded")


def _cell(it, environment: str) -> tuple:
    from originblind.types import OriginClass

    return (it.model_key, it.flow, it.origin_class, OriginClass(it.origin_class).adversary, it.secret,
            it.semantics, it.arm, it.defense, it.driver, environment)


def spec_from_manifest(data: ExperimentData):
    """The ExperimentSpec stored in a v2 manifest (None for v1 stores or unreadable specs)."""
    from originblind.config import ExperimentSpec

    if not data.spec:
        return None
    try:
        return ExperimentSpec(**data.spec)
    except Exception:  # noqa: BLE001 - a stale/invalid stored spec is reported, not fatal
        return None


def code_changes(data: ExperimentData) -> list[dict]:
    """Every code_sha256 recorded by the per-invocation manifests (in invocation order), with the
    invocations that ran under it: more than one entry = the harness code changed mid-run."""
    out: list[dict] = []
    for m in data.manifests:
        sha = m.get("code_sha256")
        if not sha:
            continue
        if not out or out[-1]["code_sha256"] != sha:
            out.append({"code_sha256": str(sha), "first_invocation": m.get("_file"), "invocations": []})
        out[-1]["invocations"].append(m.get("_file"))
    return out


def identity_check(data: ExperimentData, spec=None) -> dict:
    mans = ([data.manifest] if data.manifest else []) + list(data.manifests)
    values = {k: sorted({str(m.get(k)) for m in mans if m.get(k)}) for k in IDENTITY_KEYS}
    out: dict = {"n_manifests": len(mans), "values": values,
                 "consistent": all(len(v) <= 1 for v in values.values()),
                 "code_sha256_changes": code_changes(data),
                 "prereg_sha256": sorted({str(m.get("prereg_sha256")) for m in mans if m.get("prereg_sha256")}),
                 "git_sha": sorted({str((m.get("git") or {}).get("sha")) for m in mans if m.get("git")}),
                 "git_dirty": sorted({bool((m.get("git") or {}).get("dirty")) for m in mans if m.get("git")})}
    if spec is not None:
        from originblind.run.plan import plan_hash

        want = {"config_hash": spec.config_hash(), "plan_sha256": plan_hash(spec)}
        out["matches_config"] = {k: (values[k] == [v]) if values[k] else None for k, v in want.items()}
    try:
        from originblind.run.plan import code_hash

        cur = code_hash()
        out["code_matches_current"] = (values["code_sha256"] == [cur]) if values["code_sha256"] else None
    except Exception:  # noqa: BLE001
        out["code_matches_current"] = None
    return out


def validate_experiment(exp_dir: str | Path, spec=None, *, prereg: Prereg | None = None,
                        data: ExperimentData | None = None) -> dict:
    """Validation report (dict) for one experiment directory. `spec` defaults to the spec stored
    in the run manifest; without any spec only plan-free checks are made."""
    exp_dir = Path(exp_dir)
    data = data or load_experiment(exp_dir)
    spec_source = "config" if spec is not None else None
    if spec is None:
        spec = spec_from_manifest(data)
        spec_source = "manifest" if spec is not None else None
    threshold = prereg.exclusion_threshold if prereg is not None else 0.05
    status_counts = Counter(str(s) for s in data.attempts["status"]) if len(data.attempts) else Counter()
    rep: dict = {
        "exp_dir": str(exp_dir), "experiment": data.experiment, "episodes_sha256": data.sha256,
        "spec_source": spec_source, "config_name": getattr(spec, "name", None),
        "n_lines": data.n_lines, "n_corrupt_lines": data.n_corrupt, "n_records": int(len(data.attempts)),
        "statuses": {s: int(status_counts.get(s, 0)) for s in STATUSES}
        | {"other": int(sum(v for k, v in status_counts.items() if k not in STATUSES))},
        "schema_versions": data.schema_versions,
        "duplicates": {"extra_complete_records": data.n_duplicate_complete},
        "probes": {"keys": len(data.probes), "corrupt_lines": data.n_probe_corrupt},
        "exclusion_threshold": threshold,
    }
    eps = data.episodes
    complete_ids = set(eps["episode_id"]) if len(eps) else set()

    # attempts per episode id (any status)
    att_by_id: dict[str, Counter] = defaultdict(Counter)
    for eid, st in zip(data.attempts["episode_id"], data.attempts["status"]):
        att_by_id[str(eid)][str(st)] += 1
    rep["attempts"] = {"episode_ids_with_retries": int(sum(1 for c in att_by_id.values() if sum(c.values()) > 1)),
                       "max_attempts_per_id": int(max((sum(c.values()) for c in att_by_id.values()), default=0))}

    problems: list[str] = []
    if spec is not None:
        from originblind.run.plan import ordered

        items = ordered(spec)
        planned_ids = {it.episode_id for it in items}
        spec_env = str(getattr(spec, "environment", None) or data.environment)
        cells: dict[tuple, dict] = {}
        for it in items:
            c = cells.setdefault(_cell(it, spec_env), {"planned": 0, "complete": 0, "infra_error": 0,
                                                       "harness_error": 0, "excluded": 0})
            c["planned"] += 1
            if it.episode_id in complete_ids:
                c["complete"] += 1
            for st in ("infra_error", "harness_error", "excluded"):
                c[st] += att_by_id.get(it.episode_id, Counter()).get(st, 0)
        rows = []
        for key in sorted(cells):
            c = cells[key]
            denom = c["complete"] + c["excluded"]
            rate = c["excluded"] / denom if denom else 0.0
            rows.append(dict(zip(CELL_COLUMNS, key)) | c | {
                "missing": c["planned"] - c["complete"], "exclusion_rate": round(rate, 6),
                "flag_exclusion": rate > threshold})
        unplanned = sorted(complete_ids - planned_ids)
        by_model: dict[str, dict] = {}
        for r in rows:
            b = by_model.setdefault(r["model"], {"planned": 0, "complete": 0, "missing": 0, "infra_error": 0,
                                                 "harness_error": 0, "excluded": 0, "flagged_cells": 0})
            for k in ("planned", "complete", "missing", "infra_error", "harness_error", "excluded"):
                b[k] += r[k]
            b["flagged_cells"] += int(r["flag_exclusion"])
        for b in by_model.values():
            b["status"] = "complete" if b["missing"] == 0 else ("not_started" if b["complete"] == 0 else "partial")
        rep.update({
            "planned": len(items), "complete": len(complete_ids & planned_ids),
            "missing": len(planned_ids - complete_ids),
            "unplanned": {"complete_records": len(unplanned), "examples": unplanned[:5]},
            "by_model": dict(sorted(by_model.items())),
            "cells": rows, "n_cells": len(rows),
            "flagged_cells": sum(1 for r in rows if r["flag_exclusion"]),
        })
        rep["complete_run"] = rep["missing"] == 0
        if unplanned:
            problems.append(f"{len(unplanned)} complete records are not in the plan")
        if rep["flagged_cells"]:
            problems.append(f"{rep['flagged_cells']} cells exceed the {threshold:.0%} exclusion threshold")
    else:
        n_planned = (data.manifest or {}).get("n_planned")
        rep.update({"planned": n_planned, "complete": len(complete_ids),
                    "missing": (n_planned - len(complete_ids)) if isinstance(n_planned, int) else None,
                    "by_model": {}, "cells": [], "n_cells": 0, "flagged_cells": 0,
                    "unplanned": {"complete_records": None, "examples": []}})
        rep["complete_run"] = (rep["missing"] == 0) if rep["missing"] is not None else None
        problems.append("no plan available (no --config and no spec in manifest.json): completeness from "
                        "manifest n_planned only" if n_planned is not None else
                        "no plan available: completeness unknown")

    # stimulus environment: configuration vs manifests vs records
    rec_env = Counter(str(e) for e in eps["environment"]) if len(eps) else Counter()
    src = Counter(str(e) for e in eps["environment_source"]) if len(eps) else Counter()
    want_env = str(getattr(spec, "environment", None) or data.environment) if spec is not None else data.environment
    env_rep = {"configured": want_env, "manifests": data.manifest_environments,
               "records": dict(sorted(rec_env.items())), "record_source": dict(sorted(src.items())),
               "consistent": set(rec_env) <= {want_env} and set(data.manifest_environments) <= {want_env}}
    rep["environment"] = env_rep
    if not env_rep["consistent"]:
        problems.append(f"environment mismatch: configured {want_env}, manifests {data.manifest_environments}, "
                        f"records {dict(rec_env)}")

    # oracle channel agreement (meaningful only with a real browser: channel B exists)
    if len(eps):
        br = eps[eps["driver"].astype(str).str.startswith("chromium")]
        ca = br["channel_agreement"].dropna()
        rep["channel_agreement"] = {"applicable": bool(len(br)), "n": int(len(ca)), "agree": int(ca.sum()),
                                    "rate": round(float(ca.mean()), 6) if len(ca) else None}
    else:
        rep["channel_agreement"] = {"applicable": False, "n": 0, "agree": 0, "rate": None}

    ident = identity_check(data, spec)
    rep["identity"] = ident
    if not ident["consistent"]:
        problems.append("run identity differs across manifests: " +
                        ", ".join(k for k, v in ident["values"].items() if len(v) > 1))
    if len(ident["code_sha256_changes"]) > 1:
        problems.append("harness code changed across invocations: " + " -> ".join(
            f"{c['code_sha256'][:12]} ({c['first_invocation']}, {len(c['invocations'])} invocation(s))"
            for c in ident["code_sha256_changes"]))
    if any(v is False for v in (ident.get("matches_config") or {}).values()):
        problems.append("manifest identity does not match the validated config")
    if data.n_corrupt:
        problems.append(f"{data.n_corrupt} corrupt JSONL lines skipped")
    if data.n_duplicate_complete:
        problems.append(f"{data.n_duplicate_complete} duplicate complete records ignored (first kept)")
    if rep["statuses"]["harness_error"]:
        problems.append(f"{rep['statuses']['harness_error']} harness_error attempts (instrument bugs)")
    if rep.get("missing"):
        problems.append(f"{rep['missing']} planned episodes have no complete record")
    rep["problems"] = problems
    return rep


def write_validation(rep: dict, exp_dir: Path, out: Path | None = None) -> Path:
    path = out or (Path(exp_dir) / "validation.json")
    path.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="originblind validate",
                                 description="Compare an experiment's episode store with its plan.")
    ap.add_argument("exp_dir", help="experiment directory (or its episodes.jsonl)")
    ap.add_argument("--config", default=None, help="experiment config (default: the spec stored in manifest.json)")
    ap.add_argument("--allow-partial", action="store_true", help="exit 0 even if planned episodes are missing")
    ap.add_argument("--prereg", default=str(DEFAULT_PREREG), help="pre-registration (exclusion threshold)")
    ap.add_argument("--out", default=None, help="output path (default: <exp_dir>/validation.json)")
    args = ap.parse_args(argv)

    exp_dir = Path(args.exp_dir)
    if exp_dir.suffix == ".jsonl":
        exp_dir = exp_dir.parent
    spec = None
    if args.config:
        try:
            from originblind.config import load_spec

            spec = load_spec(args.config)
        except Exception as e:  # noqa: BLE001
            print(f"invalid config: {e}", file=sys.stderr)
            return 5
    try:
        prereg = load_prereg(args.prereg) if args.prereg and Path(args.prereg).exists() else None
    except Exception as e:  # noqa: BLE001
        print(f"invalid prereg: {e}", file=sys.stderr)
        return 5
    if not (exp_dir / "episodes.jsonl").exists():
        print(f"no episodes.jsonl in {exp_dir}", file=sys.stderr)
    rep = validate_experiment(exp_dir, spec, prereg=prereg)
    if exp_dir.exists():
        path = write_validation(rep, exp_dir, Path(args.out) if args.out else None)
        rep["written_to"] = str(path)
    brief = {k: rep.get(k) for k in ("experiment", "config_name", "planned", "complete", "missing", "statuses",
                                      "n_corrupt_lines", "flagged_cells", "channel_agreement", "complete_run",
                                      "problems", "written_to")}
    brief["environment"] = (rep.get("environment") or {}).get("configured")
    brief["by_model"] = {m: f"{b['complete']}/{b['planned']} ({b['status']})" for m, b in (rep.get("by_model") or {}).items()}
    brief["identity_consistent"] = rep["identity"]["consistent"]
    print(json.dumps(brief, indent=2, default=str))
    if rep.get("complete_run") or args.allow_partial:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
