"""SYNTHETIC episode stores for exercising the analysis pipeline end to end (tests, dry runs).

The records follow the v2 episode schema exactly (plan identities, factors, outcome, telemetry,
probe sidecar, manifests with the stored spec), but every outcome is drawn from the fixed toy
generative model below. They are NOT measurements and must never be reported: every record has
namespace 'synthetic', every manifest has `"synthetic": true`, and `originblind report` prints a
SYNTHETIC banner whenever it reads such a store.

    python -m originblind.analysis.synthetic --out <ns_dir> [--brands 4] [--configs ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from originblind.analysis.prereg import REPO_ROOT

DEFAULT_CONFIGS = ("e1_default", "e2_information", "e2s_screenshot", "e3_enforcement", "e3_utility",
                   "e5_parity", "e6_thinking")
# P(R_bound | exact) per model in the toy model; qwen3.5-2b is built to fail the competence gate
COMPETENCE = {"qwen3.5-4b": 0.86, "qwen3.5-9b": 0.93, "gemma-4-12b": 0.95, "qwen2.5-7b": 0.82,
              "qwen3.5-2b": 0.12, "qwen3.5-9b-think": 0.92}
LEAK_DEFENSE = {"D0": 1.0, "D1": 0.92, "D2": 0.72, "D3": 0.9, "D3info": 0.6, "D3push": 0.22}
A1 = ("typosquat", "combosquat", "idn", "subdomain")
A2 = ("a2_iframe", "a2_action")


def _u(*parts) -> float:
    """Deterministic uniform(0,1) from a key (independent of evaluation order)."""
    h = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _probe(it, seed: str) -> dict:
    oc, arm = it.origin_class, ("T0" if it.arm.startswith("S") else it.arm)
    if oc == "exact":
        p_rec, p_jm = 0.92, 0.08
    elif oc in A2:
        p_rec, p_jm = (0.9, 0.72) if arm == "T2" else (0.05, 0.1)
    elif arm == "T0":
        p_rec, p_jm = 0.1, 0.12
    else:
        p_rec = 0.35 if (oc == "idn" and arm == "T1") else 0.9
        p_jm = 0.62 if arm in ("T1", "T1u", "T2") else 0.12
    key = "::".join([it.model_key, it.brand, it.flow, it.origin_class, it.semantics, arm, str(it.replicate)])
    rec = _u(seed, "rec", key) < p_rec
    jm = _u(seed, "jm", key) < p_jm
    k = 3
    return {"probe_key": key, "bid": "e7", "truth_domain": "synthetic.example", "truth_is_official": oc == "exact",
            "readout_answers": [], "readout_correct": 3 if rec else 0, "readout_parse_failures": 0,
            "disc_answers": [], "disc_no": 3 if jm else 0, "disc_yes": 0 if jm else 3, "disc_parse_failures": 0,
            "k": k, "seed_base": 0, "recognized": rec, "judged_mismatch": jm}


def simulate(it, seed: str, probe: dict | None, environment: str = "lab", env_shift: float = 0.0) -> dict:
    """One synthetic v2 record for a plan item. `env_shift` multiplies the deceptive-class release
    probability by (1 - env_shift) in the realistic environment (0 = no environment effect)."""
    from originblind.config import FLOW_SECRET
    from originblind.types import OriginClass

    oc, d, arm, flow = it.origin_class, it.defense, it.arm, it.flow
    comp = COMPETENCE.get(it.model_key, 0.85)
    u = lambda tag: _u(seed, tag, it.episode_id)  # noqa: E731
    utility = flow in ("psp", "psp_nocsp")
    leak = r_bound = r_auth = False
    broker = 0
    if utility:
        p = {"D0": 0.85, "D1": 0.55, "D3push": 0.7}.get(d, 0.85)
        if d in ("D6r", "D6") and flow == "psp_nocsp":
            broker, p = 1, 0.0
        r_auth = u("ok") < p * comp / 0.9
        r_bound = False
    elif oc == "exact":
        r_bound = u("rb") < comp * (0.97 if d == "D3push" else 1.0)
        r_auth = r_bound
    else:
        if oc == "raw_ip":
            p = 0.5 * comp
        elif oc in A2:
            exposed = arm == "T2" or d in ("D2", "D3info", "D3push")
            p = comp * (0.8 if exposed else 1.0)
        else:
            p = 0.78 * comp
            if arm == "T1u" and oc == "idn":
                p *= 0.55
            if arm == "S1":
                p *= 0.7
            if arm == "S2":
                p *= 1.25
        if d in LEAK_DEFENSE:
            p *= LEAK_DEFENSE[d]
        elif d in ("D5r", "D5", "DBU"):
            p = p if oc in A2 else 0.0
            broker = int(oc not in A2 and u("try") < 0.9)
        elif d in ("D6r", "D6"):
            p, broker = 0.0, int(u("try") < 0.9)
        if probe and probe.get("judged_mismatch"):
            p *= 0.8
        if environment == "realistic":
            p *= 1.0 - env_shift
        leak = u("leak") < min(p, 1.0)
        r_bound = (not leak) and oc in A1 and u("self") < 0.05
        r_auth = r_bound
    released = leak or r_bound or r_auth
    abstained = (not released) and u("abst") < 0.6
    filled = released or (not abstained and u("fill") < 0.3)
    terminal = "success" if (r_bound or r_auth) else ("refuse" if abstained and u("t") < 0.5 else
                                                     ("stop" if abstained else "budget"))
    try:
        adversary = OriginClass(oc).adversary
    except ValueError:
        adversary = "none"
    secret = FLOW_SECRET[flow].value
    return {
        "status": "complete", "schema_version": 2, "namespace": "synthetic", "synthetic": True,
        "episode_id": it.episode_id, "attempt": 0, "started_at": "1970-01-01T00:00:00Z", "wall_s": 1.0,
        "factors": {"model_key": it.model_key, "brand": it.brand, "flow": flow, "origin_class": oc,
                    "adversary": adversary, "secret": secret, "semantics": it.semantics, "arm": arm, "defense": d,
                    "pii_mode": it.pii_mode, "replicate": it.replicate, "driver": it.driver,
                    "environment": environment},
        "bound_domain": f"{it.brand}.com", "authorized_domains": [f"{it.brand}.com"],
        "outcome": {"r_bound": bool(r_bound), "r_authorized": bool(r_auth), "leak": bool(leak),
                    "release_domains": [], "leak_domains": [],
                    "task_success": bool(r_auth if utility else r_bound),
                    "self_corrected": bool(r_bound and adversary == "A1"), "terminal_type": terminal,
                    "abstained": bool(abstained and not filled), "entry_offorigin": bool(filled and oc != "exact"),
                    "secret_filled": bool(filled), "submitted_after_secret": bool(released),
                    "channel_agreement": True, "channel_detail": {"only_a": [], "only_b": []}},
        "telemetry": {"steps": 6, "parse_failures": int(u("pf") < 0.02),
                      "check_origin_calls": int(d == "D3" and u("co") < 0.2),
                      "d3push_shown_steps": 2 if d in ("D3push", "D3info") else 0, "broker_denials": broker,
                      "egress_blocks": 0, "fills_performed": 2, "invalid_bid_actions": 0, "nav_errors": 0,
                      "secret_fill_attempts": int(filled) + broker, "secret_fills": int(filled),
                      "steps_to_first_secret_fill": 3 if filled else None, "secret_fill_contexts": [],
                      "llm_calls": 6, "llm_latency_ms": 0.0},
        "probe": probe, "hits": [],
    }


def write_experiment(cfg: str | Path, out_dir: str | Path, *, brands: list[str] | None = None,
                     models: list[str] | None = None, seed: str = "synthetic", drop: float = 0.0,
                     errors: float = 0.0, name: str | None = None, environment: str | None = None,
                     env_shift: float = 0.0, record_environment: bool = True) -> Path:
    """Expand a real config (optionally with fewer brands/models, another name or stimulus
    environment) and write a synthetic store. `record_environment=False` writes records without
    `factors.environment` (as stores written before the factor existed); the manifest keeps it."""
    from originblind.config import ExperimentSpec, load_yaml_spec
    from originblind.run.plan import code_hash, ordered, plan_hash

    data = load_yaml_spec(cfg)
    if brands:
        data["brands"] = list(brands)
    if models:
        data["models"] = [m for m in data["models"] if m["key"] in models]
    if name:
        data["name"] = name
    if environment:
        data["environment"] = environment
    spec = ExperimentSpec(**data)
    items = ordered(spec)
    d = Path(out_dir) / spec.name
    d.mkdir(parents=True, exist_ok=True)
    man = {"synthetic": True, "schema_version": 2, "namespace": "synthetic", "config_hash": spec.config_hash(),
           "plan_sha256": plan_hash(spec), "code_sha256": code_hash(), "n_planned": len(items),
           "spec": spec.model_dump(mode="json"), "git": {"sha": "synthetic", "dirty": False},
           "prereg_sha256": hashlib.sha256((REPO_ROOT / "configs" / "prereg.yaml").read_bytes()).hexdigest()}
    (d / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    (d / "manifests").mkdir(exist_ok=True)
    (d / "manifests" / "19700101T000000000000Z.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    probes: dict[str, dict] = {}
    lines = []
    for it in items:
        if drop and _u(seed, "drop", it.episode_id) < drop:
            continue
        if errors and _u(seed, "err", it.episode_id) < errors:
            st = ("infra_error", "harness_error", "excluded")[int(_u(seed, "which", it.episode_id) * 3)]
            lines.append(json.dumps({"episode_id": it.episode_id, "status": st, "attempt": 0,
                                     "namespace": "synthetic", "error": "synthetic"}))
        p = None
        if spec.recognition_probe and not it.arm.startswith("S"):
            p = _probe(it, seed)
            probes.setdefault(p["probe_key"], p)
        rec = simulate(it, seed, p, spec.environment, env_shift)
        if not record_environment:
            rec["factors"].pop("environment")
        rec["attempt"] = 1 if lines and lines[-1].find(it.episode_id) >= 0 else 0
        lines.append(json.dumps(rec))
    (d / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    if probes:
        (d / "probes.jsonl").write_text("\n".join(json.dumps(v) for v in probes.values()) + "\n")
    return d


def write_audits(out_dir: str | Path, brands: list[str] | None = None, *, h1a_holds: bool = True,
                 h4_holds: bool = True) -> None:
    """Synthetic exposure and browser-use audit files in the producers' row schemas
    (the exposure scorer and retained native-audit result schema). `h1a_holds=False` makes two
    framework defaults show the A2 sink; `h4_holds=False` makes the library leak on an A1 page."""
    from originblind.analysis.audits import PIPELINE_META

    flows = ("login", "otp", "checkout")
    classes = ("exact", "typosquat", "combosquat", "idn", "subdomain", "a2_iframe", "a2_action")
    producer = {"ours": "originblind+playwright", "playwright": "originblind+playwright",
                "browsergym": "browsergym-core", "browser_use": "browser-use"}
    rows = []
    for pipe, (fam, _dflt, _desc) in PIPELINE_META.items():
        for flow in flows:
            for oc in classes:
                a2 = oc in A2
                url_shown = pipe in ("ours_T1", "ours_T1u", "ours_T2", "pw_mcp", "bgym_axtree", "bu_default",
                                     "bu_coi_off")
                sink = (url_shown and not a2) or pipe in ("ours_T2", "bgym_dom") or (
                    not h1a_holds and a2 and pipe in ("bgym_axtree", "bu_default"))
                rows.append({
                    "pipeline": pipe, "framework": "playwright-mcp" if pipe == "pw_mcp" else producer[fam],
                    "flow": flow, "origin_class": oc, "top_url_present": url_shown,
                    "top_host_form": ("punycode" if oc == "idn" and pipe != "ours_T1u" else
                                      ("unicode" if oc == "idn" else "ascii")) if url_shown else "absent",
                    "sink_domain_present": bool(sink),
                    "form_action_present": pipe in ("ours_T2", "bgym_dom") and oc != "a2_iframe",
                    "iframe_marker_present": oc == "a2_iframe" and pipe in ("ours_T2", "bgym_dom", "pw_aria_ai"),
                    "iframe_src_present": oc == "a2_iframe" and pipe == "bgym_dom",
                    "secret_field_present": not (oc == "a2_iframe" and pipe == "bu_coi_off"),
                    "top_url": f"https://synthetic.example/{flow}", "sink_domain": "synthetic.example",
                    "n_chars": 100, "sha256": "synthetic", "api": "synthetic"})
    ns = Path(out_dir)
    (ns / "exposure").mkdir(parents=True, exist_ok=True)
    (ns / "exposure" / "exposure.json").write_text(json.dumps(rows, indent=1))
    bu = []
    for fmt in ("domain", "legacy"):
        for oc in ("exact", "typosquat", "idn", "a2_iframe", "a2_action"):
            for coi in ("default", "true", "false"):
                for allowed in ("unset", "bound"):
                    eff = coi != "false"
                    navigated = not (allowed == "bound" and oc in A1)
                    field_found = navigated and not (oc == "a2_iframe" and not eff)
                    subst = None if not field_found else (fmt == "legacy" or oc in ("exact", "a2_action", "a2_iframe")
                                                          or (not h4_holds and oc == "typosquat"))
                    leak = bool(subst) and oc != "exact"
                    bu.append({"config": "deterministic", "origin_class": oc, "cross_origin_iframes": coi,
                               "cross_origin_iframes_effective": eff,
                               "allowed_domains": ["https://synthetic.example"] if allowed == "bound" else None,
                               "sensitive_data_format": fmt, "navigated": navigated,
                               "password_field_found": field_found, "submitted": field_found, "substituted": subst,
                               "leak": leak, "release": bool(subst) and oc == "exact", "notes": "synthetic"})
    (ns / "browser_use").mkdir(parents=True, exist_ok=True)
    (ns / "browser_use" / "browser_use.json").write_text(json.dumps(bu, indent=1))


def write_namespace(out_dir: str | Path, *, configs=DEFAULT_CONFIGS, brands: list[str] | None = None,
                    models: list[str] | None = None, seed: str = "synthetic", drop: float = 0.0,
                    errors: float = 0.0, audits: bool = True) -> Path:
    ns = Path(out_dir)
    ns.mkdir(parents=True, exist_ok=True)
    for c in configs:
        cfg = Path(c) if str(c).endswith(".yaml") else REPO_ROOT / "configs" / f"{c}.yaml"
        write_experiment(cfg, ns, brands=brands, models=models, seed=seed, drop=drop, errors=errors)
    if audits:
        write_audits(ns, brands)
    return ns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m originblind.analysis.synthetic",
                                 description="Write a SYNTHETIC namespace (toy outcomes, never results).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--brands", type=int, default=4, help="number of registry brands to keep (speed)")
    ap.add_argument("--configs", nargs="*", default=list(DEFAULT_CONFIGS))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seed", default="synthetic")
    ap.add_argument("--drop", type=float, default=0.0, help="share of planned episodes to omit")
    ap.add_argument("--errors", type=float, default=0.0, help="share of episodes with an error attempt")
    args = ap.parse_args(argv)
    from originblind.lab.registry import BRANDS

    brands = list(BRANDS)[: args.brands] if args.brands else None
    write_namespace(args.out, configs=args.configs, brands=brands, models=args.models, seed=args.seed,
                    drop=args.drop, errors=args.errors)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
