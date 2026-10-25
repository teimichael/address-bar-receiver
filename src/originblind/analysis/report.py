"""`originblind report <ns_dir> [--include NS_DIR ...] [--out DIR] [--prereg configs/prereg.yaml]
                    [--legacy-pilot] [--allow-partial] [--paired-env OTHER_NS_DIR] [--audits NS_DIR]`

Turns one results namespace into every artefact the paper reports, from the raw episode stores
only (no number is ever typed by hand):

  --include               further namespace dirs merged into ONE analysis (e.g. results/pilot_v2r +
                          results/pilot_v2r_x: same environment, disjoint models); experiment dirs are
                          keyed <namespace>/<dir>, the analysis frame keeps the first complete record
                          per environment + experiment + episode_id (sources in the order given; any
                          overlap of experiment x model x environment is listed in SUMMARY.md), and
                          validation stays per experiment directory (validation/<namespace>/<dir>.json)
  --audits                the namespace dir holding exposure/ and browser_use/ (default: the first of
                          ns_dir and the --include dirs that has them)
  validation/<exp>.json   plan completeness, attrition, exclusions, identity (analysis.validate)
  claims.json / claims.md the pre-registered hypotheses, decided EXACTLY by configs/prereg.yaml
                          (analysis.claims; deterministic audits via analysis.audits), computed
                          separately per stimulus environment (lab | realistic; never pooled) under
                          `by_environment`; the confirmatory claims are the pre-registration's
                          confirmatory environment's
  tables/*.csv + *.tex    descriptive tables with every factor column (environment included), n and
                          intervals (analysis.tables); the CSV keeps full precision
  figures/*.pdf|png|csv   publication figures with their table-view twins (analysis.figures); one set
                          per environment (suffix _<environment>) when a namespace holds several
  awareness               verbalised evaluation awareness in the transcripts (analysis.envcheck):
                          tables/awareness*.csv, envcheck.md
  --paired-env            the paired lab-vs-realistic check against another namespace (same plan
                          items, other environment): tables/envcheck_paired.*, figures/fig_envcheck.*,
                          envcheck.json / envcheck.md
  pilot pairs             a non-pre-registered pilot run that repeats plan items of a pre-registered
                          experiment with one setting changed (analysis.pilotpairs; e.g. the
                          push_legend check e3_legend vs e3_enforcement): paired McNemar + bootstrap,
                          tables/pilotpair_<exp>.*, pilot_pairs.json, a SUMMARY.md section
  SUMMARY.md              every source dir, planned vs complete per experiment and model, the
                          hypotheses, the competence gate, exclusions, provenance identity,
                          figure/table index
  report.json             machine index: input hashes, output hashes, figure statuses

Determinism: identical inputs give byte-identical outputs (sorted keys, fixed rounding, seeded
order-independent bootstraps, no timestamps). Incomplete namespaces are reported, never hidden:
claims on an incomplete experiment are `insufficient_data` unless --allow-partial (then every
affected claim carries the incompleteness in its notes and details.partial = true).

--legacy-pilot re-derives the October-2026 pilot (results/pilot, schema v1) with the corrected
estimators: arms and defenses are never pooled, OSI = 1 - RR is unclipped with Katz and
brand-cluster bootstrap intervals, and the v1 readout probe is analysed at the page-template level
(v1 computed it once per template and reused it across replicates and defenses). Those data were
seen before pre-registration v2; they are exploratory and never enter a confirmatory claim.

Exit codes: 0 report written (complete or not); 1 nothing to report or an unusable prereg.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from originblind.analysis import envcheck as EC
from originblind.analysis import estimators as est
from originblind.analysis import pilotpairs as PP
from originblind.analysis import tables as T
from originblind.analysis.audits import load_audits
from originblind.analysis.claims import evaluate_environments
from originblind.analysis.estimators import _r
from originblind.analysis.figures import FigureData, make_figures, roster_order
from originblind.analysis.load import (ENVIRONMENT_ORDER, ExperimentData, concat_episodes, environments_of,
                                       load_namespace, order_values)
from originblind.analysis.prereg import DEFAULT_PREREG, Prereg, load_prereg
from originblind.analysis.validate import validate_experiment

E5, E6 = T.E5_PARITY, T.E6_THINKING


# ====================================================================== helpers

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _r(float(o))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return sorted(o) if isinstance(o, set) else list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _clean(o):
    """NaN/inf -> JSON-safe (None / "inf"), recursively; numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return _r(float(o), 10)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, sort_keys=True, default=_json_default, ensure_ascii=False) + "\n")


def sha256_bytes(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _f2(x, nd: int = 2) -> str:
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "--"
    if math.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    return f"{xf:.{nd}f}"


def _ci(ci) -> str:
    if not ci or ci[0] is None:
        return "--"
    return f"[{_f2(ci[0])}, {_f2(ci[1])}]"


def md_table(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    cols = cols or list(df.columns)

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        if isinstance(v, float):
            return _f2(v, 3)
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(str(c).replace("|", "\\|") for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def experiment_roles(pr: Prereg) -> dict[str, str]:
    """Which pre-registered experiment plays which role (the names come from prereg.yaml)."""
    h = pr.hypotheses
    return {"e1": pr.gate.get("experiment", "e1_default"),
            "e2": (h.get("H1b") or {}).get("experiment", "e2_information"),
            "e2s": (h.get("H5b") or {}).get("experiment", "e2s_screenshot"),
            "e3": (h.get("H3") or {}).get("experiment", "e3_enforcement"),
            "e3u": (h.get("H6") or {}).get("experiment", "e3_utility"),
            "e5": E5, "e6": E6}


def is_synthetic(exps: dict[str, ExperimentData]) -> bool:
    for e in exps.values():
        if (e.manifest or {}).get("synthetic"):
            return True
        if len(e.episodes) and (e.episodes["namespace"] == "synthetic").any():
            return True
    return False


# ====================================================================== inputs

def _namespaces(ns_dirs) -> list[Path]:
    ns_list = [Path(ns_dirs)] if isinstance(ns_dirs, (str, Path)) else [Path(d) for d in ns_dirs]
    names = [d.name for d in ns_list]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"namespace directories must have distinct names (they label the sources): {dup}")
    return ns_list


def load_inputs(ns_dirs: Path | list[Path], pr: Prereg, *, audits_from: Path | None = None) -> dict:
    """Experiments, validations, per-environment completeness (environment -> experiment -> ...),
    the analysis frame and the audits of one namespace, or of several (`--include`): their
    experiment directories are merged into one analysis (keyed `<namespace>/<dir>`; the analysis
    frame keeps the first complete record per environment + experiment + episode_id, sources in the
    order given), while validation stays per experiment directory. Audits come from `audits_from`,
    else from the first namespace that has any."""
    ns_list = _namespaces(ns_dirs)
    multi = len(ns_list) > 1
    exps: dict[str, ExperimentData] = {}
    by_ns: dict[str, dict[str, ExperimentData]] = {}
    for ns in ns_list:
        loaded = load_namespace(ns)
        by_ns[str(ns)] = loaded
        for name, e in loaded.items():
            key = f"{ns.name}/{name}" if multi else name
            exps[key] = dataclasses.replace(e, name=key, episodes=e.episodes.assign(exp_dir=key)) if multi else e
    validations: dict[str, dict] = {}
    completeness: dict[str, dict] = {}
    seen_cells: dict[tuple, str] = {}
    overlaps: list[str] = []
    for name, e in exps.items():
        for env in e.environments:
            for m in sorted(set(e.episodes["model"])) if len(e.episodes) else []:
                k = (env, e.experiment, m)
                if k in seen_cells:
                    overlaps.append(f"{e.experiment} / {m} / {env} is in both {seen_cells[k]} and {name}: the first "
                                    "source's record is used per episode_id")
                else:
                    seen_cells[k] = name
        v = validate_experiment(e.path, None, prereg=pr, data=e)
        validations[name] = v
        c = completeness.setdefault(e.environment, {}).setdefault(
            e.experiment, {"complete": True, "planned": 0, "missing": 0, "by_model": {}, "exp_dirs": []})
        c["exp_dirs"].append(name)
        if v.get("complete_run") is not True:
            c["complete"] = False if v.get("complete_run") is False else (c["complete"] and None)
        for k in ("planned", "missing"):
            if isinstance(v.get(k), int):
                c[k] += v[k]
        for m, b in (v.get("by_model") or {}).items():
            c["by_model"][m] = b
    df = concat_episodes(exps)
    audits, audit_src = {}, None
    for cand in ([Path(audits_from)] if audits_from is not None else ns_list):
        a = load_audits(cand)
        if any(v is not None for v in a.values()):
            audits, audit_src = a, cand
            break
    if not audits:
        audits = load_audits(Path(audits_from) if audits_from is not None else ns_list[0])
    sources = {env: [ns.name for ns in ns_list if any(env in e.environments for e in by_ns[str(ns)].values())]
               for env in environments_of(df)}
    return {"exps": exps, "exps_by_ns": by_ns, "namespaces": ns_list, "validations": validations,
            "completeness": completeness, "df": df, "audits": audits, "audits_from": audit_src,
            "synthetic": is_synthetic(exps), "environments": environments_of(df), "sources": sources,
            "overlaps": overlaps}


# ====================================================================== tables

def h3_secondary_frame(claims: dict) -> pd.DataFrame:
    """H3's pre-registered secondary metrics of one environment's claims."""
    h3 = ((claims.get("hypotheses") or {}).get("H3") or {}).get("details") or {}
    sec = h3.get("secondary") or {}
    rows = []
    for name, v in sorted(sec.items()):
        if name == "abstention_by_defense":
            for d, e in sorted(v.items()):
                rows.append({"metric": f"abstention ({d}, deceptive)", "estimate": e.get("point"),
                             "ci_lo": (e.get("ci") or [None, None])[0], "ci_hi": (e.get("ci") or [None, None])[1],
                             "k": e.get("k"), "n": e.get("n"), "method": e.get("method")})
            continue
        if "no_secret_fill" in v:
            for part in ("no_secret_fill", "no_leak"):
                e = v[part]
                rows.append({"metric": f"{name}: {part}", "estimate": e.get("point"),
                             "ci_lo": (e.get("ci") or [None, None])[0], "ci_hi": (e.get("ci") or [None, None])[1],
                             "k": e.get("k"), "n": e.get("n"), "method": e.get("method")})
            continue
        rows.append({"metric": name, "estimate": v.get("point"), "ci_lo": (v.get("ci") or [None, None])[0],
                     "ci_hi": (v.get("ci") or [None, None])[1], "k": v.get("k"), "n": v.get("n"),
                     "method": v.get("method") or v.get("note")})
    fr = h3.get("false_refusal")
    if fr:
        rows.append({"metric": f"false refusal ({fr.get('defense')}, legitimate cells)", "estimate": fr.get("point"),
                     "ci_lo": (fr.get("ci") or [None, None])[0], "ci_hi": (fr.get("ci") or [None, None])[1],
                     "k": fr.get("k"), "n": fr.get("n"), "method": fr.get("method")})
    t = pd.DataFrame(rows, columns=["metric", "estimate", "ci_lo", "ci_hi", "k", "n", "method"])
    return t.assign(environment=claims.get("environment"))[["environment", *t.columns]] if len(t) else \
        pd.DataFrame(columns=["environment", *t.columns])


def h3_secondary_tex(t: pd.DataFrame) -> pd.DataFrame:
    if t.empty:
        return t
    return pd.DataFrame({"env.": t["environment"], "metric": t["metric"],
                         "estimate [95% CI]": [T.fmt_ci(_num(p), _num(lo), _num(hi)) for p, lo, hi in
                                               zip(t["estimate"], t["ci_lo"], t["ci_hi"])],
                         "k/n": [T.fmt_kn(k, n) if k is not None and n else "--" for k, n in zip(t["k"], t["n"])]})


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_tables(inp: dict, pr: Prereg, claims: dict, *, reps: int, seed: int) -> dict[str, tuple]:
    """name -> (csv frame, tex frame or None, LaTeX comment). `claims` is the evaluate_environments
    output: claim-derived tables are stacked over environments with an environment column."""
    df, alpha = inp["df"], pr.ci_alpha
    roles = experiment_roles(pr)
    a1, a2, dec = pr.classes("a1"), pr.classes("a2"), pr.classes("deceptive")
    groups = {"A1 pooled": a1, "A2 pooled": a2, "deceptive pooled": dec}
    out: dict[str, tuple] = {}
    by_env = claims.get("by_environment") or {}
    comp = _concat([T.competence_frame(c.get("competence_gate") or {}, env) for env, c in by_env.items()])
    gate = next(iter(by_env.values()), {}).get("competence_gate") or {}
    out["competence"] = (comp, T.competence_tex(comp), f"competence gate per environment: {gate.get('statistic')} of "
                                                       f"R_bound in {gate.get('experiment')} {gate.get('cell')}")
    cf = _concat([T.claims_frame(c, env) for env, c in by_env.items()])
    out["claims"] = (cf, T.claims_tex(cf), f"pre-registered hypotheses per environment (never pooled; prereg sha256 "
                                           f"{pr.sha256[:12]}; confirmatory environment {pr.confirmatory_environment})")
    cells = T.cells_table(df, alpha)
    out["cells"] = (cells, T.cells_tex(cells), "every cell; Wilson 95% intervals")
    rr = T.rr_table(df, alpha=alpha, reps=reps, seed=seed, groups=groups)
    out["rr"] = (rr, T.rr_tex(rr), "relative release RR = P(leak | class) / P(R_bound | exact), unclipped")
    for role, label in (("e1", "E1 framework default (A2 in T1 = placebo; raw IP = floor)"), ("e2", "E2 information")):
        sub = rr[rr["experiment"] == roles[role]] if len(rr) else rr
        out[f"rr_{role}"] = (sub, T.rr_tex(sub), label)
    e1c = T.e1_class_table(cells, rr, roles["e1"])
    out["e1_classes"] = (e1c, T.e1_class_tex(e1c), "E1 per class x model x defense x flow: leak (exact: completion), "
                                                   "RR vs exact of the same stratum; placebo = A2 in T1, floor = raw IP")
    go = T.group_outcomes_table(df, alpha)
    out["outcomes_by_group"] = (go, T.group_outcomes_tex(go), "per class group: leak, completion, abstention, "
                                                              "off-origin entry (floor = raw IP)")
    probe = T.probe_table(df, alpha)
    out["probe"] = (probe, None, "")
    kdg = T.kdg_table(df, dec, alpha, reps=reps, seed=seed)
    out["kdg"] = (kdg, T.kdg_tex(kdg), "probe validity per arm (exact keys) and KDG = P(leak | judged mismatch), "
                                       "key level")
    d3 = T.defense_table(df, roles["e3"], dec, a1, a2, alpha)
    out["defense"] = (d3, T.defense_tex(d3), "E3: completion vs deceptive leak per defense (the frontier)")
    h3s = _concat([h3_secondary_frame(c) for c in by_env.values()])
    out["h3_secondary"] = (h3s, h3_secondary_tex(h3s), "H3 pre-registered secondary metrics (pooled over "
                                                       "gate-passing models, per environment)")
    ut = T.utility_table(df, roles["e3u"], alpha)
    out["utility"] = (ut, T.utility_tex(ut), "E3-U utility: psp = processor declared in CSP")
    h5a = pr.hyp("H5a")
    ar = _concat([
        T.arm_rates_table(df, roles["e2"], pr.classes([((h5a.get("cell") or {}).get("origin_class") or "idn")]),
                          ["T1", "T1u"], "H5a", alpha, cell={k: v for k, v in (h5a.get("cell") or {}).items()}),
        T.arm_rates_table(df, roles["e2s"], a1, ["S0", "S1", "S2"], "H5b/H5c", alpha)])
    out["arm_rates"] = (ar, T.arm_rates_tex(ar), "H5 cells: leak per arm")
    par = T.parity_table(df, roles["e1"], alpha)
    out["parity"] = (par, T.parity_tex(par), "E5: real Chromium (E1 T1/D1/login) vs in-process")
    th = T.thinking_table(df, roles["e2"], alpha)
    out["thinking"] = (th, T.thinking_tex(th), "E6: thinking mode vs the same weights in E2 (unpaired: other seed)")
    term = T.terminal_table(df)
    out["terminals"] = (term, T.terminal_tex(term), "terminal types per stratum x class group")
    known = set(roles.values()) | {h.get("experiment") for h in pr.hypotheses.values()}
    for exp in sorted(set(df["experiment"]) - known) if len(df) else []:
        dt = T.descriptive_table(df, exp, alpha, pr.gate_level())
        out[f"descriptive_{exp}"] = (dt, T.descriptive_tex(dt), f"{exp}: not pre-registered; descriptive only "
                                                               f"(gate lower = {pr.gate.get('statistic')}, exact cells)")
    vf = T.validation_frame(inp["validations"])
    out["validation"] = (vf, vf.drop(columns=["models"]) if len(vf) else vf, "plan completeness and attrition")
    ex = T.exclusions_frame(inp["validations"])
    out["exclusions"] = (ex, ex, "cells with error or excluded attempts")
    expo = T.exposure_frame((inp["audits"] or {}).get("exposure"))
    out["exposure"] = (expo, _exposure_tex(expo), "RQ1 exposure audit: share of pages per pipeline x class")
    bu = T.browser_use_frame((inp["audits"] or {}).get("browser_use"))
    out["browser_use"] = (bu, bu, "RQ4: real browser-use, deterministic tool-executor runs")
    bul = T.browser_use_frame((inp["audits"] or {}).get("browser_use_llm"))
    out["browser_use_llm"] = (bul, bul, "RQ4: browser-use Agent, LLM slice (descriptive)")
    return out


def _exposure_tex(e: pd.DataFrame) -> pd.DataFrame:
    if e.empty:
        return e
    cols = {"env.": e["environment"], "pipeline": e["pipeline"],
            "default": e["default_config"].map(lambda b: "yes" if b else "no"),
            "class": e["origin_class"], "pages": e["pages"]}
    for c in ("sink_domain_present", "top_url_present", "form_action_present", "iframe_src_present"):
        if c in e.columns:
            cols[c.replace("_present", "")] = e[c].map(lambda v: _f2(v))
    cols["host form"] = e.get("top_host_form", pd.Series([""] * len(e)))
    return pd.DataFrame(cols)


def write_tables(tables: dict[str, tuple], out: Path) -> list[str]:
    written = []
    for name, (csv_df, tex_df, comment) in sorted(tables.items()):
        frame = csv_df if isinstance(csv_df, pd.DataFrame) else pd.DataFrame()
        paths = T.write_table(frame, out / name, tex=tex_df if isinstance(tex_df, pd.DataFrame) else None,
                              comment=comment)
        written += [p.name for p in paths]
    return written


# ====================================================================== claims.md / SUMMARY.md

def _claims_block(c: dict, pr: Prereg, level: str = "##") -> list[str]:
    """claims.md body for one environment's claims."""
    L: list[str] = []
    g = c.get("competence_gate") or {}
    L += [f"{level} Competence gate", "",
          f"{g.get('statistic')} of R_bound in {g.get('experiment')} {g.get('cell')} >= {g.get('threshold')}; "
          f"on fail: {g.get('on_fail')}", ""]
    gm = pd.DataFrame([{"model": m, "k/n": f"{v.get('k')}/{v.get('n')}", "p": v.get("p"), "lower": v.get("lower"),
                        "status": v.get("status")} for m, v in sorted((g.get("models") or {}).items())])
    L += [md_table(gm), ""]
    for hid, r in (c.get("hypotheses") or {}).items():
        L += [f"{level} {hid}: {r.get('name')}", "",
              f"- **status: {r.get('status')}** (decision rule alone: {r.get('decision_rule')}); family: "
              f"{r.get('family') or '--'}; kind: {r.get('kind')}; experiment: {r.get('experiment')}",
              f"- rule: `{r.get('rule_text')}`",
              f"- estimate: {_f2(r.get('estimate'), 3)} {_ci(r.get('ci'))} ({r.get('ci_method') or '--'}); n = {r.get('n')}",
              f"- p = {r.get('p') if r.get('p') is not None else '--'} ({r.get('p_source') or 'no test'}); "
              f"Holm-adjusted p = {r.get('p_holm') if r.get('p_holm') is not None else '--'}"
              + (f" (m = {r.get('holm_m')})" if r.get("holm_m") else ""),
              f"- models included: {', '.join(r.get('models_included') or []) or '--'}; excluded: "
              + (", ".join(f"{m} ({why})" for m, why in sorted((r.get('models_excluded') or {}).items())) or "--")]
        conds = r.get("conditions") or {}
        for block in ("support", "falsify"):
            for cc in conds.get(block) or []:
                L.append(f"  - {block}: `{cc.get('key')}` threshold {cc.get('threshold')} value {cc.get('value')} -> "
                         f"{cc.get('holds')}")
        for n in r.get("notes") or []:
            L.append(f"- note: {n}")
        L.append("")
    hol = c.get("holm") or {}
    if hol:
        L += [f"{level} Holm families", ""]
        for fam, v in sorted(hol.items()):
            L.append(f"- {fam}: members {v.get('members')}, tested {v.get('tested')}, untested (p = 1) "
                     f"{v.get('untested_as_p1')}, adjusted {v.get('adjusted')}")
        L.append("")
    return L


def _env_heading(env: str, pr: Prereg) -> str:
    role = "confirmatory environment" if env == pr.confirmatory_environment else "not the confirmatory environment"
    return f"Environment `{env}` ({role})"


def claims_markdown(claims: dict, pr: Prereg) -> str:
    envs = claims.get("environments") or []
    L = ["# Pre-registered claims", "",
         f"Rules: `{pr.path}` (sha256 `{pr.sha256}`, version {pr.version}, frozen: {str(pr.frozen).lower()}). "
         f"Interval level {pr.ci_level:.0%}; bootstrap: {pr.bootstrap['reps']} reps, cluster = "
         f"{pr.bootstrap['cluster']}, stratified by model, seed {pr.bootstrap['seed']}. "
         + ("Interim/partial data allowed (--allow-partial). " if claims.get("allow_partial") else "")
         + f"Claims are computed separately per stimulus environment and never pooled; the confirmatory "
           f"environment is `{pr.confirmatory_environment}`. Environments here: {', '.join(envs) or '--'}.", ""]
    if pr.amendments:
        L += ["Amendments (post-pilot, pre-freeze; what had been seen is part of each record):", ""]
        for a in pr.amendments:
            L.append(f"- {a.get('id')} ({a.get('date')}); seen: {a.get('seen')}")
            for ch in a.get("changes") or []:
                L.append(f"  - {ch}")
        L.append("")
    for env in envs:
        c = (claims.get("by_environment") or {}).get(env) or {}
        L += [f"# {_env_heading(env, pr)}", ""] if len(envs) > 1 else \
             [f"_{_env_heading(env, pr)}._", ""]
        L += _claims_block(c, pr, "##" if len(envs) == 1 else "###")
    return "\n".join(L)


def _identity_rows(inp: dict) -> pd.DataFrame:
    rows = []
    for name, v in sorted(inp["validations"].items()):
        ident = v.get("identity") or {}
        vals = ident.get("values") or {}
        e = inp["exps"][name]
        served = sorted({str(s.get("id")) for m in ([e.manifest] if e.manifest else []) + list(e.manifests)
                         for s in (m.get("served_models") or []) if isinstance(s, dict)})
        rows.append({"exp_dir": name, "config_hash": ",".join(x[:12] for x in vals.get("config_hash") or []) or "--",
                     "plan_sha256": ",".join(x[:12] for x in vals.get("plan_sha256") or []) or "--",
                     "code_sha256": ",".join(x[:12] for x in vals.get("code_sha256") or []) or "--",
                     "consistent": ident.get("consistent"), "prereg_sha256": ",".join(
                         x[:12] for x in ident.get("prereg_sha256") or []) or "--",
                     "git": ",".join(x[:10] for x in ident.get("git_sha") or []) or "--",
                     "dirty": ",".join(str(x).lower() for x in ident.get("git_dirty") or []) or "--",
                     "manifests": ident.get("n_manifests"), "served": ",".join(served) or "--"})
    return pd.DataFrame(rows)


def _gate_rows(c: dict) -> pd.DataFrame:
    g = c.get("competence_gate") or {}
    return pd.DataFrame([{"model": m, "k/n": f"{v.get('k')}/{v.get('n')}", "lower": v.get("lower"),
                          "status": v.get("status")} for m, v in sorted((g.get("models") or {}).items())])


def _hyp_rows(c: dict) -> pd.DataFrame:
    rows = []
    for hid, r in (c.get("hypotheses") or {}).items():
        rows.append({"id": hid, "family": r.get("family") or "--", "status": r.get("status"),
                     "estimate [CI]": f"{_f2(r.get('estimate'))} {_ci(r.get('ci'))}", "n": r.get("n"),
                     "p (Holm)": r.get("p_holm") if r.get("p_holm") is not None else "--",
                     "models": ",".join(r.get("models_included") or []) or "--",
                     "note": (r.get("notes") or [""])[0][:160]})
    return pd.DataFrame(rows)


def _sources_md(inp: dict, paired_env) -> list[str]:
    L = ["## Sources", ""]
    for i, d in enumerate(inp.get("namespaces") or []):
        dirs = sorted((inp.get("exps_by_ns") or {}).get(str(d), {}))
        L.append(f"- {'namespace' if i == 0 else '--include'} `{d}`: {len(dirs)} experiment dir(s)"
                 + (f" ({', '.join(dirs)})" if dirs else ""))
    a = inp.get("audits_from")
    L.append(f"- deterministic audits: `{a}`" if a is not None else "- deterministic audits: none found")
    if paired_env is not None:
        L.append(f"- --paired-env `{paired_env}` (paired environment check and awareness scan only; never pooled "
                 "into the claims)")
    for env, names in sorted((inp.get("sources") or {}).items()):
        L.append(f"- environment `{env}`: {', '.join(names)}")
    for o in inp.get("overlaps") or []:
        L.append(f"- **overlap**: {o}")
    L.append("")
    return L


def summary_markdown(ns_dir: Path, inp: dict, pr: Prereg, claims: dict, figs: list, table_files: list[str],
                     env_md: str = "", pilot_md: str = "", *, paired_env=None) -> str:
    names = [d.name for d in inp.get("namespaces") or [ns_dir]]
    L = [f"# Report: {' + '.join(f'`{n}`' for n in names)}", ""]
    if inp["synthetic"]:
        L += ["> **SYNTHETIC DATA.** This namespace was written by `originblind.analysis.synthetic` (toy outcomes). "
              "None of these numbers is a measurement; never report them.", ""]
    conf = pr.raw.get("namespace_confirmatory")
    other = [n for n in names if n != conf]
    if other:
        L += [f"> Namespace {', '.join(f'`{n}`' for n in other)} is not the pre-registered confirmatory namespace "
              f"(`{conf}`): every number below is exploratory (pilot), and the hypothesis statuses only show what the "
              "pre-registered rules would conclude on these data.", ""]
    L += _sources_md(inp, paired_env)
    envs = claims.get("environments") or []
    L += [f"Pre-registration: `{pr.path}` sha256 `{pr.sha256[:16]}`, version {pr.version}, frozen: "
          f"{str(pr.frozen).lower()}. Claims computed with allow_partial = {str(claims.get('allow_partial')).lower()}, "
          f"separately per stimulus environment (never pooled): {', '.join(envs) or '--'}; the confirmatory "
          f"environment is `{pr.confirmatory_environment}`.", ""]
    if envs and pr.confirmatory_environment not in envs:
        L += [f"> No episode here was run in the confirmatory environment (`{pr.confirmatory_environment}`).", ""]
    if pr.amendments:
        L += ["Pre-registration amendments applied: " + "; ".join(
            f"{a.get('id')} ({a.get('date')}, seen: {a.get('seen')})" for a in pr.amendments) + ".", ""]
    run_prereg = sorted({s for v in inp["validations"].values() for s in (v.get("identity") or {}).get("prereg_sha256") or []})
    if run_prereg:
        match = run_prereg == [pr.sha256]
        L += [f"Run manifests recorded prereg sha256 {', '.join(s[:16] for s in run_prereg)} -> "
              + ("matches the rules applied here." if match else "**differs from the rules applied here** (the "
                 "pre-registration changed after these data were produced)."), ""]
    # completeness
    L += ["## Planned vs complete", ""]
    rows = []
    for name, v in sorted(inp["validations"].items()):
        st = v.get("statuses") or {}
        rows.append({"exp_dir": name, "experiment": v.get("experiment"),
                     "environment": (v.get("environment") or {}).get("configured"), "planned": v.get("planned"),
                     "complete": v.get("complete"), "missing": v.get("missing"),
                     "per model (complete/planned)": "; ".join(f"{m} {b['complete']}/{b['planned']}" for m, b in
                                                               (v.get("by_model") or {}).items()) or "--",
                     "infra/harness/excluded": f"{st.get('infra_error', 0)}/{st.get('harness_error', 0)}/"
                                               f"{st.get('excluded', 0)}",
                     "complete_run": v.get("complete_run")})
    L += [md_table(pd.DataFrame(rows)), ""]
    present = {e for comp in inp["completeness"].values() for e in comp}
    absent = [e for e in sorted(set(experiment_roles(pr).values()) | {h.get("experiment") for h in pr.hypotheses.values()
                                                                       if h.get("kind") == "confirmatory"})
              if e and e not in present]
    if absent:
        L += [f"Experiments not in this namespace: {', '.join(absent)}.", ""]
    # gate + hypotheses per environment
    for env in envs:
        c = (claims.get("by_environment") or {}).get(env) or {}
        g = c.get("competence_gate") or {}
        suffix = f" -- {_env_heading(env, pr)}"
        L += [f"## Competence gate{suffix}", "",
              f"{g.get('statistic')} of exact-origin R_bound in {g.get('experiment')} {g.get('cell')} >= "
              f"{g.get('threshold')}.", "", md_table(_gate_rows(c)), ""]
        L += [f"## Hypotheses{suffix}", "", md_table(_hyp_rows(c)), ""]
    L += ["Details, conditions and per-model estimates: `claims.md` / `claims.json` (`by_environment`).", ""]
    # exclusions
    L += ["## Exclusions and attrition", ""]
    erows = []
    for name, v in sorted(inp["validations"].items()):
        ca = v.get("channel_agreement") or {}
        erows.append({"exp_dir": name, "records": v.get("n_records"), "corrupt lines": v.get("n_corrupt_lines"),
                      "duplicates": (v.get("duplicates") or {}).get("extra_complete_records"),
                      "flagged cells (> " + f"{v.get('exclusion_threshold', 0.05):.0%})": v.get("flagged_cells"),
                      "channel agreement": (f"{ca.get('agree')}/{ca.get('n')}" if ca.get("applicable") else "n/a"),
                      "problems": "; ".join(v.get("problems") or []) or "--"})
    L += [md_table(pd.DataFrame(erows)), ""]
    # provenance
    L += ["## Provenance identity", "", md_table(_identity_rows(inp)), ""]
    au = inp.get("audits") or {}
    for key in ("exposure", "browser_use", "browser_use_llm", "browser_use_differential"):
        a = au.get(key)
        if a and a.get("rows") is not None:
            L.append(f"- audit `{key}`: {a.get('path')} ({len(a['rows'])} rows, sha256 {str(a.get('sha256'))[:16]})")
        elif a and a.get("error"):
            L.append(f"- audit `{key}`: unreadable ({a.get('error')})")
    L.append("")
    if env_md:
        L += [env_md, ""]
    if pilot_md:
        L += [pilot_md, ""]
    # outputs
    L += ["## Figures", ""]
    for f in figs:
        L.append(f"- `{f.name}`: {f.status}" + (f" -- {f.note}" if f.note else ""))
    L += ["", "## Tables", "", ", ".join(f"`{t}`" for t in sorted({Path(x).stem for x in table_files})), ""]
    return "\n".join(L)


# ====================================================================== drivers

def _collect_outputs(out: Path) -> dict[str, str]:
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "report.json":
            files[str(p.relative_to(out))] = sha256_bytes(p)
    return files


AUDIT_TABLES = ("exposure", "browser_use", "browser_use_llm")  # their environment is the audit's, not the episodes'


def _env_tables(tables: dict[str, tuple], env: str) -> dict[str, pd.DataFrame]:
    """The report's tables restricted to one episode environment (audit tables and tables without
    the column pass through)."""
    out = {}
    for k, v in tables.items():
        t = v[0]
        if k not in AUDIT_TABLES and isinstance(t, pd.DataFrame) and len(t) and "environment" in t.columns:
            t = t[t["environment"] == env].reset_index(drop=True)
        out[k] = t
    return out


def build_report(ns_dir: str | Path, out_dir: str | Path | None = None, *, prereg_path: str | Path | None = None,
                 allow_partial: bool = False, reps: int | None = None, figures: bool = True,
                 paired_env: str | Path | None = None, include: list[str | Path] | tuple = (),
                 audits_dir: str | Path | None = None) -> dict:
    ns = Path(ns_dir)
    out = Path(out_dir) if out_dir else ns / "report"
    pr = load_prereg(prereg_path)
    for d in [*include, *([audits_dir] if audits_dir is not None else [])]:
        if not Path(d).is_dir():
            raise FileNotFoundError(f"{d} is not a directory")
    inp = load_inputs([ns, *map(Path, include)], pr, audits_from=Path(audits_dir) if audits_dir is not None else None)
    au = inp["audits"] or {}
    has_audit = any(a and a.get("rows") is not None for a in au.values())
    if not inp["exps"] and not has_audit:
        raise FileNotFoundError(f"no experiment directory (*/episodes.jsonl) and no audit under "
                                f"{', '.join(str(d) for d in inp['namespaces'])}")
    if paired_env is not None and not Path(paired_env).is_dir():
        raise FileNotFoundError(f"--paired-env: {paired_env} is not a directory")
    out.mkdir(parents=True, exist_ok=True)
    n_reps = int(reps or pr.bootstrap["reps"])
    seed = pr.bootstrap["seed"]
    claims = evaluate_environments(inp["df"], pr, completeness=inp["completeness"], allow_partial=allow_partial,
                                   audits=au, reps=n_reps)
    claims["namespace"] = ns.name
    claims["namespaces"] = [d.name for d in inp["namespaces"]]
    claims["synthetic"] = inp["synthetic"]
    for name, v in inp["validations"].items():
        dump_json(v, out / "validation" / f"{name}.json")
    dump_json(claims, out / "claims.json")
    (out / "claims.md").write_text(claims_markdown(claims, pr))
    tables = compute_tables(inp, pr, claims, reps=n_reps, seed=seed)
    table_files = write_tables(tables, out / "tables")

    # paired pilot checks of non-pre-registered runs against the experiment they repeat (e.g. the
    # push_legend check pv2r_e3_legend vs pv2r_e3_enforcement)
    classes = {k: pr.classes(k) for k in (pr.raw.get("classes") or {})}
    pilot_pairs = PP.paired_pilot_checks(inp["df"], inp["exps"], classes, alpha=pr.ci_alpha, reps=n_reps, seed=seed,
                                         model_order=roster_order())
    table_files += PP.write_pilotpairs(pilot_pairs, out / "tables")
    if pilot_pairs:
        dump_json({"checks": [c.to_json() for c in pilot_pairs]}, out / "pilot_pairs.json")

    # evaluation awareness (every report) and the paired environment check (--paired-env)
    if paired_env is not None:
        ec = EC.build_envcheck([*inp["namespaces"], Path(paired_env)], classes, alpha=pr.ci_alpha, reps=n_reps,
                               seed=seed, model_order=roster_order(), exps_by_ns=dict(inp["exps_by_ns"]))
        table_files += EC.write_envcheck(ec, out, figures=False)
        env_md = EC.envcheck_markdown(ec)
        env_json = ec.to_json()
    else:
        aw = EC.awareness_only(inp["exps"], classes.get("deceptive", []), alpha=pr.ci_alpha, model_order=roster_order())
        ec = EC.EnvCheck(pairs=pd.DataFrame(), paired=pd.DataFrame(), unpaired=pd.DataFrame(), identity=[],
                         awareness=aw["table"], terms=aw["terms"], examples=aw["examples"], scan_info=aw["info"],
                         sources=inp["sources"],
                         notes=["no --paired-env: verbalised-awareness scan of the report's own sources only"])
        for name, frame, tex in (("awareness", ec.awareness, EC.awareness_tex(ec.awareness)),
                                 ("awareness_terms", ec.terms, None)):
            table_files += [x.name for x in T.write_table(frame, out / "tables" / name, tex=tex,
                                                           comment=f"pattern: {EC.AWARENESS_PATTERN}")]
        env_md = EC.envcheck_markdown(ec, title="Verbalised evaluation awareness (this namespace)")
        env_json = ec.to_json()
    dump_json(env_json, out / "envcheck.json")
    (out / "envcheck.md").write_text(env_md + "\n")

    figs = []
    if figures:
        envs = claims.get("environments") or []
        for env in envs:
            c = (claims.get("by_environment") or {}).get(env) or {}
            fd = FigureData(tables=_env_tables(tables, env), claims=c, experiments=experiment_roles(pr),
                            classes=classes)
            res = make_figures(fd, out / "figures", suffix=f"_{env}" if len(envs) > 1 else "")
            figs += res
        if paired_env is not None:
            from originblind.analysis.figures import fig_envcheck

            try:
                figs.append(fig_envcheck(ec.paired, out / "figures"))
            except Exception as e:  # noqa: BLE001 - a broken figure must not lose the tables/claims
                from originblind.analysis.figures import FigureResult

                figs.append(FigureResult("fig_envcheck", "skipped", [], f"error: {type(e).__name__}: {e}"))
    (out / "SUMMARY.md").write_text(summary_markdown(ns, inp, pr, claims, figs, table_files, env_md,
                                                    PP.pilotpairs_markdown(pilot_pairs), paired_env=paired_env))
    index = {"namespace": ns.name, "mode": "v2", "synthetic": inp["synthetic"], "prereg": pr.summary(),
             "allow_partial": allow_partial, "bootstrap_reps": n_reps, "environments": claims.get("environments"),
             "sources": {"namespaces": [str(d) for d in inp["namespaces"]],
                         "audits": str(inp["audits_from"]) if inp["audits_from"] is not None else None,
                         "paired_env": str(paired_env) if paired_env is not None else None,
                         "by_environment": inp["sources"], "overlaps": inp["overlaps"]},
             "pilot_pairs": [c.name for c in pilot_pairs],
             "paired_env": Path(paired_env).name if paired_env is not None else None,
             "inputs": {name: {"episodes_sha256": e.sha256, "n_lines": e.n_lines, "corrupt": e.n_corrupt,
                               "schema_versions": e.schema_versions, "experiment": e.experiment,
                               "environment": e.environment, "complete_episodes": int(len(e.episodes))}
                        for name, e in sorted(inp["exps"].items())},
             "audits": {k: ({"path": v.get("path"), "sha256": v.get("sha256"), "rows": len(v.get("rows") or [])}
                            if v else None) for k, v in sorted(au.items())},
             "figures": [f.to_json() for f in figs], "outputs": _collect_outputs(out)}
    dump_json(index, out / "report.json")
    return index


# ====================================================================== legacy (October pilot, schema v1)

LEGACY_CORRECTIONS = [
    "arms (T0/T1/T2) and defenses are never pooled: every estimate is per (directory, model, arm, defense)",
    "OSI = 1 - RR is unclipped (an agent that leaks more often than it completes has OSI < 0) with Katz "
    "(Haldane-corrected) and brand-cluster bootstrap intervals that include the denominator's uncertainty",
    "the v1 readout probe is analysed once per page template (model, brand, flow, origin, semantics, arm): v1 "
    "computed it per template with fixed seeds and reused it across replicates and defenses, so counting it per "
    "episode double-counts templates",
    "A2 in T0/T1 is a placebo by construction (the a2_action observation is byte-identical to the exact one; "
    "a2_iframe differs only by element ids): OSI ~ 0 there is not evidence of origin blindness",
    "D3 never had check_origin in the action grammar (invocation is 0 by construction); DBU is decision-equivalent "
    "to D5 on every lab context (it cannot stand in for a deployed library)",
    "v1 'deceptive' = every non-exact class including raw IP (kept for comparability; v2 excludes raw IP)",
]


def _old_claims(ns: Path, name: str) -> dict | None:
    p = ns / name / "derived" / "claims.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except json.JSONDecodeError:
        return None


def legacy_estimates(ns: Path, df: pd.DataFrame, tables: dict[str, tuple], pr: Prereg) -> dict:
    alpha = pr.ci_alpha
    rr = tables["rr"][0]
    kdg = tables["kdg"][0]
    dfn = tables["defense"][0]
    out: dict = {"status": "exploratory: seen before pre-registration v2; never confirmatory",
                 "corrections": LEGACY_CORRECTIONS, "H1_A2_OSI": [], "H2_KDG_readout": [], "H3_D3push": [],
                 "H4_DBU": [], "competence": [], "as_originally_reported": {}}
    if len(rr):
        for _, r in rr[(rr["unit"] == "group") & (rr["origin_class"] == "A2 pooled")].iterrows():
            lo, hi, src = T.rr_ci(r)
            out["H1_A2_OSI"].append({
                "experiment": r["experiment"], "model": r["model"], "arm": r["arm"], "defense": r["defense"],
                "a2_leak": f"{int(r['leak_k'])}/{int(r['leak_n'])}",
                "exact_r_bound": f"{int(r['exact_r_bound_k'])}/{int(r['exact_r_bound_n'])}",
                "rr": r["rr"], "osi": r["osi"], "osi_katz": [1 - _num(r["rr_katz_hi"]), 1 - _num(r["rr_katz_lo"])],
                "osi_boot": [1 - _num(r["rr_boot_hi"]), 1 - _num(r["rr_boot_lo"])],
                "boot_degenerate": bool(r["boot_degenerate"]),
                "placebo": T.is_placebo("a2_action", r["arm"], r["defense"])})
    if len(kdg):
        for _, r in kdg.iterrows():
            out["H2_KDG_readout"].append({
                "experiment": r["experiment"], "model": r["model"], "arm": r["arm"], "defense": r["defense"],
                "templates_deceptive": int(r["deceptive_keys"]),
                "readout_correct": f"{int(r['deceptive_readout_correct_k'])}/{int(r['deceptive_readout_correct_n'])}",
                "kdg_readout": f"{int(r['kdg_readout_k'])}/{int(r['kdg_readout_n'])}",
                "kdg_readout_p": r["kdg_readout_p"], "wilson": [r["kdg_readout_lo"], r["kdg_readout_hi"]],
                "boot": [r["kdg_readout_boot_lo"], r["kdg_readout_boot_hi"]],
                "unit": "page template (leak averaged over the template's episodes)"})
    for exp in sorted(set(df["experiment"])):
        g = df[df["experiment"] == exp]
        for m in sorted(set(g["model"])):
            gm = g[g["model"] == m]
            nonexact = gm[gm["origin_class"] != "exact"]
            for arm in [a for a in ("T0", "T1", "T2") if a in set(gm["arm"])]:
                for d in sorted(set(gm["defense"]), key=lambda x: T._ORDERS["defense"].index(x)
                                if x in T._ORDERS["defense"] else 99):
                    ex = gm[(gm["arm"] == arm) & (gm["defense"] == d) & (gm["origin_class"] == "exact")]
                    if len(ex):
                        k, n = T._kn(ex["r_bound"])
                        w = est.wilson(k, n, alpha)
                        out["competence"].append({"experiment": exp, "model": m, "arm": arm, "defense": d,
                                                  "exact_r_bound": f"{k}/{n}", "p": k / n, "wilson": [w.lo, w.hi]})
            if {"D0", "D3push"} <= set(gm["defense"]):
                a = nonexact[nonexact["defense"] == "D3push"]
                b = nonexact[nonexact["defense"] == "D0"]
                ka, na = T._kn(a["leak"])
                kb, nb = T._kn(b["leak"])
                cp = est.clopper_pearson(ka, na, alpha)
                nc = est.newcombe_diff(kb, nb, ka, na, alpha)
                d3 = gm[gm["defense"] == "D3"]
                kt, nt = T._kn(d3["check_origin_any"]) if len(d3) else (0, 0)
                d6 = nonexact[nonexact["defense"].isin(["D6", "D6r"])]
                k6, n6 = T._kn(d6["leak"]) if len(d6) else (0, 0)
                cp6 = est.clopper_pearson(k6, n6, alpha)
                out["H3_D3push"].append({
                    "experiment": exp, "model": m, "d3push_nonexact_leak": f"{ka}/{na}", "d3push_cp": [cp.lo, cp.hi],
                    "d0_nonexact_leak": f"{kb}/{nb}", "rd_d0_minus_d3push": nc.point,
                    "rd_newcombe": [nc.lo, nc.hi], "d3_tool_invoked": f"{kt}/{nt}",
                    "d6_nonexact_leak": f"{k6}/{n6}" if n6 else None,
                    "d6_cp": [cp6.lo, cp6.hi] if n6 else None})
            if "DBU" in set(gm["defense"]):
                rec = {"experiment": exp, "model": m, "note": "DBU is decision-equivalent to D5 on every lab context "
                                                              "(by construction); not a deployed-library test"}
                for d in ("DBU", "D5"):
                    for grp, cls in (("a1", pr.classes("a1") + pr.classes("floor")), ("a2", pr.classes("a2"))):
                        s = nonexact[(nonexact["defense"] == d) & nonexact["origin_class"].isin(cls)]
                        k, n = T._kn(s["leak"])
                        rec[f"{d}_{grp}_leak"] = f"{k}/{n}"
                out["H4_DBU"].append(rec)
    if len(dfn):
        out["defense_rows"] = int(len(dfn))
    for name in sorted({e for e in df["exp_dir"]}):
        oc = _old_claims(ns, name)
        if oc is not None:
            out["as_originally_reported"][name] = oc
    return out


def legacy_markdown(ns: Path, inp: dict, est_: dict, figs: list, table_files: list[str]) -> str:
    L = [f"# Legacy pilot re-derivation: `{ns.name}`", "",
         "> **EXPLORATORY.** October-2026 pilot (schema v1, in-process driver, old harness). These data were seen "
         "before pre-registration v2 and are never pooled into, or used as, a confirmatory test.", "",
         "## Corrections applied", ""]
    L += [f"- {c}" for c in est_["corrections"]] + [""]
    L += ["## Inputs", ""]
    L += [md_table(pd.DataFrame([{"exp_dir": n, "complete": len(e.episodes), "lines": e.n_lines,
                                  "corrupt": e.n_corrupt, "schema": ",".join(map(str, e.schema_versions)),
                                  "sha256": e.sha256[:16]} for n, e in sorted(inp["exps"].items())])), ""]
    L += ["## H1 (v1): A2 OSI per directory x arm x defense (unclipped)", ""]
    h1 = pd.DataFrame(est_["H1_A2_OSI"])
    if len(h1):
        h1 = h1.assign(osi_katz=h1["osi_katz"].map(_ci), osi_boot=h1["osi_boot"].map(_ci))
    L += [md_table(h1, ["experiment", "model", "arm", "defense", "a2_leak", "exact_r_bound", "rr", "osi", "osi_katz",
                        "osi_boot", "placebo"] if len(h1) else None), ""]
    L += ["## H2 (v1): P(leak | sink read correctly), page-template level", ""]
    h2 = pd.DataFrame(est_["H2_KDG_readout"])
    if len(h2):
        h2 = h2.assign(wilson=h2["wilson"].map(_ci), boot=h2["boot"].map(_ci))
    L += [md_table(h2, ["experiment", "model", "arm", "defense", "templates_deceptive", "readout_correct",
                        "kdg_readout", "kdg_readout_p", "wilson", "boot"] if len(h2) else None), ""]
    L += ["## H3 (v1): D3push residual leak on non-exact classes", ""]
    h3 = pd.DataFrame(est_["H3_D3push"])
    if len(h3):
        h3 = h3.assign(d3push_cp=h3["d3push_cp"].map(_ci), rd_newcombe=h3["rd_newcombe"].map(_ci))
    L += [md_table(h3, ["experiment", "model", "d3push_nonexact_leak", "d3push_cp", "d0_nonexact_leak",
                        "rd_d0_minus_d3push", "rd_newcombe", "d3_tool_invoked", "d6_nonexact_leak"]
                   if len(h3) else None), ""]
    L += ["## H4 (v1): DBU vs D5", "", md_table(pd.DataFrame(est_["H4_DBU"])), ""]
    L += ["## Exact-origin completion per stratum", ""]
    comp = pd.DataFrame(est_["competence"])
    if len(comp):
        comp = comp.assign(wilson=comp["wilson"].map(_ci))
    L += [md_table(comp), ""]
    L += ["## Figures", ""] + [f"- `{f.name}`: {f.status}" + (f" -- {f.note}" if f.note else "") for f in figs]
    L += ["", "## Tables", "", ", ".join(f"`{t}`" for t in sorted({Path(x).stem for x in table_files})), "",
          "Originally reported values (the old `derived/claims.json`) are kept under `as_originally_reported` in "
          "`legacy_estimates.json` for side-by-side comparison.", ""]
    return "\n".join(L)


def build_legacy_report(ns_dir: str | Path, out_dir: str | Path | None = None, *,
                        prereg_path: str | Path | None = None, reps: int | None = None, figures: bool = True) -> dict:
    ns = Path(ns_dir)
    out = Path(out_dir) if out_dir else ns / "report_legacy"
    pr = load_prereg(prereg_path)  # class lists, alpha and bootstrap settings only
    exps = load_namespace(ns)
    if not exps:
        raise FileNotFoundError(f"no experiment directory (*/episodes.jsonl) under {ns}")
    out.mkdir(parents=True, exist_ok=True)
    validations = {name: validate_experiment(e.path, None, prereg=pr, data=e) for name, e in exps.items()}
    df = concat_episodes(exps)
    inp = {"exps": exps, "validations": validations, "df": df}
    n_reps = int(reps or pr.bootstrap["reps"])
    seed = pr.bootstrap["seed"]
    alpha = pr.ci_alpha
    a1, a2, floor = pr.classes("a1"), pr.classes("a2"), pr.classes("floor")
    nonexact = a1 + a2 + floor
    groups = {"A1 pooled": a1, "A2 pooled": a2, "deceptive pooled (v2, no raw IP)": a1 + a2,
              "non-exact pooled (v1 deceptive)": nonexact}
    tables: dict[str, tuple] = {}
    cells = T.cells_table(df, alpha)
    tables["cells"] = (cells, T.cells_tex(cells), "legacy pilot cells; Wilson 95%")
    rr = T.rr_table(df, alpha=alpha, reps=n_reps, seed=seed, groups=groups)
    tables["rr"] = (rr, T.rr_tex(rr), "legacy pilot: RR unclipped per directory x arm x defense")
    probe = T.probe_table(df, alpha, key_col="template_key")
    tables["probe"] = (probe, None, "")
    kdg = T.kdg_table(df, nonexact, alpha, key_col="template_key", reps=n_reps, seed=seed)
    tables["kdg"] = (kdg, T.kdg_tex(kdg), "legacy: v1 readout probe at the page-template level (no discrimination)")
    dfn = pd.concat([T.defense_table(df, e, nonexact, a1, a2, alpha) for e in sorted(set(df["experiment"]))],
                    ignore_index=True)
    tables["defense"] = (dfn, T.defense_tex(dfn), "legacy: deceptive = every non-exact class incl. raw IP (v1)")
    go = T.group_outcomes_table(df, alpha)
    tables["outcomes_by_group"] = (go, T.group_outcomes_tex(go), "legacy: per class group (abstention is a v1 "
                                                                 "approximation: refuse/stop without release)")
    term = T.terminal_table(df)
    tables["terminals"] = (term, T.terminal_tex(term), "legacy terminal types")
    vf = T.validation_frame(validations)
    tables["validation"] = (vf, vf.drop(columns=["models"]) if len(vf) else vf, "legacy stores (no stored plan)")
    table_files = write_tables(tables, out / "tables")
    est_ = legacy_estimates(ns, df, tables, pr)
    dump_json(est_, out / "legacy_estimates.json")
    for name, v in validations.items():
        dump_json(v, out / "validation" / f"{name}.json")
    figs = []
    if figures:
        fd = FigureData(tables={k: v[0] for k, v in tables.items()}, claims=None, legacy=True)
        figs = make_figures(fd, out / "figures")
    md = legacy_markdown(ns, inp, est_, figs, table_files)
    (out / "legacy_estimates.md").write_text(md)
    (out / "SUMMARY.md").write_text(md)
    index = {"namespace": ns.name, "mode": "legacy-pilot", "prereg": pr.summary(), "bootstrap_reps": n_reps,
             "inputs": {name: {"episodes_sha256": e.sha256, "n_lines": e.n_lines, "corrupt": e.n_corrupt,
                               "schema_versions": e.schema_versions, "complete_episodes": int(len(e.episodes))}
                        for name, e in sorted(exps.items())},
             "figures": [f.to_json() for f in figs], "outputs": _collect_outputs(out)}
    dump_json(index, out / "report.json")
    return index


# ====================================================================== CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="originblind report",
                                 description="Tables, pre-registered claims, figures and SUMMARY.md for a namespace.")
    ap.add_argument("ns_dir", help="results/<namespace> (or results/pilot with --legacy-pilot)")
    ap.add_argument("--out", default=None, help="output directory (default: <ns_dir>/report, or "
                                                "<ns_dir>/report_legacy with --legacy-pilot)")
    ap.add_argument("--prereg", default=str(DEFAULT_PREREG), help="pre-registration rules (configs/prereg.yaml)")
    ap.add_argument("--legacy-pilot", action="store_true",
                    help="re-derive the October-2026 pilot (schema v1) with the corrected estimators")
    ap.add_argument("--allow-partial", action="store_true",
                    help="evaluate claims on incomplete experiments (interim/pilot look; flagged in every claim)")
    ap.add_argument("--reps", type=int, default=None, help="bootstrap replicates (default: prereg bootstrap.reps)")
    ap.add_argument("--no-figures", action="store_true", help="skip the figures (tables and claims only)")
    ap.add_argument("--paired-env", default=None, metavar="OTHER_NS_DIR",
                    help="namespace holding the same plan items in the other stimulus environment (e.g. results/"
                         "pilot_v2r for results/pilot_v2): paired lab-vs-realistic check")
    ap.add_argument("--include", action="extend", nargs="+", default=[], metavar="NS_DIR",
                    help="further namespace dirs merged into the analysis (e.g. results/pilot_v2r_x with results/"
                         "pilot_v2r: same environment, other models); validation stays per experiment dir")
    ap.add_argument("--audits", default=None, metavar="NS_DIR",
                    help="namespace dir holding exposure/ and browser_use/ (default: the first of ns_dir and the "
                         "--include dirs that has them)")
    args = ap.parse_args(argv)
    if args.legacy_pilot and (args.include or args.audits or args.paired_env):
        print("report: --legacy-pilot takes one namespace (no --include/--audits/--paired-env)", file=sys.stderr)
        return 1
    try:
        if args.legacy_pilot:
            idx = build_legacy_report(args.ns_dir, args.out, prereg_path=args.prereg, reps=args.reps,
                                      figures=not args.no_figures)
        else:
            idx = build_report(args.ns_dir, args.out, prereg_path=args.prereg, allow_partial=args.allow_partial,
                               reps=args.reps, figures=not args.no_figures, paired_env=args.paired_env,
                               include=args.include, audits_dir=args.audits)
    except (FileNotFoundError, ValueError) as e:
        print(f"report: {e}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else Path(args.ns_dir) / ("report_legacy" if args.legacy_pilot else "report")
    brief = {"out": str(out), "mode": idx.get("mode"), "inputs": {k: v.get("complete_episodes") for k, v in
                                                                   idx.get("inputs", {}).items()},
             "figures": {f["name"]: f["status"] for f in idx.get("figures", [])}, "outputs": len(idx.get("outputs", {}))}
    if idx.get("mode") == "v2":
        claims = json.loads((out / "claims.json").read_text())
        brief["hypotheses"] = {env: {h: r.get("status") for h, r in (c.get("hypotheses") or {}).items()}
                               for env, c in (claims.get("by_environment") or {}).items()}
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
