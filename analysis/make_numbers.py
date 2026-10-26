"""Emit outputs/generated/numbers.tex: one \\defval{key}{text} per number used in the paper.

Sources: the regenerated report and descriptive analyses; browser-use quantities use explicitly
retained aggregate fixtures because the dedicated reproduction and detailed native records are withheld.
Formatting conventions (applied everywhere):
  rates           -> percent, integer unless 0 < rate < 10% (one decimal); intervals in brackets, same precision
  ratios (RR)     -> two decimals; differences of rates -> signed percentage points ("pp")
  counts          -> k/n with thousands separators; p-values -> "p<0.001" or two significant digits
The script fails if any required value is missing and records the sha256 of claims.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict

from common import (A1_E2, A1_E3, A2, DEFENSES_E3, EXPECTED, GEN, MODELS, REPORT, SHORT, THINK, fmt_dec, fmt_dec_ci, fmt_int,
                    fmt_p, fmt_pct, fmt_pct_ci, fmt_pp, fmt_pp_ci, hyp, num, pct_places, point_ci_places, read_derived, table, wilson)

V: dict[str, str] = {}


def put(key: str, text) -> None:
    if key in V:
        raise KeyError(f"duplicate key {key}")
    if text is None or text == "":
        raise ValueError(f"empty value for {key}")
    V[key] = str(text)


def rate(key: str, k: int, n: int, ci: tuple[float, float] | None = None) -> None:
    """k/n with percent and a 95% interval (Wilson unless given)."""
    k, n = int(k), int(n)
    if n <= 0:
        raise ValueError(f"n = 0 for {key}")
    p = k / n
    lo, hi = ci if ci is not None else wilson(k, n)
    places, cplaces = point_ci_places(p, lo, hi)
    put(f"{key}.k", fmt_int(k))
    put(f"{key}.n", fmt_int(n))
    put(f"{key}.kn", f"{fmt_int(k)}/{fmt_int(n)}")
    put(f"{key}.pct", fmt_pct(p, places))
    put(f"{key}.ci", fmt_pct_ci(lo, hi, cplaces))
    put(f"{key}.full", f"{fmt_pct(p, places)} {fmt_pct_ci(lo, hi, cplaces)}")
    put(f"{key}.knpct", f"{fmt_int(k)}/{fmt_int(n)} ({fmt_pct(p, places)})")


def prop(key: str, p: float, lo: float, hi: float) -> None:
    """A proportion that is not a simple k/n (pooled means): percent with its interval."""
    places, cplaces = point_ci_places(p, lo, hi)
    put(f"{key}.pct", fmt_pct(p, places))
    put(f"{key}.ci", fmt_pct_ci(lo, hi, cplaces))
    put(f"{key}.full", f"{fmt_pct(p, places)} {fmt_pct_ci(lo, hi, cplaces)}")


def ratio(key: str, rr: float, lo: float, hi: float) -> None:
    put(f"{key}.rr", fmt_dec(rr))
    put(f"{key}.rrci", fmt_dec_ci(lo, hi))
    put(f"{key}.rrfull", f"{fmt_dec(rr)} {fmt_dec_ci(lo, hi)}")


def diff(key: str, d: float, lo: float, hi: float) -> None:
    places = 1 if abs(100 * d) < 9.95 and d != 0 else 0
    put(f"{key}.pp", fmt_pp(d, places))
    put(f"{key}.ci", fmt_pp_ci(lo, hi, places))
    put(f"{key}.full", f"{fmt_pp(d, places)}~pp {fmt_pp_ci(lo, hi, places)}")


def rows(name: str, **flt):
    out = [r for r in table(name) if all(r.get(k) == v for k, v in flt.items())]
    return out


def one(name: str, **flt) -> dict:
    out = rows(name, **flt)
    if len(out) != 1:
        raise LookupError(f"{name} {flt}: {len(out)} rows")
    return out[0]


# design
design = {r["experiment"]: r for r in read_derived("design")}
put("n.total", fmt_int(int(design["all"]["episodes"])))
for exp, key in (("e1_default", "e1"), ("e2_information", "e2"), ("e2s_screenshot", "e2s"), ("e3_enforcement", "e3"),
                 ("e3_utility", "e3u"), ("e5_parity", "e5"), ("e6_thinking", "e6")):
    put(f"n.{key}", fmt_int(int(design[exp]["episodes"])))
put("n.models", "seven")
put("n.modelsnum", str(len(MODELS)))
put("n.brands", str(int(design["e3_enforcement"]["brands"])))
put("n.handlerepair", fmt_int(sum(int(design[e]["handle_repair_episodes"]) for e in design if e != "all")))

# competence gate
put("comp.min.k", str(min(int(r["k"]) for r in rows("competence") if r["status"] != "not_evaluable")))
for r in rows("competence"):
    if r["status"] == "not_evaluable":
        continue
    m = SHORT[r["model"]]
    put(f"comp.{m}.kn", f"{r['k']}/{r['n']}")
    put(f"comp.{m}.k", f"{r['k']}")
    put(f"comp.{m}.lo", fmt_dec(num(r["lower"])))

# E2 per arm and group
E2_GROUPS = {"exact": ["exact"], "a1": A1_E2, "iframe": ["a2_iframe"], "action": ["a2_action"], "a2": A2,
             "dec": A1_E2 + A2, "typo": ["typosquat"], "combo": ["combosquat"], "idn": ["idn"]}
cells = table("cells")
for m in MODELS:
    for arm in ("T0", "T1", "T1u", "T2"):
        sub = [r for r in cells if r["experiment"] == "e2_information" and r["model"] == m and r["arm"] == arm]
        for g, classes in E2_GROUPS.items():
            cs = [r for r in sub if r["origin_class"] in classes]
            if not cs:
                continue
            what = "r_bound" if g == "exact" else "leak"
            rate(f"e2.{arm}.{SHORT[m]}.{g}", sum(int(r[f"{what}_k"]) for r in cs), sum(int(r[f"{what}_n"]) for r in cs))
            if g != "exact":
                rate(f"e2.{arm}.{SHORT[m]}.{g}.abst", sum(int(r["abstained_k"]) for r in cs),
                     sum(int(r["abstained_n"]) for r in cs))
                rate(f"e2.{arm}.{SHORT[m]}.{g}.selfcorr", sum(int(r["self_corrected_k"]) for r in cs),
                     sum(int(r["self_corrected_n"]) for r in cs))

# release ratios with union intervals (binding plane)
for r in read_derived("binding_plane"):
    exp = "e2" if r["experiment"] == "e2_information" else "e3"
    cond = r["arm"] if exp == "e2" else r["defense"]
    ratio(f"rr.{exp}.{cond}.{SHORT[r['model']]}.{r['group']}", float(r["rr"]), float(r["lo"]), float(r["hi"]))

# E3 per defense
for r in rows("defense"):
    m, d = SHORT[r["model"]], r["defense"]
    for g, col in (("a1", "a1_leak"), ("a2", "a2_leak"), ("dec", "deceptive_leak"), ("exact", "exact_r_bound"),
                   ("abst", "deceptive_abstained"), ("tool", "check_origin_any"), ("shown", "deceptive_d3push_shown"),
                   ("deny", "deceptive_broker_deny"), ("exactfr", "exact_false_refusal")):
        rate(f"e3.{d}.{m}.{g}", r[f"{col}_k"], r[f"{col}_n"])
for d in DEFENSES_E3:
    rs = rows("defense", defense=d)
    for g, col in (("a1", "a1_leak"), ("a2", "a2_leak"), ("dec", "deceptive_leak"), ("exact", "exact_r_bound"),
                   ("abst", "deceptive_abstained"), ("tool", "check_origin_any")):
        rate(f"e3.{d}.all.{g}", sum(int(r[f"{col}_k"]) for r in rs), sum(int(r[f"{col}_n"]) for r in rs))
# per-class E3 cells (for the iframe/action split under the pushed verdict)
for m in MODELS:
    for d in DEFENSES_E3:
        sub = [r for r in cells if r["experiment"] == "e3_enforcement" and r["model"] == m and r["defense"] == d]
        for g, classes in (("iframe", ["a2_iframe"]), ("action", ["a2_action"]), ("subdomain", ["subdomain"])):
            cs = [r for r in sub if r["origin_class"] in classes]
            rate(f"e3.{d}.{SHORT[m]}.{g}", sum(int(r["leak_k"]) for r in cs), sum(int(r["leak_n"]) for r in cs))

# H3 (knowing-doing)
h2 = hyp("H2")
prop("kdg.pooled", h2["estimate"], h2["ci"][0], h2["ci"][1])
put("kdg.pooled.p", fmt_p(h2["p_holm"]))
put("kdg.pooled.n", fmt_int(h2["n"]))
put("kdg.models", str(len(h2["models_included"])))
for r in read_derived("coupling"):
    m = SHORT[r["model"]]
    rate(f"kdg.{m}", r["kdg_k"], r["kdg_n"])
    rate(f"kdg.{m}.valid", r["exact_false_mismatch"], r["exact_n"])
    rate(f"kdg.{m}.jm", r["dec_jm"], r["dec_n"])
    rate(f"kdg.{m}.uncond", r["uncond_k"], r["uncond_n"])
    rate(f"kdg.{m}.recog", r["dec_recognized"], r["dec_n"])
    if r["nojm_n"] not in ("", "0"):
        rate(f"kdg.{m}.nojm", r["nojm_k"], r["nojm_n"])
cpl = [r for r in read_derived("coupling") if r["valid"] == "True"]
put("kdg.coupling.maxabs", fmt_pp(max(abs(float(r["kdg_minus_uncond"])) for r in cpl), 1).lstrip("+"))
loo = read_derived("loo")
kl = [float(r["value"]) for r in loo if r["what"] == "kdg_pooled" and r["left_out"] != "none"]
put("kdg.loo.min", fmt_pct(min(kl), 0))
put("kdg.loo.max", fmt_pct(max(kl), 0))
lobo = read_derived("lobo")
kb = [float(r["kdg_pooled"]) for r in lobo if r["left_out_brand"] != "none"]
put("kdg.lobo.min", fmt_pct(min(kb), 0))
put("kdg.lobo.max", fmt_pct(max(kb), 0))

# H4 (verdict residual)
h3 = hyp("H3")
pm = h3["details"]["per_model"]["qwen3.5-9b"]["D3push"]
lk = h3["details"]["D3push_leak"]["per_model"]["qwen3.5-9b"]
rate("h4", lk["k"], lk["n"], ci=tuple(pm["ci95"]))
put("h4.lower", fmt_pct(pm["one_sided_bounds"][0], 0))
put("h4.lowerdec", fmt_dec(pm["one_sided_bounds"][0]))
put("h4.p", fmt_p(h3["p_holm"]))
put("h4.alpha", fmt_dec(0.05 / 7, 4))
het = h3["details"]["heterogeneity"]["D3push"]["p"]
put("h4.fh.p", r"$p<10^{-5}$" if het < 1e-5 else fmt_p(het))
eff = {r["model"]: r for r in read_derived("effective_n")}
put("h4.lower.distinct", fmt_pct(float(eff["qwen3.5-9b"]["lower_distinct"]), 0))
put("h4.distinct", eff["qwen3.5-9b"]["distinct"])
put("h4.concordant", eff["qwen3.5-9b"]["concordant"])
lb = [r for r in lobo if r["left_out_brand"] != "none"]
put("h4.lobo.min", fmt_pct(min(float(r["q9b_push_p"]) for r in lb), 0))
put("h4.lobo.max", fmt_pct(max(float(r["q9b_push_p"]) for r in lb), 0))
put("h4.lobo.lowermin", fmt_pct(min(float(r["q9b_push_lower_alpha7"]) for r in lb), 0))
others = [m for m in MODELS if m != "qwen3.5-9b"]
put("h4.others.maxpct", fmt_pct(max(int(h3["details"]["D3push_leak"]["per_model"][m]["k"]) /
                                    int(h3["details"]["D3push_leak"]["per_model"][m]["n"]) for m in others)))
put("h4.others.maxlower", fmt_pct(max(h3["details"]["per_model"][m]["D3push"]["one_sided_bounds"][0] for m in others), 1))
d6 = h3["details"]["D6r_leak"]
rate("d6.all", d6["k"], d6["n"])
d3p = h3["details"]["D3push_leak"]
rate("d3push.all", d3p["k"], d3p["n"])

# defense-ladder secondary
for r in table("h3_secondary"):
    met = r["metric"]
    est, lo, hi = float(r["estimate"]), float(r["ci_lo"]), float(r["ci_hi"])
    if met in ("D2_vs_D1", "D3info_vs_D3push"):
        diff(f"sec.{met}", est, lo, hi)
    elif met == "D3_invocation_rate":
        rate("sec.toolrate", r["k"], r["n"], ci=(lo, hi))
    elif met.startswith("D3push_compliance_given_DENY"):
        key = "sec.deny." + met.split(": ")[1].replace("_", "")
        prop(key, est, lo, hi)  # equal-weight mean over models, as reported
        put(f"{key}.kn", f"{fmt_int(int(r['k']))}/{fmt_int(int(r['n']))}")
for m in MODELS:
    sec = h3["details"]["secondary"]["D3push_compliance_given_DENY"]
    pm2 = sec.get("per_model", {}).get(m) if isinstance(sec, dict) else None
    if pm2:
        for kk, vv in pm2.items():
            if isinstance(vv, dict) and "k" in vv and "n" in vv:
                rate(f"sec.deny.{SHORT[m]}.{kk.replace('_', '')}", vv["k"], vv["n"])

# pooled release ratio
h1b = hyp("H1b")
put("rrpool.val", fmt_dec(h1b["estimate"]))
put("rrpool.ci", fmt_dec_ci(h1b["ci"][0], h1b["ci"][1]))
put("rrpool.full", f"{fmt_dec(h1b['estimate'])} {fmt_dec_ci(h1b['ci'][0], h1b['ci'][1])}")
put("rrpool.p", fmt_p(h1b["p_holm"]))
rl = [float(r["value"]) for r in loo if r["what"] == "rr_pooled" and r["left_out"] != "none"]
put("rrpool.loo.min", fmt_dec(min(rl)))
put("rrpool.loo.max", fmt_dec(max(rl)))

# representation effects
for hid, key in (("H5a", "idn"), ("H5b", "omnibox"), ("H5c", "fakebar")):
    h = hyp(hid)
    diff(f"rep.{key}", h["estimate"], h["ci"][0], h["ci"][1])
    put(f"rep.{key}.p", fmt_p(h["p_holm"]))
    for m, e in h["details"]["per_model"].items():
        diff(f"rep.{key}.{SHORT[m]}", e["rd"], e["newcombe"][0], e["newcombe"][1])
        rate(f"rep.{key}.{SHORT[m]}.a", e["a_k"], e["a_n"])
        rate(f"rep.{key}.{SHORT[m]}.b", e["b_k"], e["b_n"])

# utility (H6 measurement)
h6 = hyp("H6")
d1 = h6["details"]["D1"]
prop("h6", h6["estimate"], h6["ci"][0], h6["ci"][1])
for r in read_derived("utility"):
    m = "all" if r["model"] == "all" else SHORT[r["model"]]
    rate(f"util.{r['flow']}.{r['defense']}.{m}", r["success_k"], r["success_n"])
fr_rows = rows("utility", flow="psp", defense="D1")
rate("h6.fr", sum(int(r["false_refusal_k"]) for r in fr_rows), sum(int(r["false_refusal_n"]) for r in fr_rows))
fr3 = h3["details"]["false_refusal"]
rate("d3push.falserefusal", fr3["k"], fr3["n"])

# contrasts
for r in read_derived("contrasts"):
    m = "all" if r["model"] == "all" else SHORT[r["model"]]
    diff(f"con.{r['name']}.{m}", float(r["d"]), float(r["lo"]), float(r["hi"]))
    rate(f"con.{r['name']}.{m}.a", r["k1"], r["n1"])
    rate(f"con.{r['name']}.{m}.b", r["k2"], r["n2"])

# parity, tool, overrides
par = {r["model"]: r for r in read_derived("parity")}
a = par["all"]
put("par.pairs", fmt_int(int(a["pairs"])))
rate("par.agree", a["agree"], a["pairs"])
put("par.onlya", a["only_chromium"])
put("par.onlyb", a["only_inprocess"])
put("par.p", fmt_p(float(a["mcnemar_p"])) if float(a["mcnemar_p"]) < 1 else r"$p=1.0$")
rate("par.chromium", a["chromium_k"], a["pairs"])
rate("par.inprocess", a["inprocess_k"], a["pairs"])
for r in read_derived("tool_deny"):
    rate(f"tool.{SHORT[r['model']]}.{r['group']}.{r['what']}", r["leaked"], r["episodes"]) if int(r["episodes"]) else \
        put(f"tool.{SHORT[r['model']]}.{r['group']}.{r['what']}.kn", "0/0")
for r in read_derived("overrides"):
    put(f"ovr.{SHORT[r['model']]}.{r['group']}.leaked", r["leaked"])

# E1 calibration by flow
for r in read_derived("e1_flows"):
    m = SHORT[r["model"]]
    for g in ("exact", "a1", "rawip", "a2", "a1_abst", "a1_selfcorr"):
        rate(f"e1.{r['defense']}.{r['flow']}.{m}.{g.replace('_', '')}", r[f"{g}_k"], r[f"{g}_n"])

# text vs pixels
for r in read_derived("modality"):
    m = SHORT[r["model"]]
    rate(f"mod.{m}.{r['arm']}.a1", r["a1_k"], r["a1_n"])
    rate(f"mod.{m}.{r['arm']}.exact", r["exact_k"], r["exact_n"])
    rate(f"mod.{m}.{r['arm']}.action", r["action_k"], r["action_n"])

# awareness
aw = [r for r in rows("awareness") if r["experiment"] == "(all)"]
for r in aw:
    rate(f"aware.{SHORT[r['model']]}", r["aware_k"], r["aware_n"])
base = [r for r in aw if r["model"] in MODELS]
put("aware.max", fmt_pct(max(int(r["aware_k"]) / int(r["aware_n"]) for r in base), 1))

# thinking
for r in rows("thinking"):
    m = SHORT[r["model"]]
    rate(f"think.{m}.{r['arm']}.{r['origin_class']}", r["think_k"], r["think_n"])
    rate(f"think.{m}.{r['arm']}.{r['origin_class']}.ref", r["ref_k"], r["ref_n"])
og = rows("outcomes_by_group", experiment="e6_thinking")
for r in og:
    if r["class_group"] in ("a1", "a2"):
        rate(f"think.{SHORT[r['model']]}.{r['arm']}.{r['class_group']}", r["leak_k"], r["leak_n"])

# audits
expo = table("exposure")
put("expo.dumps", str(sum(int(r["pages"]) for r in expo)))
put("expo.pipelines", str(len({r["pipeline"] for r in expo})))
put("expo.defaults", str(len({r["pipeline"] for r in expo if r["default_config"] == "True" and
                              not r["pipeline"].startswith("ours")})))
put("expo.frameworkpipelines", str(len({r["pipeline"] for r in expo if not r["pipeline"].startswith("ours")})))
bu = table("browser_use")
put("bu.rows", str(len(bu)))
put("bu.domainrows", str(sum(1 for r in bu if r["sensitive_data_format"] == "domain")))
bul = table("browser_use_llm")
put("bu.llm.episodes", str(sum(int(r["n"]) for r in bul)))
put("bu.llm.permodel", str(sum(int(r["n"]) for r in bul if r["model"] == "gemma-4-12b")))
retained_counts = json.loads((EXPECTED / "browser_use_counts.json").read_text())
put("bu.diffdomain", str(retained_counts["bu.diffdomain"]))
put("bu.difflook", str(retained_counts["bu.difflook"]))
for m in ("gemma-4-12b", "qwen3.5-9b"):
    sub = [r for r in bul if r["model"] == m]
    ifr = [r for r in sub if r["origin_class"] == "a2_iframe" and r["cross_origin_iframes_effective"] == "True"]
    put(f"bu.llm.{SHORT[m]}.iframeleak", f"{sum(int(r['leak']) for r in ifr)}/{sum(int(r['n']) for r in ifr)}")
    a2l = [r for r in sub if r["origin_class"] in A2]
    put(f"bu.llm.{SHORT[m]}.a2leak", f"{sum(int(r['leak']) for r in a2l if r['cross_origin_iframes_effective']=='True' or r['origin_class']=='a2_action')}/"
        f"{sum(int(r['n']) for r in a2l if r['cross_origin_iframes_effective']=='True' or r['origin_class']=='a2_action')}")
    a1 = [r for r in sub if r["origin_class"] in A1_E3]
    put(f"bu.llm.{SHORT[m]}.a1leak", f"{sum(int(r['leak']) for r in a1)}/{sum(int(r['n']) for r in a1)}")

# ranges over models
def rng(key: str, values: list[float]) -> None:
    lo, hi = min(values), max(values)
    put(f"{key}.min", fmt_pct(lo, 0))
    put(f"{key}.max", fmt_pct(hi, 0))
    put(f"{key}.range", f"{fmt_pct(lo, 0, False)}--{fmt_pct(hi, 0)}")


def e2rate(m, arm, classes, what="leak"):
    cs = [r for r in cells if r["experiment"] == "e2_information" and r["model"] == m and r["arm"] == arm
          and r["origin_class"] in classes]
    return sum(int(r[f"{what}_k"]) for r in cs) / sum(int(r[f"{what}_n"]) for r in cs)


UNBOUND = ["qwen3.5-4b", "qwen3.5-9b", "gemma-4-12b", "ministral-3-8b"]
rng("range.e2T1.a2", [e2rate(m, "T1", A2) for m in MODELS])
rng("range.e2T2.unbound.iframe", [e2rate(m, "T2", ["a2_iframe"]) for m in UNBOUND])
rng("range.e2T2.unbound.a1", [e2rate(m, "T2", A1_E2) for m in UNBOUND])
rng("range.e2T2.releasing.iframe", [e2rate(m, "T2", ["a2_iframe"]) for m in MODELS if m != "ministral-3-14b"])
rng("range.e2T2.exact", [e2rate(m, "T2", ["exact"], "r_bound") for m in MODELS])


def e2k(m, arm, classes, what="leak"):
    cs = [r for r in cells if r["experiment"] == "e2_information" and r["model"] == m and r["arm"] == arm
          and r["origin_class"] in classes]
    return sum(int(r[f"{what}_k"]) for r in cs), sum(int(r[f"{what}_n"]) for r in cs)


# largest change in injected-receiver release from URL to URL+receiver among the models other than Ministral-3-14B
shifts = [(abs(e2k(m, "T2", A2)[0] - e2k(m, "T1", A2)[0]), e2k(m, "T1", A2)[1]) for m in MODELS if m != "ministral-3-14b"]
if len({n for _, n in shifts}) != 1:
    raise ValueError(f"unequal injected-receiver cells: {shifts}")
put("shift.e2a2.others.kofn", f"{max(k for k, _ in shifts)} of {shifts[0][1]}")

# observation identity: which deceptive login pages each arm leaves identical to the legitimate page
oid = defaultdict(dict)
for r in read_derived("obs_identity"):
    oid[r["brand"]][(r["arm"], r["origin_class"])] = r["identical_to_legit"] == "True"
LOOK = ("typosquat", "combosquat", "idn")
ladder = [b for b, d in oid.items()
          if all(d[("T0", c)] for c in LOOK + tuple(A2)) and all(d[("T1", c)] for c in A2)
          and not any(d[("T1", c)] for c in LOOK) and not any(d[("T2", c)] for c in LOOK + tuple(A2))]
if len(ladder) != len(oid):
    raise ValueError(f"observation ladder holds in {len(ladder)} of {len(oid)} brands; the text says all")
put("obsid.ladder.brands", str(len(ladder)))
cpl_all = {r["model"]: r for r in read_derived("coupling")}
rng("range.kdg.unbound", [int(cpl_all[m]["kdg_k"]) / int(cpl_all[m]["kdg_n"]) for m in UNBOUND])
rng("range.kdg.jm", [int(r["dec_jm"]) / int(r["dec_n"]) for r in cpl])
rng("range.kdg.recog", [int(r["dec_recognized"]) / int(r["dec_n"]) for r in cpl])
d5 = {r["model"]: r for r in rows("defense", defense="D5r")}
rng("range.d5r.a2", [int(d5[m]["a2_leak_k"]) / int(d5[m]["a2_leak_n"]) for m in MODELS])
rng("range.d3push.others", [int(r["deceptive_leak_k"]) / int(r["deceptive_leak_n"])
                            for r in rows("defense", defense="D3push") if r["model"] != "qwen3.5-9b"])
d1 = {r["model"]: r for r in rows("defense", defense="D1")}
rng("range.d1.unbound.dec", [int(d1[m]["deceptive_leak_k"]) / int(d1[m]["deceptive_leak_n"]) for m in UNBOUND])
fac = {r["model"]: r for r in rows("defense", defense="D3info")}
rng("range.d3info.unbound.dec", [int(fac[m]["deceptive_leak_k"]) / int(fac[m]["deceptive_leak_n"]) for m in UNBOUND])
ann = {r["model"]: r for r in rows("defense", defense="D2")}
rng("range.d2.unbound.dec", [int(ann[m]["deceptive_leak_k"]) / int(ann[m]["deceptive_leak_n"]) for m in UNBOUND])
put("bu.diffrows", str(retained_counts["bu.diffrows"]))

# probe per class (E2, URL+receiver)
for r in rows("probe", experiment="e2_information", arm="T2"):
    m, c = SHORT[r["model"]], r["origin_class"]
    rate(f"probe.{m}.{c}.jm", r["judged_mismatch_k"], r["judged_mismatch_n"])
    rate(f"probe.{m}.{c}.readout", r["readout_correct_k"], r["readout_correct_n"])
    if int(r["leak_given_mismatch_n"]):
        rate(f"probe.{m}.{c}.lgm", r["leak_given_mismatch_k"], r["leak_given_mismatch_n"])

# calibration and coupling summaries
e1f = read_derived("e1_flows")
vals = [int(r["a1_k"]) / int(r["a1_n"]) for r in e1f if r["defense"] == "D1" and r["flow"] in ("login", "otp", "checkout")
        and r["model"] in UNBOUND]
rng("range.e1D1.unbound.a1", vals)
nj = [cpl_all[m] for m in UNBOUND]
k_nojm = sum(int(r["nojm_k"]) for r in nj); n_nojm = sum(int(r["nojm_n"]) for r in nj)
k_jm = sum(int(r["kdg_k"]) for r in nj); n_jm = sum(int(r["kdg_n"]) for r in nj)
rate("coupling.unbound.nojm", k_nojm, n_nojm)
rate("coupling.unbound.jm", k_jm, n_jm)

# how episodes end (E3, utility)
FLOW_METRICS = {"attempted": "attempted", "denied_eps": "denied", "substituted": "subst", "released": "released",
                "completed": "completed", "abstained": "abst", "step_limit": "steplimit", "parse_fail": "parsefail",
                "egress_eps": "egress"}
for r in read_derived("outcome_flow"):
    m = "all" if r["model"] == "all" else SHORT[r["model"]]
    n = int(r["n"])
    if r["experiment"] == "e3_enforcement":
        if r["defense"] not in ("D1", "D3push", "D5r", "D6r"):
            continue
        base = f"flow.{r['defense']}.{m}.{r['group']}"
        wanted = FLOW_METRICS if r["group"] == "dec" else {"denied_eps": "denied", "completed": "completed"}
    else:
        if m != "all" or r["defense"] not in ("D3push", "D5r", "D6r"):
            continue
        base = f"flow.util.{r['group'].replace('_', '')}.{r['defense']}"
        wanted = {"denied_eps": "denied", "completed": "completed", "abstained": "abst", "step_limit": "steplimit"}
    for col, short in wanted.items():
        rate(f"{base}.{short}", r[col], n)
    put(f"{base}.deniedfills", fmt_int(int(r["denied_fills"])))

# substitution into the attacker's field versus network release (A2 pages)
gaps = []
for r in read_derived("exposure"):
    m = "all" if r["model"] == "all" else SHORT[r["model"]]
    oc = {"a2_iframe": "iframe", "a2_action": "action"}[r["origin_class"]]
    cond = r["arm"] if r["experiment"] == "e2_information" else r["defense"]
    exp = "e2" if r["experiment"] == "e2_information" else "e3"
    rate(f"sub.{exp}.{cond}.{m}.{oc}", r["substituted"], r["n"])
    if m == "all":
        rate(f"subrel.{exp}.{cond}.all.{oc}", r["released"], r["n"])
        if oc == "iframe" and cond not in ("D3push",):
            gaps.append((int(r["substituted_not_released"]), int(r["n"])))
    if int(r["released_not_substituted"]):
        raise ValueError(f"release without substitution: {r}")
put("sub.othergap.iframe.max", str(max(g[0] for g in gaps)))
put("sub.othergap.iframe.n", fmt_int(gaps[0][1]))

# handle repairs and the estimates without the repaired episodes
reps = read_derived("repairs")
put("hrep.total", fmt_int(sum(int(r["repaired"]) for r in reps)))
for exp, key in (("e1_default", "e1"), ("e2_information", "e2"), ("e2s_screenshot", "e2s"), ("e3_enforcement", "e3"),
                 ("e3_utility", "e3u")):
    put(f"hrep.{key}", fmt_int(sum(int(r["repaired"]) for r in reps if r["experiment"] == exp)))
put("hrep.codecard", fmt_int(sum(int(r["repaired_otp"]) + int(r["repaired_checkout"]) + int(r["repaired_psp"]) +
                                 int(r["repaired_psp_nocsp"]) for r in reps)))
put("hrep.gates", fmt_int(sum(int(r["repaired_gates"]) for r in reps)))
repaired_models = sorted({r["model"] for r in reps if int(r["repaired"])})
put("hrep.models", ", ".join(SHORT[m] for m in repaired_models))
for r in read_derived("repair_sensitivity"):
    if r["estimate"] == "h3_pooled":
        put("hrep.h3.pct", fmt_pct(float(r["kept_value"]), 0))
    else:
        key = {"h4_q9b_push": "hrep.h4", "d6r_release": "hrep.d6", "util_psp_d6r": "hrep.utilpsp",
               "util_nocsp_d6r": "hrep.utilnocsp", "m14b_e1_otp": "hrep.m14b.otp",
               "m14b_e1_checkout": "hrep.m14b.card", "m14b_util_psp_d3push": "hrep.m14b.verdict"}[r["estimate"]]
        rate(key, r["kept_k"], r["kept_n"])
        if r.get("repaired_hits") not in (None, ""):
            put(f"{key}.repairedhits", r["repaired_hits"])
            rate(f"{key}.all", r["all_k"], r["all_n"])

# resampling sensitivity: one brand draw shared by all models
jb = read_derived("joint_bootstrap")[0]
put("kdg.joint.ci", fmt_pct_ci(float(jb["lo"]), float(jb["hi"]), 0))
for r in read_derived("kdg_variants"):
    put(f"kdg.var.{r['variant'].replace('_', '')}", fmt_pct(float(r["value"]), 0))

# browser-use's own agent on lookalike pages
for r in read_derived("bu_native"):
    m = SHORT[r["model"]]
    rate(f"bunat.{m}.submitted", r["submitted"], r["lookalike_episodes"])
    rate(f"bunat.{m}.invented", r["invented_password"], r["lookalike_episodes"])

# accounting: both capture channels, egress guard
acct = read_derived("accounting")
rate("acct.twochannel", sum(int(r["two_channel_agree"]) for r in acct), sum(int(r["two_channel"]) for r in acct))
rate("acct.egress", sum(int(r["egress_eps"]) for r in acct), sum(int(r["episodes"]) for r in acct))

# exposure audit: composition of the dumps
expo_rows = table("exposure")
pages = lambda rs: sum(int(r["pages"]) for r in rs)  # noqa: E731
put("expo.framework", str(pages([r for r in expo_rows if r["framework_family"] != "ours"])))
put("expo.ours", str(pages([r for r in expo_rows if r["framework_family"] == "ours"])))
a2def = [r for r in expo_rows if r["framework_family"] != "ours" and r["default_config"] == "True" and r["origin_class"] in A2]
put("expo.a2defaults", str(pages(a2def)))
put("expo.a2defaults.recv", str(round(sum(float(r["sink_domain_present"]) * int(r["pages"]) for r in a2def))))
put("expo.defaultpipelines", str(len({r["pipeline"] for r in expo_rows if r["framework_family"] != "ours" and
                                      r["default_config"] == "True"})))

# parity per model
par_m = [r for r in read_derived("parity") if r["model"] != "all"]
for r in par_m:
    rate(f"par.{SHORT[r['model']]}.agree", r["agree"], r["pairs"])
rng("par.agree", [int(r["agree"]) / int(r["pairs"]) for r in par_m])

# fill-step keyword counts under the pushed verdict
for r in read_derived("overrides"):
    m = SHORT[r["model"]]
    for k in ("verdict", "task", "url", "delegation", "aware"):
        put(f"ovr.{m}.{r['group']}.f{k}", r[f"fill_{k}"])

# write
sha = hashlib.sha256((REPORT / "claims.json").read_bytes()).hexdigest()
GEN.mkdir(parents=True, exist_ok=True)
lines = [f"% Generated by scripts/make_numbers.py from the regenerated report and retained browser-use aggregates (claims.json sha256 {sha}).",
         "% Do not edit by hand. Regenerate with `./reproduce.sh`."]
lines += [f"\\defval{{{k}}}{{{v}}}" for k, v in sorted(V.items())]
(GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
(GEN / "numbers.json").write_text(json.dumps(V, indent=0, sort_keys=True, ensure_ascii=False))
print(f"wrote {len(V)} values; claims.json sha256 {sha[:16]}")

provenance = {
    "generated_claims_sha256": sha,
    "withheld_browser_use_raw_verification": "not_run",
    "key_modes": {k: "retained_only" if k in retained_counts else
                  "aggregate" if k.startswith(("bu.", "bunat.")) else "raw" for k in sorted(V)},
}
(GEN / "numbers_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
