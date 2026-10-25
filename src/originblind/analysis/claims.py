"""Pre-registered claims, evaluated EXACTLY as written in configs/prereg.yaml.

Nothing here hard-codes a threshold, class list, cell or contrast: each hypothesis function reads
its block from the pre-registration, computes the named quantities, and a generic rule engine
evaluates the `support` / `falsify` blocks key by key (`<quantity>_<ge|le|gt|lt>`,
`ci_excludes_zero`, `tost_margin`, a boolean `<flag>: true|false`, `any_model: {<sub-block>}` =
the sub-block holds for at least one gate-passing model, and the modifier
`multiplicity: bonferroni_over_models`). Every result records the block it applied, each
condition's value, and the prereg sha256.

Inference (identical for every hypothesis):
  competence gate  per model, `competence_gate.statistic` (wilson_lower_<level>) of R_bound in the
                   gate cell >= threshold; gate-failed or gate-unevaluable models never enter a
                   pooled test and are reported separately (details.per_model).
  pooled estimate  equal-weight mean over gate-passing models of the per-model estimate
                   (`estimands.pooled`).
  intervals        `interval_rule: union_cluster_bootstrap_and_episode_level` (amendment v2.1):
                   every interval used in a decision is the UNION (min lower, max upper) of
                     (a) the brand-cluster bootstrap interval, stratified by model (`bootstrap:`
                         reps/seed, percentile), and
                     (b) the episode-level interval of the SAME estimand (episodes independent):
                         rates: Wilson at Kish's effective n (= Wilson on the counts for one model,
                         on the pooled counts when every model has the same n), Clopper-Pearson
                         when the estimate is 0 or 1 and for broker cells (enforcement outcomes of
                         gate arms, zero-event by design);
                         RR: Katz (Haldane-corrected) for one model, MOVER combination of the
                         per-model Katz intervals for the equal-weight mean;
                         RD: Newcombe hybrid score (MOVER-Wilson for the equal-weight mean).
                   One-sided / Bonferroni levels apply to both components (a one-sided 1 - a bound
                   is the matching limit of both two-sided 1 - 2a intervals); the TOST (1 - 2 alpha)
                   interval is a union too. A degenerate bootstrap (every replicate equal, e.g.
                   zero events) contributes a point, so the episode-level interval binds. Both
                   components and which one binds each limit are recorded (`interval`).
                   Without the rule (pre-amendment prereg): the bootstrap interval, exact/Katz on
                   pooled counts only when it is degenerate.
  p-values         the hypothesis' `test:` line if present (one-sided bootstrap p); otherwise the
                   p-value of the support block's interval condition(s), obtained by inverting the
                   same interval procedure (one-sided conditions get one-sided p-values; max over
                   conditions: intersection-union); an `any_model` condition gets the smallest
                   per-model p times the number of gate-passing models (Bonferroni). Under the
                   union rule every p-value is the larger of the bootstrap and the episode-level
                   p-value (the union excludes the null iff both components do).
  unit: model      (H3 since amendment v2.1) the decision is made per gate-passing model: per-model
                   rates, one-sided bounds at alpha / M under Bonferroni over the M models (union
                   rule at that level); `pooled_*` quantities are the equal-weight means;
                   heterogeneity across models is an exact Freeman-Halton test.
  Holm             within each declared `holm_families` family; a member that produced no p-value
                   (not run, insufficient data, interim look) enters as p = 1, so the family size is
                   the pre-registered one and never shrinks with the data (conservative).
                   A hypothesis is `supported` only if its support block holds AND it survives Holm;
                   `falsified` follows the falsify block alone.
  completeness     an incomplete experiment (planned episodes missing) yields `insufficient_data`
                   (estimates kept under details, flagged partial) unless allow_partial is set.
  environment      claims are evaluated separately per stimulus environment (lab | realistic) by
                   `evaluate_environments`; `evaluate` refuses a frame that mixes environments, so no
                   estimate, gate decision or Holm family ever pools them. The confirmatory claims are
                   those of the pre-registration's confirmatory environment.

Statuses: supported | falsified | inconclusive | not_run (experiment absent) | insufficient_data.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from originblind.analysis import estimators as est
from originblind.analysis.estimators import _r
from originblind.analysis.prereg import UNION_RULE, Prereg

STATUSES = ("supported", "falsified", "inconclusive", "not_run", "insufficient_data")
NAN = float("nan")
_COND_RE = re.compile(r"^(?P<q>.+)_(?P<op>ge|le|gt|lt)$")
_CI_COND_RE = re.compile(r"^(?:(?P<b>.+?)_)?ci_(?P<bound>lo|hi)_(?P<op>gt|lt|ge|le)$")
_OPS: dict[str, Callable[[float, float], bool]] = {
    "ge": lambda v, t: v >= t, "le": lambda v, t: v <= t, "gt": lambda v, t: v > t, "lt": lambda v, t: v < t}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _gate_defense(defense: str) -> bool:
    """Enforcement arms (a broker gate): their leak is zero-event by design."""
    try:
        from originblind.defense.defenses import SPECS

        return SPECS[defense].gate is not None
    except KeyError:
        return False


# ====================================================================== context

@dataclass
class Context:
    df: pd.DataFrame
    pr: Prereg
    completeness: dict[str, dict] = field(default_factory=dict)  # experiment -> {complete, planned, missing}
    allow_partial: bool = False
    audits: dict = field(default_factory=dict)  # {"exposure": {...rows...}, "browser_use": {...}}
    reps: int | None = None  # override the prereg bootstrap reps (tests only)
    gate: dict | None = None

    @property
    def alpha(self) -> float:
        return self.pr.ci_alpha

    @property
    def n_reps(self) -> int:
        return int(self.reps or self.pr.bootstrap["reps"])

    @property
    def union(self) -> bool:
        """prereg v2.1: every confirmatory interval is the union of the cluster-bootstrap and the
        episode-level interval."""
        return self.pr.interval_rule == UNION_RULE

    def bootstrap(self, rows: pd.DataFrame, stat, label: str, strata: str | None = "model",
                  alpha: float | None = None) -> est.BootResult:
        b = self.pr.bootstrap
        return est.cluster_bootstrap(rows, stat, cluster=b["cluster"], strata=strata, reps=self.n_reps,
                                     seed=est.derive_seed(b["seed"], label),
                                     alpha=self.alpha if alpha is None else alpha)

    def gate_status(self, model: str) -> str:
        return ((self.gate or {}).get("models", {}).get(model) or {}).get("status", "not_evaluable")

    def split_models(self, considered: list[str], usable: Callable[[str], bool] | None = None,
                     ) -> tuple[list[str], dict[str, str]]:
        inc, exc = [], {}
        for m in considered:
            g = self.gate_status(m)
            if g != "pass":
                exc[m] = f"competence gate: {g}"
            elif usable is not None and not usable(m):
                exc[m] = "no data in a required cell"
            else:
                inc.append(m)
        return inc, exc

    def incomplete(self, experiments: list[str]) -> list[str]:
        out = []
        for e in experiments:
            c = self.completeness.get(e) or {}
            if c.get("complete") is False:
                pending = [f"{m} {b.get('status', '').replace('_', ' ')} ({b.get('complete')}/{b.get('planned')})"
                           for m, b in sorted((c.get("by_model") or {}).items()) if b.get("status") != "complete"]
                out.append(f"{e} incomplete: {c.get('missing')} of {c.get('planned')} planned episodes missing"
                           + (f" [{'; '.join(pending)}]" if pending else ""))
        return out


def select(df: pd.DataFrame, experiment: str | None = None, **cell) -> pd.DataFrame:
    if df.empty:
        return df
    m = np.ones(len(df), dtype=bool)
    if experiment is not None:
        m &= (df["experiment"] == experiment).to_numpy()
    for k, v in cell.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple, set)):
            m &= df[k].isin(list(v)).to_numpy()
        else:
            m &= (df[k] == v).to_numpy()
    return df[m]


def has_experiment(df: pd.DataFrame, experiment: str) -> bool:
    return bool(len(df)) and bool((df["experiment"] == experiment).any())


def _counts(rows: pd.DataFrame, col: str) -> tuple[int, int]:
    s = pd.to_numeric(rows[col], errors="coerce").dropna() if len(rows) else pd.Series(dtype=float)
    return int(round(float(s.sum()))), int(len(s))


def _model_mask(v: est.WeightedView, m: str) -> np.ndarray:
    return (v.df["model"] == m).to_numpy()


# ====================================================================== pooled estimators

@dataclass
class Estimate:
    point: float
    lo: float
    hi: float
    method: str
    k: int = 0
    n: int = 0
    per_model: dict = field(default_factory=dict)
    boot: est.BootResult | None = None
    pfun: Callable[[float, str], float] | None = None
    extra: dict = field(default_factory=dict)
    interval: dict | None = None  # union rule: both component intervals, which one binds each limit

    def to_json(self) -> dict:
        d = {"point": _r(self.point), "ci": [_r(self.lo), _r(self.hi)], "method": self.method,
             "k": self.k, "n": self.n, "per_model": self.per_model}
        if self.boot is not None:
            d["bootstrap"] = self.boot.to_json()
        if self.interval is not None:
            d["interval"] = self.interval
        d.update(self.extra)
        return d


def _max_pfun(*fs: Callable[[float, str], float] | None) -> Callable[[float, str], float]:
    """p-value consistent with the union of intervals: the union excludes the null iff every
    component does, i.e. at the largest of the component p-values (intersection-union). An
    undefined component p makes the combined p undefined (never smaller)."""
    parts = [f for f in fs if f is not None]

    def f(null: float, side: str) -> float:
        ps = [g(null, side) for g in parts]
        if not ps or any(_missing(p) for p in ps):
            return NAN
        return float(max(ps))
    return f


def _binding(b: float, e: float, lower: bool) -> str:
    if _missing(b) or _missing(e):
        return "undefined"
    if b == e:
        return "both"
    return ("episode_level" if e < b else "bootstrap") if lower else ("episode_level" if e > b else "bootstrap")


def union_components(br: est.BootResult, boot_ci: tuple[float, float], ep_ci: tuple[float, float],
                     ep_method: str, alpha: float, **ep_extra) -> tuple[float, float, dict]:
    """(lo, hi, record) of the v2.1 interval rule: the union of the brand-cluster bootstrap interval
    and the episode-level interval at the same two-sided level `alpha`."""
    lo, hi = est.union_interval(boot_ci, ep_ci)
    rec = {"rule": UNION_RULE, "level": _r(1 - alpha, 10),
           "bootstrap": {"ci": [_r(boot_ci[0]), _r(boot_ci[1])], "degenerate": bool(br.degenerate),
                         "n_clusters": br.n_clusters, "n_reps": br.n_reps, "n_undefined": br.n_undefined,
                         "method": "brand-cluster bootstrap stratified by model (percentile)"},
           "episode_level": {"ci": [_r(ep_ci[0]), _r(ep_ci[1])], "method": ep_method,
                             **{k: _r(v) if isinstance(v, float) else v for k, v in ep_extra.items()}},
           "binding": {"lo": _binding(boot_ci[0], ep_ci[0], True), "hi": _binding(boot_ci[1], ep_ci[1], False)}}
    return lo, hi, rec


def _union_method(rec: dict) -> str:
    return (f"union of brand-cluster bootstrap [{_r(rec['bootstrap']['ci'][0], 4)}, {_r(rec['bootstrap']['ci'][1], 4)}]"
            + (" (degenerate)" if rec["bootstrap"]["degenerate"] else "")
            + f" and episode-level {rec['episode_level']['method']} [{_r(rec['episode_level']['ci'][0], 4)}, "
              f"{_r(rec['episode_level']['ci'][1], 4)}] (prereg v2.1 interval rule)")


def _inversion_p(ci_at: Callable[[float], tuple[float, float]], null: float, side: str) -> float:
    """p-value by inverting a two-sided interval procedure. For a one-sided side ('greater' /
    'less') the smallest two-sided alpha whose interval lies beyond the null is twice the
    one-sided p, so it is halved: the same scale as a one-sided bootstrap p-value."""
    p = est.ci_inversion_p(ci_at, null, side)
    return p if side == "two" else min(1.0, p / 2.0)


def _cp_pfun(k: int, n: int) -> Callable[[float, str], float]:
    def f(null: float, side: str) -> float:
        def ci(a):
            iv = est.clopper_pearson(k, n, a)
            return iv.lo, iv.hi
        return _inversion_p(ci, null, side)
    return f


def _boot_pfun(br: est.BootResult) -> Callable[[float, str], float]:
    return lambda null, side: br.p_value(null, side)


def _binmean_pfun(per: list[tuple[int, int]], exact: bool) -> Callable[[float, str], float]:
    def f(null: float, side: str) -> float:
        def ci(a):
            iv = est.binomial_mean_interval(per, a, exact=exact)
            return iv.lo, iv.hi
        return _inversion_p(ci, null, side)
    return f


def pooled_rate(ctx: Context, rows: pd.DataFrame, col: str, models: list[str], label: str, *,
                exact_ci: bool = False, alpha: float | None = None) -> Estimate:
    """Equal-weight mean over `models` of the per-model mean of a 0/1 column (one model = that
    model's rate). `alpha` overrides the two-sided interval level (e.g. 2 alpha / M for a one-sided
    Bonferroni bound); it applies to both components of the union.

    Interval (prereg v2.1 `interval_rule`): the union of the brand-cluster bootstrap interval
    (stratified by model) and the episode-level interval of the same equal-weight mean
    (est.binomial_mean_interval: Wilson, Clopper-Pearson when the estimate is 0 or 1, and for
    broker cells, `exact_ci`, always). Without the rule (pre-amendment prereg): the bootstrap, or
    Clopper-Pearson on the pooled counts when it is degenerate or the cell is a broker cell."""
    alpha = ctx.alpha if alpha is None else alpha
    per = {}
    for m in models:
        k, n = _counts(rows[rows["model"] == m], col)
        w = est.wilson(k, n, alpha)
        per[m] = {"k": k, "n": n, "p": _r(k / n if n else NAN), "wilson": [_r(w.lo), _r(w.hi)]}
    models = [m for m in models if per[m]["n"] > 0]
    k = sum(per[m]["k"] for m in models)
    n = sum(per[m]["n"] for m in models)
    if not models:
        return Estimate(NAN, NAN, NAN, "no data", 0, 0, per)
    kn = [(per[m]["k"], per[m]["n"]) for m in models]
    if exact_ci and not ctx.union:
        cp = est.clopper_pearson(k, n, alpha)
        return Estimate(k / n, cp.lo, cp.hi, "Clopper-Pearson on pooled counts (broker / zero-event cell)",
                        k, n, per, None, _cp_pfun(k, n))
    sub = rows[rows["model"].isin(models)]

    def stat(v: est.WeightedView):
        return np.mean([v.mean(col, _model_mask(v, m)) for m in models], axis=0)

    br = ctx.bootstrap(sub, stat, label, alpha=alpha)
    if ctx.union:
        ep = est.binomial_mean_interval(kn, alpha, exact=exact_ci)
        lo, hi, rec = union_components(br, (br.lo, br.hi), (ep.lo, ep.hi), ep.method
                                       + (" (broker cell)" if exact_ci else "")
                                       + (" on pooled counts" if len(models) > 1 and len({x[1] for x in kn}) == 1
                                          else (" at Kish effective n" if len(models) > 1 else "")),
                                       alpha, k_eff=ep.k_eff, n_eff=ep.n_eff)
        point = br.point if not _missing(br.point) else ep.point
        return Estimate(point, lo, hi, _union_method(rec), k, n, per, br,
                        _max_pfun(_boot_pfun(br), _binmean_pfun(kn, exact_ci)), interval=rec)
    if br.degenerate:
        cp = est.clopper_pearson(k, n, alpha)
        return Estimate(br.point, cp.lo, cp.hi, "Clopper-Pearson on pooled counts (bootstrap degenerate)",
                        k, n, per, br, _cp_pfun(k, n))
    return Estimate(br.point, br.lo, br.hi, "brand-cluster bootstrap stratified by model (percentile)",
                    k, n, per, br, _boot_pfun(br))


def pooled_rr(ctx: Context, num: pd.DataFrame, den: pd.DataFrame, models: list[str], label: str) -> Estimate:
    """Equal-weight mean over models of RR_m = P(leak | num rows) / P(R_bound | den rows)."""
    per = {}
    for m in models:
        k1, n1 = _counts(num[num["model"] == m], "leak")
        k0, n0 = _counts(den[den["model"] == m], "r_bound")
        iv = est.katz_rr_ci(k1, n1, k0, n0, ctx.alpha)
        per[m] = {"leak_k": k1, "leak_n": n1, "r_bound_k": k0, "r_bound_n": n0,
                  "rr": _r(est.rr_point(k1, n1, k0, n0)), "katz": [_r(iv.lo), _r(iv.hi)]}
    models = [m for m in models if per[m]["leak_n"] and per[m]["r_bound_n"]]
    if not models:
        return Estimate(NAN, NAN, NAN, "no data", 0, 0, per)
    k1 = sum(per[m]["leak_k"] for m in models)
    n1 = sum(per[m]["leak_n"] for m in models)
    k0 = sum(per[m]["r_bound_k"] for m in models)
    n0 = sum(per[m]["r_bound_n"] for m in models)
    num_ids = set(num.index)
    rows = pd.concat([num[num["model"].isin(models)].assign(_num=1.0),
                      den[den["model"].isin(models) & ~den.index.isin(num_ids)].assign(_num=0.0)])

    def stat(v: est.WeightedView):
        is_num = v.df["_num"].to_numpy() == 1.0
        return np.mean([est.safe_ratio(v.mean("leak", is_num & _model_mask(v, m)),
                                       v.mean("r_bound", ~is_num & _model_mask(v, m))) for m in models], axis=0)

    br = ctx.bootstrap(rows, stat, label)
    if ctx.union:
        quad = [(per[m]["leak_k"], per[m]["leak_n"], per[m]["r_bound_k"], per[m]["r_bound_n"]) for m in models]
        ep = est.mover_katz_mean(quad, ctx.alpha)

        def ep_pf(null: float, side: str) -> float:
            return _inversion_p(lambda a: (lambda iv: (iv.lo, iv.hi))(est.mover_katz_mean(quad, a)), null, side)
        lo, hi, rec = union_components(br, (br.lo, br.hi), (ep.lo, ep.hi),
                                       "Katz (Haldane-corrected)" if len(models) == 1 else
                                       "MOVER combination of per-model Katz (Haldane-corrected) intervals",
                                       ctx.alpha)
        point = br.point if not _missing(br.point) else ep.point
        return Estimate(point, lo, hi, _union_method(rec), k1 + k0, n1 + n0, per, br,
                        _max_pfun(_boot_pfun(br), ep_pf), interval=rec)
    if br.degenerate:
        kz = est.katz_rr_ci(k1, n1, k0, n0, ctx.alpha)

        def kpf(null: float, side: str) -> float:
            return _inversion_p(lambda a: (lambda iv: (iv.lo, iv.hi))(est.katz_rr_ci(k1, n1, k0, n0, a)),
                                null, side)
        return Estimate(br.point, kz.lo, kz.hi, "Katz-Haldane on pooled counts (bootstrap degenerate)",
                        k1 + k0, n1 + n0, per, br, kpf)
    return Estimate(br.point, br.lo, br.hi, "brand-cluster bootstrap stratified by model (percentile)",
                    k1 + k0, n1 + n0, per, br, _boot_pfun(br))


def pooled_rd(ctx: Context, rows_a: pd.DataFrame, rows_b: pd.DataFrame, models: list[str], col: str,
              label: str) -> Estimate:
    """Equal-weight mean over models of P(col | a) - P(col | b): Newcombe/MOVER-Wilson interval,
    (1 - 2 alpha) interval for TOST, bootstrap interval as a sensitivity analysis."""
    per, terms_in = {}, []
    for m in models:
        ka, na = _counts(rows_a[rows_a["model"] == m], col)
        kb, nb = _counts(rows_b[rows_b["model"] == m], col)
        entry = {"a_k": ka, "a_n": na, "b_k": kb, "b_n": nb}
        if na and nb:
            nc = est.newcombe_diff(ka, na, kb, nb, ctx.alpha)
            entry.update({"rd": _r(nc.point), "newcombe": [_r(nc.lo), _r(nc.hi)]})
        per[m] = entry
    models = [m for m in models if per[m]["a_n"] and per[m]["b_n"]]
    if not models:
        return Estimate(NAN, NAN, NAN, "no data", 0, 0, per)
    terms_in = [(per[m]["a_k"], per[m]["a_n"], per[m]["b_k"], per[m]["b_n"]) for m in models]
    terms = est.pooled_rd_terms(terms_in)
    iv = est.mover_wilson(terms, ctx.alpha)
    iv90 = est.mover_wilson(terms, 2 * ctx.alpha)

    def pfun(null: float, side: str) -> float:
        return _inversion_p(lambda a: (lambda x: (x.lo, x.hi))(est.mover_wilson(terms, a)), null, side)

    both = pd.concat([rows_a[rows_a["model"].isin(models)].assign(_a=1.0),
                      rows_b[rows_b["model"].isin(models)].assign(_a=0.0)])

    def stat(v: est.WeightedView):
        a = v.df["_a"].to_numpy() == 1.0
        return np.mean([v.mean(col, a & _model_mask(v, m)) - v.mean(col, ~a & _model_mask(v, m))
                        for m in models], axis=0)

    br = ctx.bootstrap(both, stat, label + "::boot")
    n = sum(t[1] + t[3] for t in terms_in)
    k = sum(t[0] + t[2] for t in terms_in)
    method = "Newcombe hybrid score" if len(models) == 1 else "MOVER-Wilson (Newcombe) for the equal-weight mean"
    if ctx.union:
        # both the 1 - alpha interval and the 1 - 2 alpha TOST interval are unions
        lo, hi, rec = union_components(br, (br.lo, br.hi), (iv.lo, iv.hi), method, ctx.alpha)
        b90 = est.percentile_interval(br.reps, 2 * ctx.alpha)
        lo90, hi90, rec90 = union_components(br, b90, (iv90.lo, iv90.hi), method, 2 * ctx.alpha)
        rec["tost_interval"] = {k_: rec90[k_] for k_ in ("level", "bootstrap", "episode_level", "binding")}
        return Estimate(iv.point, lo, hi, _union_method(rec), k, n, per, br, _max_pfun(pfun, _boot_pfun(br)),
                        {"ci90": [_r(lo90), _r(hi90)], "ci90_lo": lo90, "ci90_hi": hi90,
                         "bootstrap_ci": [_r(br.lo), _r(br.hi)]}, interval=rec)
    return Estimate(iv.point, iv.lo, iv.hi, method, k, n, per, br, pfun,
                    {"ci90": [_r(iv90.lo), _r(iv90.hi)], "ci90_lo": iv90.lo, "ci90_hi": iv90.hi,
                     "bootstrap_ci": [_r(br.lo), _r(br.hi)]})


# ====================================================================== rule engine

MODIFIERS = ("multiplicity",)  # block keys that qualify how quantities were computed (not conditions)
SPECIAL_KEYS = ("ci_excludes_zero", "tost_margin")  # conditions with their own semantics
PER_MODEL = "__per_model__"  # q[PER_MODEL] = {model: {quantity: value}} for `any_model` conditions


def evaluate_block(block: dict | None, q: dict) -> tuple[bool | None, list[dict]]:
    """Evaluate a support/falsify block. True/False when decidable, None if a needed quantity
    is missing. Conditions are AND-ed; one definite False decides the block."""
    evs: list[dict] = []
    for key, thr in (block or {}).items():
        if key in MODIFIERS:
            evs.append({"key": key, "threshold": thr, "value": q.get(f"__{key}__"), "holds": True, "modifier": True})
            continue
        if key == "any_model":
            per = q.get(PER_MODEL) or {}
            res = {m: evaluate_block(thr if isinstance(thr, dict) else {}, qm) for m, qm in sorted(per.items())}
            verdicts = {m: r[0] for m, r in res.items()}
            holds = (True if any(v is True for v in verdicts.values()) else
                     (False if verdicts and all(v is False for v in verdicts.values()) else None))
            evs.append({"key": key, "threshold": thr, "value": {m: v for m, v in verdicts.items()}, "holds": holds,
                        "models_meeting": sorted(m for m, v in verdicts.items() if v is True),
                        "per_model": {m: r[1] for m, r in res.items()}})
            continue
        if isinstance(thr, bool) and key not in SPECIAL_KEYS and not _COND_RE.match(key):
            v = q.get(key)
            evs.append({"key": key, "threshold": thr, "value": v, "holds": None if v is None else bool(v) == thr})
            continue
        if key == "ci_excludes_zero":
            lo, hi = q.get("ci_lo"), q.get("ci_hi")
            holds = None if (_missing(lo) or _missing(hi)) else (((lo > 0) or (hi < 0)) == bool(thr))
            evs.append({"key": key, "threshold": thr, "value": [_r(lo), _r(hi)], "holds": holds})
        elif key == "tost_margin":
            lo, hi = q.get("ci90_lo"), q.get("ci90_hi")
            holds = None if (_missing(lo) or _missing(hi)) else (lo > -float(thr) and hi < float(thr))
            evs.append({"key": key, "threshold": thr, "value": [_r(lo), _r(hi)], "holds": holds})
        else:
            m = _COND_RE.match(key)
            if not m:
                evs.append({"key": key, "threshold": thr, "value": None, "holds": None,
                            "note": "unrecognised condition"})
                continue
            v = q.get(m.group("q"))
            holds = None if _missing(v) else bool(_OPS[m.group("op")](float(v), float(thr)))
            evs.append({"key": key, "threshold": thr, "value": _r(v), "holds": holds})
    conds = [e for e in evs if not e.get("modifier")]
    if not conds:
        return None, evs
    if any(e["holds"] is False for e in conds):
        return False, evs
    if all(e["holds"] is True for e in conds):
        return True, evs
    return None, evs


def decide(h: dict, q: dict) -> tuple[str, dict]:
    sup, sev = evaluate_block(h.get("support"), q)
    for e in sev:  # `no_model_supports` (falsify) = the support block's any_model condition is definitely False
        if e["key"] == "any_model":
            q = q | {"no_model_supports": None if e["holds"] is None else (e["holds"] is False)}
    fal, fev = evaluate_block(h.get("falsify"), q)
    if sup is True:
        d = "supported"
    elif fal is True:
        d = "falsified"
    elif sup is None and fal is None:
        d = "insufficient_data"
    else:
        d = "inconclusive"
    return d, {"support": sev, "falsify": fev}


def holm_pvalue(pr: Prereg, hid: str, h: dict, pfuns: dict[str, Callable], main: str
                ) -> tuple[float | None, str | None]:
    """The p-value entered into Holm: the `test:` line if any, else the support block's interval
    condition(s) (intersection-union: the max)."""
    t = pr.test_null(hid)
    if t is not None and main in pfuns:
        op, null = t
        side = "greater" if op in ("<=", "<") else "less"
        return pfuns[main](null, side), f"test: {h.get('test')}"
    ps, srcs = [], []
    for key, thr in (h.get("support") or {}).items():
        if key == "any_model" and isinstance(thr, dict):
            per = pfuns.get(PER_MODEL) or {}
            for ck, cthr in thr.items():
                m = _CI_COND_RE.match(ck)
                if not m or not per:
                    continue
                base = m.group("b") or main
                side = "greater" if m.group("bound") == "lo" else "less"
                pm = {mod: f[base](float(cthr), side) for mod, f in sorted(per.items()) if base in f}
                if not pm:
                    continue
                mult = (h.get("support") or {}).get("multiplicity")
                k = len(pm) if mult == "bonferroni_over_models" else 1
                ps.append(min(1.0, k * min(pm.values())))
                srcs.append(f"any_model {ck}={cthr}: min over {len(pm)} models of the one-sided p"
                            + (f" x {k} (Bonferroni)" if k > 1 else ""))
            continue
        if key == "ci_excludes_zero" and thr and main in pfuns:
            ps.append(pfuns[main](0.0, "two"))
            srcs.append("two-sided interval excludes 0")
            continue
        m = _CI_COND_RE.match(key)
        if m:
            base = m.group("b") or main
            if base not in pfuns:
                continue
            side = "greater" if m.group("bound") == "lo" else "less"
            ps.append(pfuns[base](float(thr), side))
            srcs.append(f"{key}={thr} (interval inversion)")
    if not ps:
        return None, None
    return float(max(ps)), "; ".join(srcs)


def rule_text(h: dict) -> str:
    keep = {k: v for k, v in h.items() if k not in ("name", "rq", "kind", "family")}
    return yaml.safe_dump(keep, default_flow_style=True, sort_keys=False, width=10_000).strip()


def _base(pr: Prereg, hid: str) -> dict:
    h = pr.hyp(hid)
    return {"id": hid, "name": h.get("name"), "rq": h.get("rq"), "kind": h.get("kind"),
            "family": pr.family_of(hid), "experiment": h.get("experiment"), "rule": h, "rule_text": rule_text(h),
            "status": "not_run", "decision_rule": None, "estimate": None, "ci": None, "ci_level": pr.ci_level,
            "ci_method": None, "interval": None, "n": 0, "models_included": [], "models_excluded": {}, "p": None,
            "p_source": None, "p_holm": None, "holm_reject": None, "conditions": None, "quantities": {},
            "notes": [], "details": {}}


def _not_run(res: dict, why: str) -> dict:
    res["status"] = "not_run"
    res["notes"].append(why)
    return res


def _insufficient(res: dict, why: str) -> dict:
    res["status"] = "insufficient_data"
    res["notes"].append(why)
    return res


def _finalize(ctx: Context, res: dict, hid: str, q: dict, pfuns: dict, main: str,
              experiments: list[str]) -> dict:
    h = ctx.pr.hyp(hid)
    d, conds = decide(h, q)
    res["decision_rule"] = d
    res["conditions"] = conds
    res["quantities"] = {k: _r(v) for k, v in q.items() if not isinstance(v, (list, dict))}
    if d != "insufficient_data":
        res["p"], res["p_source"] = holm_pvalue(ctx.pr, hid, h, pfuns, main)
        if res["p"] is not None:
            res["p"] = _r(res["p"], 10)
        if res["p_source"] and ctx.union:
            res["p_source"] += " [interval rule: the larger of the bootstrap and the episode-level p]"
    reasons = ctx.incomplete(experiments)
    if reasons and not ctx.allow_partial:
        res["details"]["partial"] = True
        res["p"] = None  # an interim look never enters Holm
        return _insufficient(res, "; ".join(reasons) + " (re-run/resume, or report --allow-partial)")
    if reasons:
        res["details"]["partial"] = True
        res["notes"] += reasons
    res["status"] = d
    return res


def _set_estimate(res: dict, e: Estimate, n: int | None = None) -> None:
    res["estimate"] = _r(e.point)
    res["ci"] = [_r(e.lo), _r(e.hi)]
    res["ci_method"] = e.method
    res["interval"] = e.interval  # union rule: both component intervals (claims.json)
    res["n"] = int(n if n is not None else e.n)


# ====================================================================== competence gate

def competence_gate(df: pd.DataFrame, pr: Prereg, completeness: dict | None = None) -> dict:
    g = pr.gate
    exp = g.get("experiment")
    cell = dict(g.get("cell") or {})
    level = pr.gate_level()
    thr = float(g.get("threshold"))
    sub = select(df, exp, **cell) if has_experiment(df, exp) else df.iloc[0:0]
    out = {"experiment": exp, "cell": cell, "statistic": g.get("statistic"), "level": level, "threshold": thr,
           "on_fail": g.get("on_fail"), "evaluable": bool(len(sub)), "models": {},
           "partial": bool((completeness or {}).get(exp, {}).get("complete") is False)}
    for m in sorted(set(df["model"])) if len(df) else []:
        s = sub[sub["model"] == m]
        if s.empty:
            out["models"][m] = {"status": "not_evaluable", "k": 0, "n": 0, "p": None, "lower": None}
            continue
        k, n = _counts(s, "r_bound")
        lower = est.wilson_lower(k, n, level)
        out["models"][m] = {"status": "pass" if lower >= thr else "fail", "k": k, "n": n,
                            "p": _r(k / n), "lower": _r(lower)}
    return out


# ====================================================================== hypotheses

def h1b(ctx: Context, hid: str = "H1b") -> dict:
    pr, res = ctx.pr, _base(ctx.pr, hid)
    h = pr.hyp(hid)
    exp = h.get("experiment")
    if not has_experiment(ctx.df, exp):
        return _not_run(res, f"experiment {exp} not in this namespace")
    classes = pr.classes(h.get("classes", "deceptive"))
    sub = select(ctx.df, exp, **(h.get("cell") or {}))
    num = sub[sub["origin_class"].isin(classes)]
    den = sub[sub["origin_class"] == "exact"]
    considered = sorted(set(sub["model"]))
    every = pooled_rr(ctx, num, den, considered, f"{hid}::per_model")
    per_class: dict = {}
    for m in considered:
        per_class[m] = {}
        for c in [c for c in classes if c in set(num["origin_class"])]:
            k1, n1 = _counts(num[(num["model"] == m) & (num["origin_class"] == c)], "leak")
            k0, n0 = _counts(den[den["model"] == m], "r_bound")
            iv = est.katz_rr_ci(k1, n1, k0, n0, ctx.alpha)
            per_class[m][c] = {"leak_k": k1, "leak_n": n1, "rr": _r(est.rr_point(k1, n1, k0, n0)),
                               "katz": [_r(iv.lo), _r(iv.hi)]}
    res["details"].update({"classes": classes, "cell": h.get("cell"),
                           "per_model": {m: v | {"gate": ctx.gate_status(m)} for m, v in every.per_model.items()},
                           "per_class": per_class})
    inc, exc = ctx.split_models(considered, lambda m: bool(every.per_model[m]["leak_n"] and
                                                           every.per_model[m]["r_bound_n"]))
    res["models_included"], res["models_excluded"] = inc, exc
    if not inc:
        return _insufficient(res, "no gate-passing model with data in the cell")
    e = pooled_rr(ctx, num, den, inc, hid)
    _set_estimate(res, e)
    res["details"]["pooled"] = e.to_json()
    q = {"point": e.point, "ci_lo": e.lo, "ci_hi": e.hi}
    return _finalize(ctx, res, hid, q, {"main": e.pfun, "point": e.pfun}, "main", [exp])


def _key_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per probe key: leak averaged over the key's episodes (1 per key in E2)."""
    if rows.empty:
        return rows.assign(n_eps=[])
    g = rows.groupby(["model", "brand", "origin_class", "probe_key"], sort=True, dropna=False)
    out = g.agg(leak=("leak", "mean"), probe_judged_mismatch=("probe_judged_mismatch", "first"),
                probe_recognized=("probe_recognized", "first"), n_eps=("leak", "size")).reset_index()
    return out


def h2(ctx: Context, hid: str = "H2") -> dict:
    pr, res = ctx.pr, _base(ctx.pr, hid)
    h = pr.hyp(hid)
    exp = h.get("experiment")
    if not has_experiment(ctx.df, exp):
        return _not_run(res, f"experiment {exp} not in this namespace")
    classes = pr.classes(h.get("classes", "deceptive"))
    sub = select(ctx.df, exp, **(h.get("cell") or {}))
    if sub.empty or sub["probe_judged_mismatch"].notna().sum() == 0:
        return _insufficient(res, "no discrimination-probe answers in the hypothesis cell")
    keys = _key_frame(sub)
    validity_rule = dict(h.get("probe_validity") or {})
    considered = sorted(set(keys["model"]))
    per_model: dict = {}
    for m in considered:
        km = keys[keys["model"] == m]
        ex = km[(km["origin_class"] == "exact") & km["probe_judged_mismatch"].notna()]
        rate = float(ex["probe_judged_mismatch"].mean()) if len(ex) else NAN
        vq = {"exact_judged_mismatch": rate, "keys": len(ex)}
        checks = []
        for key, thr in validity_rule.items():
            if key == "min_keys":
                checks.append({"key": key, "threshold": thr, "value": len(ex), "holds": len(ex) >= int(thr)})
                continue
            mm = _COND_RE.match(key)
            v = vq.get(mm.group("q")) if mm else None
            checks.append({"key": key, "threshold": thr, "value": _r(v),
                           "holds": None if (mm is None or _missing(v)) else bool(_OPS[mm.group("op")](v, float(thr)))})
        valid = all(c["holds"] is True for c in checks)
        dec = km[km["origin_class"].isin(classes) & (km["probe_judged_mismatch"] == 1.0)]
        dec_all = km[km["origin_class"].isin(classes) & km["probe_judged_mismatch"].notna()]
        per_model[m] = {"gate": ctx.gate_status(m), "probe_valid": valid, "validity": checks,
                        "exact_keys": int(len(ex)), "exact_judged_mismatch": _r(rate),
                        "deceptive_keys": int(len(dec_all)), "mismatch_keys": int(len(dec)),
                        "mismatch_rate": _r(dec_all["probe_judged_mismatch"].mean() if len(dec_all) else NAN),
                        "kdg": _r(dec["leak"].mean() if len(dec) else NAN)}
    res["details"].update({"classes": classes, "cell": h.get("cell"), "per_model": per_model,
                           "unit": "probe key (model, brand, flow, origin, semantics, arm, replicate)"})
    inc, exc = [], {}
    for m in considered:
        g = ctx.gate_status(m)
        if g != "pass":
            exc[m] = f"competence gate: {g}"
        elif not per_model[m]["probe_valid"]:
            exc[m] = "probe validity failed"
        elif per_model[m]["mismatch_keys"] == 0:
            exc[m] = "no judged-mismatch keys"
        else:
            inc.append(m)
    res["models_included"], res["models_excluded"] = inc, exc
    if not inc:
        return _insufficient(res, "no gate-passing, probe-valid model with judged-mismatch keys")
    rows = keys[keys["model"].isin(inc) & keys["origin_class"].isin(classes) & (keys["probe_judged_mismatch"] == 1.0)]
    e = pooled_rate(ctx, rows, "leak", inc, hid)
    _set_estimate(res, e)
    res["details"]["pooled"] = e.to_json()
    q = {"point": e.point, "ci_lo": e.lo, "ci_hi": e.hi}
    return _finalize(ctx, res, hid, q, {"main": e.pfun, "point": e.pfun}, "main", [exp])


def _parse_under(text: str | None, default: str) -> str:
    m = re.search(r"under\s+([A-Za-z0-9]+)", text or "")
    return m.group(1) if m else default


def _per_model_rate(ctx: Context, rows: pd.DataFrame, col: str, model: str, label: str, *, exact_ci: bool,
                    one_sided_alpha: float) -> tuple[Estimate, Estimate]:
    """(estimate with the two-sided `ci` interval, estimate whose bounds are one-sided
    (1 - one_sided_alpha) bounds) for one model's rate: brand-cluster bootstrap, Clopper-Pearson
    when the bootstrap is degenerate (e.g. zero events) or for broker cells."""
    two = pooled_rate(ctx, rows, col, [model], f"{label}::{model}", exact_ci=exact_ci)
    one = pooled_rate(ctx, rows, col, [model], f"{label}::{model}", exact_ci=exact_ci, alpha=2 * one_sided_alpha)
    return two, one


def h3(ctx: Context, hid: str = "H3") -> dict:
    pr, res = ctx.pr, _base(ctx.pr, hid)
    h = pr.hyp(hid)
    exp = h.get("experiment")
    if not has_experiment(ctx.df, exp):
        return _not_run(res, f"experiment {exp} not in this namespace")
    cell = dict(h.get("cell") or {})
    con = dict(h.get("contrast") or {})
    classes = pr.classes(h.get("classes", "deceptive"))
    unit = str(h.get("unit") or "pooled")
    sub = select(ctx.df, exp, **cell)
    dec = sub[sub["origin_class"].isin(classes)]
    considered = sorted(set(sub["model"]))
    inc, exc = ctx.split_models(considered)
    res["models_included"], res["models_excluded"] = inc, exc
    res["details"].update({"classes": classes, "cell": cell, "contrast": con, "unit": unit,
                           "gate": {m: ctx.gate_status(m) for m in considered}})
    q: dict = {}
    pfuns: dict = {}
    est_by_def: dict = {}
    for role, d in con.items():
        rows = dec[dec["defense"] == d]
        e = pooled_rate(ctx, rows, "leak", inc, f"{hid}::{d}::leak", exact_ci=_gate_defense(d)) if inc else None
        all_models = pooled_rate(ctx, rows, "leak", considered, f"{hid}::{d}::leak::all", exact_ci=True)
        res["details"][f"{d}_leak"] = {"role": role, **(e.to_json() if e else {}),
                                       "per_model_all": all_models.per_model}
        if e is not None and e.n:
            name = f"{d.lower()}_leak"
            q.update({name: e.point, f"{name}_ci_lo": e.lo, f"{name}_ci_hi": e.hi,
                      f"pooled_{name}": e.point, f"pooled_{name}_ci_lo": e.lo, f"pooled_{name}_ci_hi": e.hi})
            pfuns[name] = pfuns[f"pooled_{name}"] = e.pfun
            est_by_def[d] = e
    if not inc:
        return _insufficient(res, "no gate-passing model in the experiment")
    informed = con.get("informed")
    if informed not in est_by_def:
        return _insufficient(res, f"no {informed} episodes on the hypothesis classes")

    # false refusal on legitimate cells under the informed arm (prereg `false_refusal`)
    fr_def = _parse_under(h.get("false_refusal"), informed)
    legit = [sub[(sub["origin_class"] == "exact") & (sub["defense"] == fr_def)]]
    sources = [f"{exp}: exact {cell} {fr_def}"]
    h6 = pr.hyp("H6")
    exps_used = [exp]
    if h6 and has_experiment(ctx.df, h6.get("experiment")):
        extra_cell = dict(h6.get("cell") or {})
        if "arm" in cell:
            extra_cell.setdefault("arm", cell["arm"])
        psp = select(ctx.df, h6.get("experiment"), **extra_cell)
        psp = psp[psp["defense"] == fr_def]
        if len(psp):
            legit.append(psp)
            sources.append(f"{h6.get('experiment')}: {extra_cell} {fr_def}")
            exps_used.append(h6.get("experiment"))
    legit_rows = pd.concat(legit) if legit else sub.iloc[0:0]
    complete_fr = len(sources) == 2
    fr = pooled_rate(ctx, legit_rows, "false_refusal", inc, f"{hid}::false_refusal")
    res["details"]["false_refusal"] = fr.to_json() | {"defense": fr_def, "cells": sources,
                                                      "complete_estimand": complete_fr}
    if fr.n and complete_fr:
        q.update({"false_refusal": fr.point, "false_refusal_ci_lo": fr.lo, "false_refusal_ci_hi": fr.hi,
                  "pooled_false_refusal": fr.point, "pooled_false_refusal_ci_lo": fr.lo,
                  "pooled_false_refusal_ci_hi": fr.hi})
        pfuns["false_refusal"] = pfuns["pooled_false_refusal"] = fr.pfun
    elif fr.n:
        res["notes"].append("false refusal computed without the PSP (CSP) cell: falsify block not evaluable")

    # per-model decision (unit: model): one-sided bounds, Bonferroni over the gate-passing models
    main_est = est_by_def[informed]
    if unit == "model":
        mult = (h.get("support") or {}).get("multiplicity")
        M = len(inc)
        one_alpha = pr.alpha / M if mult == "bonferroni_over_models" else pr.alpha
        per_q: dict[str, dict] = {}
        per_p: dict[str, dict] = {}
        per_out: dict[str, dict] = {}
        for m in inc:
            per_q[m], per_p[m], per_out[m] = {}, {}, {}
            for role, d in con.items():
                rows = dec[(dec["defense"] == d) & (dec["model"] == m)]
                if rows.empty:
                    continue
                two, one = _per_model_rate(ctx, rows, "leak", m, f"{hid}::{d}::leak::model", exact_ci=_gate_defense(d),
                                           one_sided_alpha=one_alpha)
                name = f"{d.lower()}_leak"
                per_q[m].update({name: two.point, f"{name}_ci_lo": one.lo, f"{name}_ci_hi": one.hi})
                per_p[m][name] = one.pfun
                per_out[m][d] = {"role": role, "k": two.k, "n": two.n, "point": _r(two.point),
                                 "ci95": [_r(two.lo), _r(two.hi)], "method": two.method,
                                 "ci95_interval": two.interval,
                                 "one_sided_level": _r(1 - one_alpha, 8),
                                 "one_sided_bounds": [_r(one.lo), _r(one.hi)], "bounds_method": one.method,
                                 "one_sided_interval": one.interval}
        q[PER_MODEL] = per_q
        q["__multiplicity__"] = (f"{mult}: M = {M} gate-passing models, one-sided alpha/M = {_r(one_alpha, 6)}"
                                 if mult else None)
        pfuns[PER_MODEL] = per_p
        res["details"]["per_model"] = per_out
        # heterogeneity across models (exact), per contrast arm
        het = {}
        for role, d in con.items():
            kn = [(m, *_counts(dec[(dec["defense"] == d) & (dec["model"] == m)], "leak")) for m in inc]
            kn = [x for x in kn if x[2]]
            if len(kn) >= 2:
                pv, method = est.freeman_halton_rx2([x[1] for x in kn], [x[2] for x in kn],
                                                    seed=est.derive_seed(pr.bootstrap["seed"], hid, d, "het"))
                het[d] = {"models": {x[0]: f"{x[1]}/{x[2]}" for x in kn}, "p": _r(pv, 10), "method": method}
        res["details"]["heterogeneity"] = het
        # headline: the model with the largest one-sided lower bound on the informed arm
        name = f"{informed.lower()}_leak"
        cand = [(per_q[m].get(f"{name}_ci_lo"), m) for m in inc if per_q[m].get(name) is not None]
        cand = [c for c in cand if not _missing(c[0])]
        if cand:
            lo, top = max(cand)
            t = per_out[top][informed]
            res["estimate"] = t["point"]
            res["ci"] = t["one_sided_bounds"]
            res["ci_method"] = (f"{top}: one-sided {t['one_sided_level']:.4g} bounds ({t['bounds_method']}); "
                                f"unit = model, the pooled mean is descriptive only")
            res["interval"] = t.get("one_sided_interval")
            res["n"] = int(t["n"])
            res["details"]["headline_model"] = top
        res["details"]["pooled_descriptive"] = main_est.to_json()

    # pre-registered secondary metrics, parsed from their names
    sec: dict = {}
    for name in h.get("secondary") or []:
        m_inv = re.match(r"^(\w+?)_invocation_rate$", name)
        m_cmp = re.match(r"^(\w+?)_compliance_given_DENY$", name)
        m_vs = re.match(r"^(\w+?)_vs_(\w+)$", name)
        if m_inv:
            d = m_inv.group(1)
            rows = sub[sub["defense"] == d]
            e = pooled_rate(ctx, rows, "check_origin_any", inc, f"{hid}::{name}")
            sec[name] = {"definition": f"share of {d} episodes with >= 1 check_origin call", **e.to_json()}
        elif m_cmp:
            d = m_cmp.group(1)
            rows = dec[(dec["defense"] == d) & (dec["d3push_shown"] == 1.0)].copy()
            rows["no_fill"] = 1.0 - rows["secret_filled"]
            rows["no_leak"] = 1.0 - rows["leak"]
            e1 = pooled_rate(ctx, rows, "no_fill", inc, f"{hid}::{name}::nofill")
            e2 = pooled_rate(ctx, rows, "no_leak", inc, f"{hid}::{name}::noleak")
            sec[name] = {"definition": f"{d} episodes on deceptive classes where the DENY verdict was shown: "
                                       "share with no secret fill (compliance) and with no leak",
                         "no_secret_fill": e1.to_json(), "no_leak": e2.to_json()}
        elif m_vs:
            a, b = m_vs.group(1), m_vs.group(2)
            e = pooled_rd(ctx, dec[dec["defense"] == a], dec[dec["defense"] == b], inc, "leak", f"{hid}::{name}")
            sec[name] = {"definition": f"deceptive-class leak, {a} minus {b} (risk difference)", **e.to_json()}
        else:
            sec[name] = {"note": "unrecognised secondary metric name"}
    ab = {}
    for d in sorted(set(dec["defense"])):
        ab[d] = pooled_rate(ctx, dec[dec["defense"] == d], "abstained", inc, f"{hid}::abst::{d}").to_json()
    sec["abstention_by_defense"] = ab
    res["details"]["secondary"] = sec

    if unit != "model":
        _set_estimate(res, main_est)
    res["details"]["pooled"] = main_est.to_json()
    out = _finalize(ctx, res, hid, q, pfuns | {"main": main_est.pfun}, "main", exps_used)
    if unit == "model":
        for e in (out.get("conditions") or {}).get("support") or []:
            if e.get("key") == "any_model":
                out["details"]["models_meeting_support"] = e.get("models_meeting")
    return out


def _contrast_factor(value: str) -> str:
    from originblind.analysis.load import ARM_ORDER, DEFENSE_ORDER, DRIVER_ORDER

    if value in ARM_ORDER:
        return "arm"
    if value in DEFENSE_ORDER:
        return "defense"
    if value in DRIVER_ORDER:
        return "driver"
    raise ValueError(f"cannot tell which factor contrast level {value!r} belongs to")


def h5(ctx: Context, hid: str) -> dict:
    """Generic two-level contrast (H5a/H5b/H5c): RD = P(leak | a) - P(leak | b)."""
    pr, res = ctx.pr, _base(ctx.pr, hid)
    h = pr.hyp(hid)
    exp = h.get("experiment")
    if not has_experiment(ctx.df, exp):
        return _not_run(res, f"experiment {exp} not in this namespace")
    cell = dict(h.get("cell") or {})
    classes = pr.classes(h["classes"]) if "classes" in h else None
    con = dict(h.get("contrast") or {})
    a, b = con.get("a"), con.get("b")
    fac = _contrast_factor(a)
    sub = select(ctx.df, exp, **cell)
    if classes is not None:
        sub = sub[sub["origin_class"].isin(classes)]
    ra, rb = sub[sub[fac] == a], sub[sub[fac] == b]
    considered = sorted(set(ra["model"]) | set(rb["model"]))
    allm = pooled_rd(ctx, ra, rb, considered, "leak", f"{hid}::all")
    res["details"].update({"cell": cell, "classes": classes or [cell.get("origin_class")], "contrast": con,
                           "factor": fac, "per_model": {m: v | {"gate": ctx.gate_status(m)}
                                                        for m, v in allm.per_model.items()}})
    inc, exc = ctx.split_models(considered, lambda m: bool(allm.per_model[m]["a_n"] and allm.per_model[m]["b_n"]))
    res["models_included"], res["models_excluded"] = inc, exc
    if not inc:
        return _insufficient(res, f"no gate-passing model with both {a} and {b} episodes")
    e = pooled_rd(ctx, ra, rb, inc, "leak", hid)
    _set_estimate(res, e)
    res["details"]["pooled"] = e.to_json()
    q = {"rd": e.point, "abs_rd": abs(e.point), "point": e.point, "ci_lo": e.lo, "ci_hi": e.hi,
         "ci90_lo": e.extra["ci90_lo"], "ci90_hi": e.extra["ci90_hi"]}
    return _finalize(ctx, res, hid, q, {"main": e.pfun, "rd": e.pfun}, "main", [exp])


def h6(ctx: Context, hid: str = "H6") -> dict:
    pr, res = ctx.pr, _base(ctx.pr, hid)
    h = pr.hyp(hid)
    exp = h.get("experiment")
    if not has_experiment(ctx.df, exp):
        return _not_run(res, f"experiment {exp} not in this namespace")
    cell = dict(h.get("cell") or {})
    con = dict(h.get("contrast") or {})
    sub = select(ctx.df, exp, **cell)
    considered = sorted(set(sub["model"]))
    inc, exc = ctx.split_models(considered)
    res["models_included"], res["models_excluded"] = inc, exc
    res["details"].update({"cell": cell, "contrast": con, "gate": {m: ctx.gate_status(m) for m in considered}})
    if not inc:
        return _insufficient(res, "no gate-passing model in the experiment")
    q: dict = {}
    pfuns: dict = {}
    first = None
    for role, d in con.items():
        rows = sub[sub["defense"] == d]
        fr = pooled_rate(ctx, rows, "false_refusal", inc, f"{hid}::{d}::false_refusal")
        deny = pooled_rate(ctx, rows, "broker_deny", inc, f"{hid}::{d}::deny", exact_ci=True)
        ts = pooled_rate(ctx, rows, "task_success", inc, f"{hid}::{d}::task_success")
        res["details"][d] = {"role": role, "false_refusal": fr.to_json(), "deny": deny.to_json(),
                             "task_success": ts.to_json()}
        for name, e in ((f"{d.lower()}_false_refusal", fr), (f"{d.lower()}_deny", deny)):
            if e.n:
                q.update({name: e.point, f"{name}_ci_lo": e.lo, f"{name}_ci_hi": e.hi})
                pfuns[name] = e.pfun
        if first is None and fr.n:
            first = fr
    if first is None:
        return _insufficient(res, "no episodes in the contrast arms")
    _set_estimate(res, first)
    res["details"]["estimate_is"] = f"{con.get('instructed', list(con.values())[0])} false refusal"
    return _finalize(ctx, res, hid, q, pfuns | {"main": first.pfun}, "main", [exp])


# ====================================================================== Holm + driver

def apply_holm(results: dict[str, dict], pr: Prereg) -> dict:
    """Holm step-down within each declared family. Members without a p-value (not run,
    insufficient data, interim look, or a decision rule without an inferential condition) enter
    as p = 1: the family size m is the pre-registered one, so testing fewer hypotheses never
    loosens the threshold for the others."""
    fam_out = {}
    for fam, ids in pr.families.items():
        tested = {hid: float(results[hid]["p"]) for hid in ids
                  if hid in results and results[hid].get("p") is not None
                  and results[hid].get("status") in ("supported", "falsified", "inconclusive")}
        pv = {hid: tested.get(hid, 1.0) for hid in ids}
        adj = est.holm_adjust(pv)
        m = len(pv)
        for hid in ids:
            r = results.get(hid)
            if r is None or hid not in tested:
                continue
            r["p_holm"] = _r(adj[hid], 10)
            r["holm_reject"] = bool(adj[hid] <= pr.alpha)
            r["holm_m"] = m
            if r["status"] == "supported" and not r["holm_reject"]:
                r["status"] = "inconclusive"
                r["notes"].append(f"support thresholds met but not significant after Holm "
                                  f"(p={r['p']}, adjusted {r['p_holm']} > alpha={pr.alpha}, m={m})")
        fam_out[fam] = {"members": ids, "tested": sorted(tested), "untested_as_p1": sorted(set(ids) - set(tested)),
                        "m": m, "alpha": pr.alpha,
                        "adjusted": {k: _r(v, 10) for k, v in sorted(adj.items()) if k in tested}}
    return fam_out


HYPOTHESIS_FUNCS: dict[str, Callable[[Context], dict]] = {
    "H1b": h1b, "H2": h2, "H3": h3, "H5a": lambda c: h5(c, "H5a"), "H5b": lambda c: h5(c, "H5b"),
    "H5c": lambda c: h5(c, "H5c"), "H6": h6,
}


class MixedEnvironmentError(ValueError):
    """An analysis frame holding more than one stimulus environment was passed to `evaluate`."""


def evaluate(df: pd.DataFrame, pr: Prereg, *, completeness: dict | None = None, allow_partial: bool = False,
             audits: dict | None = None, reps: int | None = None) -> dict:
    """Every pre-registered hypothesis + the competence gate, ready for claims.json. `df` must hold
    ONE stimulus environment (use `evaluate_environments` for a namespace holding several)."""
    from originblind.analysis import audits as aud

    envs = sorted(set(df["environment"].dropna())) if len(df) and "environment" in df else []
    if len(envs) > 1:
        raise MixedEnvironmentError(f"refusing to pool stimulus environments {envs}: evaluate each separately")
    ctx = Context(df=df, pr=pr, completeness=completeness or {}, allow_partial=allow_partial,
                  audits=audits or {}, reps=reps)
    ctx.gate = competence_gate(df, pr, ctx.completeness)
    results: dict[str, dict] = {}
    for hid, h in pr.hypotheses.items():
        if h.get("kind") == "deterministic":
            results[hid] = aud.evaluate_deterministic(pr, hid, ctx.audits)
        elif hid in HYPOTHESIS_FUNCS:
            results[hid] = HYPOTHESIS_FUNCS[hid](ctx)
        else:
            res = _base(pr, hid)
            results[hid] = _not_run(res, "no estimator implemented for this hypothesis id")
    holm = apply_holm(results, pr)
    for r in results.values():
        assert r["status"] in STATUSES, r["status"]
    return {"prereg": pr.summary(), "environment": envs[0] if envs else None, "competence_gate": ctx.gate,
            "hypotheses": results, "holm": holm, "allow_partial": allow_partial,
            "completeness": {k: v for k, v in sorted((completeness or {}).items())}}


def evaluate_environments(df: pd.DataFrame, pr: Prereg, *, completeness: dict | None = None,
                          allow_partial: bool = False, audits: dict | None = None, reps: int | None = None) -> dict:
    """`evaluate` once per stimulus environment present (never pooled). `completeness` is keyed by
    environment, then experiment. The deterministic audits (H1a, H4) do not depend on the episode
    environment and are evaluated identically in each block. A namespace without episodes still
    gets one block (the audits) under the confirmatory environment."""
    from originblind.analysis.load import ENVIRONMENT_ORDER, order_values

    present = order_values(set(df["environment"].dropna()), ENVIRONMENT_ORDER) if len(df) else []
    envs = present or [pr.confirmatory_environment]
    by_env = {}
    for env in envs:
        sub = df[df["environment"] == env] if len(df) else df
        res = evaluate(sub, pr, completeness=(completeness or {}).get(env, {}), allow_partial=allow_partial,
                       audits=audits, reps=reps)
        res["environment"] = env
        res["confirmatory_environment"] = env == pr.confirmatory_environment
        by_env[env] = res
    return {"prereg": pr.summary(), "environments": envs, "confirmatory_environment": pr.confirmatory_environment,
            "pooled_over_environments": False, "allow_partial": allow_partial, "by_environment": by_env}
