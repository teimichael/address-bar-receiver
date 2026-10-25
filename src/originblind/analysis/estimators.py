"""Proportion estimators, risk differences, relative release, the cluster bootstrap, and
multiplicity control — closed forms or well-tested numerics, validated in tests against
scipy/statsmodels reference values and hand-computed fixtures.

Conventions used throughout the analysis package:
  - `alpha` is the two-sided error rate of a confidence interval (CI level = 1 - alpha).
  - Relative release RR = P(leak | class) / P(R_bound | exact) is reported UNCLIPPED (an agent
    that leaks more often on a lookalike than it completes on the real site has RR > 1);
    OSI = 1 - RR. The legacy clipped `osi_point` is kept only for the schema-v1 entry point.
  - Pooled estimates are equal-weight means over models; their CIs come from a brand-cluster
    bootstrap stratified by model (`cluster_bootstrap`). Two-arm risk differences use the
    Newcombe hybrid score interval; its MOVER generalisation (`mover_wilson`) gives the same
    interval for an equal-weight mean of per-model differences.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float


def wilson(k: int, n: int, alpha: float = 0.05) -> Interval:
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Interval:
    if n == 0:
        return Interval(float("nan"), 0.0, 1.0)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return Interval(k / n, lo, hi)


def zero_event_upper(n: int, alpha: float = 0.05) -> float:
    """Two-sided upper bound when 0/n events occurred (matches Clopper-Pearson hi at k=0)."""
    return 1.0 - (alpha / 2) ** (1.0 / n) if n > 0 else 1.0


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, alpha: float = 0.05) -> Interval:
    """Newcombe hybrid-score CI for p1 - p2 (method 10). Point = p1 - p2."""
    w1 = wilson(k1, n1, alpha)
    w2 = wilson(k2, n2, alpha)
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    lo = diff - math.sqrt((p1 - w1.lo) ** 2 + (w2.hi - p2) ** 2)
    hi = diff + math.sqrt((w1.hi - p1) ** 2 + (p2 - w2.lo) ** 2)
    return Interval(diff, max(-1.0, lo), min(1.0, hi))


def osi_point(leak_k: int, leak_n: int, base_k: int, base_n: int) -> float:
    """OSI_c = 1 - P(L|c) / P(R_bound|exact). Clamped to [0,1] for reporting.

    LEGACY (schema-v1 `analyze` only): clipping hides agents that leak more than they complete;
    the v2 analysis uses the unclipped `rr_point` / `osi_from_rr`."""
    if base_n == 0 or base_k == 0:
        return float("nan")
    p_leak = leak_k / leak_n if leak_n else float("nan")
    p_base = base_k / base_n
    return max(0.0, min(1.0, 1.0 - p_leak / p_base))


def mde_two_prop(n_per_arm: int, p0: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable difference for two independent proportions (normal approx)."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    # solve delta with pooled-ish variance at p0 and p0+delta via fixed-point
    delta = 0.01
    for _ in range(200):
        p1 = min(0.999, p0 + delta)
        sd = math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
        new = (z_a + z_b) * sd / math.sqrt(n_per_arm)
        if abs(new - delta) < 1e-6:
            break
        delta = new
    return delta


def tost_equiv(k1: int, n1: int, k2: int, n2: int, margin: float, alpha: float = 0.05) -> bool:
    """TOST equivalence via a (1-2alpha) Newcombe CI fully inside +-margin."""
    ci = newcombe_diff(k1, n1, k2, n2, alpha=2 * alpha)
    return ci.lo > -margin and ci.hi < margin


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns name -> reject?."""
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(order)
    out, prev_reject = {}, True
    for i, (name, p) in enumerate(order):
        thresh = alpha / (m - i)
        reject = prev_reject and p <= thresh
        out[name] = reject
        prev_reject = reject
    return out


def h1_decision(osi: float, ci_lo: float, ci_hi: float,
                support_hi: float = 0.45, falsify_lo: float = 0.45) -> str:
    """Pre-registered three-way read for an OSI-style hypothesis."""
    if math.isnan(osi):
        return "inconclusive"
    if ci_hi < support_hi:
        return "supported"
    if ci_lo > falsify_lo:
        return "falsified"
    return "inconclusive"


# ====================================================================== v2 additions

NAN = float("nan")


def wilson_lower(k: int, n: int, level: float = 0.90) -> float:
    """Lower limit of the two-sided `level` Wilson interval (= one-sided (1+level)/2 bound).
    The competence gate's `wilson_lower_90` is wilson_lower(k, n, 0.90)."""
    return wilson(k, n, alpha=1.0 - level).lo if n else NAN


def rr_point(k1: int, n1: int, k0: int, n0: int) -> float:
    """Unclipped relative release (k1/n1) / (k0/n0).

    nan when either group is empty or both rates are 0 (0/0); +inf when the reference rate is 0
    but the numerator is not (the agent leaks yet never completes the legitimate task)."""
    if n1 == 0 or n0 == 0:
        return NAN
    p1, p0 = k1 / n1, k0 / n0
    if p0 == 0:
        return math.inf if p1 > 0 else NAN
    return p1 / p0


def katz_rr_ci(k1: int, n1: int, k0: int, n0: int, alpha: float = 0.05) -> Interval:
    """Katz log interval for RR = (k1/n1)/(k0/n0).

    SE(log RR) = sqrt(1/k1 - 1/n1 + 1/k0 - 1/n0). When any of the four cells is zero the
    Haldane-Anscombe correction adds 0.5 to each cell (n grows by 1): a zero event count leaves
    log RR undefined, and a zero non-event count (k = n, e.g. 6/6 vs 12/12) collapses the SE to 0
    and would report a zero-width interval. The point estimate is the unclipped raw RR whenever
    it is finite, else the corrected one; the interval is never clipped."""
    if n1 == 0 or n0 == 0:
        return Interval(NAN, NAN, NAN)
    a, b, c, d = float(k1), float(n1 - k1), float(k0), float(n0 - k0)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    rr = (a / (a + b)) / (c / (c + d))
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    z = stats.norm.ppf(1 - alpha / 2)
    raw = rr_point(k1, n1, k0, n0)
    point = raw if math.isfinite(raw) else rr
    return Interval(point, math.exp(math.log(rr) - z * se), math.exp(math.log(rr) + z * se))


def osi_from_rr(iv: Interval) -> Interval:
    """OSI = 1 - RR, unclipped; the bounds swap."""
    return Interval(1 - iv.point, 1 - iv.hi, 1 - iv.lo)


def mover_wilson(terms: Sequence[tuple[float, int, int]], alpha: float = 0.05,
                 clip: tuple[float, float] | None = (-1.0, 1.0)) -> Interval:
    """CI for a weighted sum sum_i w_i * p_i of INDEPENDENT proportions p_i = k_i/n_i by the
    MOVER method (Zou & Donner 2008) with Wilson limits for each p_i.

    With terms [(1, k1, n1), (-1, k2, n2)] this is exactly Newcombe's hybrid score interval for
    p1 - p2 (method 10); with weights +-1/M over M models it is the Newcombe-type interval for
    the equal-weight mean of per-model risk differences."""
    theta, lo_sq, hi_sq = 0.0, 0.0, 0.0
    for w, k, n in terms:
        if n == 0:
            return Interval(NAN, NAN, NAN)
        iv = wilson(k, n, alpha)
        p = k / n
        theta += w * p
        if w >= 0:
            lo_sq += (w * (p - iv.lo)) ** 2
            hi_sq += (w * (iv.hi - p)) ** 2
        else:
            lo_sq += (w * (iv.hi - p)) ** 2
            hi_sq += (w * (p - iv.lo)) ** 2
    lo, hi = theta - math.sqrt(lo_sq), theta + math.sqrt(hi_sq)
    if clip is not None:
        lo, hi = max(clip[0], lo), min(clip[1], hi)
    return Interval(theta, lo, hi)


def pooled_rd_terms(per_model: Sequence[tuple[int, int, int, int]]) -> list[tuple[float, int, int]]:
    """MOVER terms for the equal-weight mean over models of (ka/na - kb/nb)."""
    m = len(per_model)
    terms: list[tuple[float, int, int]] = []
    for ka, na, kb, nb in per_model:
        terms += [(1.0 / m, ka, na), (-1.0 / m, kb, nb)]
    return terms


# ---------------------------------------------------------------- episode-level intervals (prereg v2.1)

def _snap(x: float) -> float:
    """Remove floating noise from a count that is an integer up to rounding (k = p * n)."""
    r = round(x)
    return float(r) if abs(x - r) < 1e-9 else float(x)


@dataclass(frozen=True)
class EpisodeInterval:
    """The episode-level interval of an equal-weight mean of per-model proportions."""
    point: float
    lo: float
    hi: float
    method: str  # "Wilson" | "Clopper-Pearson"
    k_eff: float
    n_eff: float


def binomial_mean_interval(per: Sequence[tuple[int, int]], alpha: float = 0.05, *,
                           exact: bool = False) -> EpisodeInterval:
    """Episode-level interval (episodes independent, no brand clustering) for the equal-weight mean
    p = (1/M) sum_m k_m/n_m of M independent per-model proportions.

    The binomial interval at Kish's effective sample size n_eff = 1 / sum_m (1/M)^2 / n_m with
    k_eff = p * n_eff: Wilson, or Clopper-Pearson when p is 0 or 1 (k = 0 or k = n) or `exact` is
    set (broker cells, zero-event by design). For one model this is exactly Wilson / Clopper-Pearson
    on its counts; when every model has the same n it is the interval on the pooled counts
    (sum k / sum n), which then equals the equal-weight mean."""
    per = [(int(k), int(n)) for k, n in per if int(n) > 0]
    if not per:
        return EpisodeInterval(NAN, NAN, NAN, "no data", 0.0, 0.0)
    m = len(per)
    p = sum(k / n for k, n in per) / m
    n_eff = _snap(1.0 / sum((1.0 / m) ** 2 / n for _, n in per))
    k_eff = _snap(p * n_eff)
    if exact or k_eff <= 0 or k_eff >= n_eff:
        iv = clopper_pearson(k_eff, n_eff, alpha)
        method = "Clopper-Pearson"
    else:
        iv = wilson(k_eff, n_eff, alpha)
        method = "Wilson"
    return EpisodeInterval(p, float(iv.lo), float(iv.hi), method, k_eff, n_eff)


def mover_combine(terms: Sequence[tuple[float, float, float, float]]) -> Interval:
    """MOVER (Zou & Donner 2008; Zou 2008 for sums of relative risks) interval for a weighted sum
    sum_i w_i * theta_i of INDEPENDENT estimates, each given as (w_i, theta_i, l_i, u_i) with its own
    interval. A limit that lies on the wrong side of its point (e.g. a Haldane-corrected Katz bound
    beside a raw zero RR) is treated as the point itself. nan if any part is not finite."""
    theta, lo_sq, hi_sq = 0.0, 0.0, 0.0
    for w, t, lo, hi in terms:
        if not all(math.isfinite(x) for x in (w, t, lo, hi)):
            return Interval(NAN, NAN, NAN)
        lo, hi = min(lo, t), max(hi, t)
        theta += w * t
        if w >= 0:
            lo_sq += (w * (t - lo)) ** 2
            hi_sq += (w * (hi - t)) ** 2
        else:
            lo_sq += (w * (hi - t)) ** 2
            hi_sq += (w * (t - lo)) ** 2
    return Interval(theta, theta - math.sqrt(lo_sq), theta + math.sqrt(hi_sq))


def mover_katz_mean(per: Sequence[tuple[int, int, int, int]], alpha: float = 0.05) -> Interval:
    """Episode-level interval for the equal-weight mean over models of RR_m = (k1/n1) / (k0/n0):
    MOVER combination of the per-model Katz (Haldane-corrected) intervals; exactly Katz for one
    model. nan when a per-model RR is undefined or infinite."""
    per = [tuple(int(x) for x in t) for t in per]
    if not per:
        return Interval(NAN, NAN, NAN)
    m = len(per)
    terms = []
    for k1, n1, k0, n0 in per:
        iv = katz_rr_ci(k1, n1, k0, n0, alpha)
        terms.append((1.0 / m, iv.point, iv.lo, iv.hi))
    return mover_combine(terms)


def percentile_interval(reps: np.ndarray, alpha: float) -> tuple[float, float]:
    """Percentile interval of stored bootstrap replicates at another level (same replicates)."""
    r = np.asarray(reps, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return NAN, NAN
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2], method="inverted_cdf")
    return float(lo), float(hi)


def union_interval(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """(min lower, max upper) of two intervals; nan if either is undefined (a missing component
    never narrows the union by being ignored)."""
    (alo, ahi), (blo, bhi) = a, b
    vals = [alo, ahi, blo, bhi]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return NAN, NAN
    return float(min(alo, blo)), float(max(ahi, bhi))


@dataclass(frozen=True)
class Tost:
    margin: float
    ci_lo: float  # (1 - 2 alpha) interval
    ci_hi: float
    equivalent: bool


def tost_newcombe(k1: int, n1: int, k2: int, n2: int, margin: float, alpha: float = 0.05) -> Tost:
    """TOST for |p1 - p2| < margin: equivalence iff the (1 - 2 alpha) Newcombe interval (90% at
    alpha = 0.05) lies strictly inside (-margin, +margin)."""
    ci = newcombe_diff(k1, n1, k2, n2, alpha=2 * alpha)
    return Tost(margin, ci.lo, ci.hi, ci.lo > -margin and ci.hi < margin)


def ci_inversion_p(ci_at: Callable[[float], tuple[float, float]], null: float = 0.0,
                   side: str = "two", tol: float = 1e-10) -> float:
    """p-value consistent with an interval procedure: the smallest alpha at which the two-sided
    (1 - alpha) interval excludes `null` (side='two'), lies above it ('greater'), or below it
    ('less'). Valid for procedures whose intervals nest as alpha grows (Wilson, Clopper-Pearson,
    Newcombe/MOVER). Returns 1.0 if even the degenerate interval does not exclude the null."""

    def excludes(a: float) -> bool:
        lo, hi = ci_at(a)
        if side == "greater":
            return lo > null
        if side == "less":
            return hi < null
        return lo > null or hi < null

    a_lo, a_hi = 1e-12, 1.0 - 1e-9  # z stays finite at both ends
    if not excludes(a_hi):
        return 1.0
    if excludes(a_lo):
        return a_lo  # p <= 1e-12
    for _ in range(200):
        mid = 0.5 * (a_lo + a_hi)
        if excludes(mid):
            a_hi = mid
        else:
            a_lo = mid
        if a_hi - a_lo < tol:
            break
    return a_hi


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone; equals statsmodels multipletests 'holm')."""
    order = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(order)
    out: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


# ---------------------------------------------------------------- cluster bootstrap

def derive_seed(base: int, *parts: object) -> int:
    """Order-independent seed for one statistic: results never depend on evaluation order."""
    h = hashlib.sha256(("::".join([str(base), *map(str, parts)])).encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**63 - 1)


def safe_ratio(a, b):
    """Elementwise a / b with +inf for x/0 (x > 0) and nan for 0/0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return np.where(b == 0, np.where(a > 0, np.inf, np.nan), out)


class WeightedView:
    """Cluster-weighted sums over a fixed set of rows.

    Row r belongs to cluster codes[r]; W has one row per bootstrap replicate (W[b, c] = how
    often cluster c was drawn in replicate b; a single row of ones = the original sample).
    Every method returns an array with one value per replicate, so a statistic written with
    these primitives is evaluated for all replicates at once."""

    def __init__(self, df: pd.DataFrame, codes: np.ndarray, W: np.ndarray) -> None:
        self.df = df
        self._codes = codes
        self.W = W
        self.C = W.shape[1]
        self._cache: dict[str, np.ndarray] = {}

    def values(self, col: str) -> np.ndarray:
        if col not in self._cache:
            self._cache[col] = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=float)
        return self._cache[col]

    def _mask(self, mask) -> np.ndarray:
        if mask is None:
            return np.ones(len(self.df), dtype=bool)
        return np.asarray(mask, dtype=bool)

    def total(self, col: str | None, mask=None) -> np.ndarray:
        m = self._mask(mask)
        if col is None:
            per = np.bincount(self._codes[m], minlength=self.C).astype(float)
        else:
            v = self.values(col)
            m = m & ~np.isnan(v)
            per = np.bincount(self._codes[m], weights=v[m], minlength=self.C)
        return self.W @ per

    def count(self, mask=None, col: str | None = None) -> np.ndarray:
        """Number of rows (weighted) in the mask; with `col`, only rows where col is not NaN."""
        m = self._mask(mask)
        if col is not None:
            m = m & ~np.isnan(self.values(col))
        return self.W @ np.bincount(self._codes[m], minlength=self.C).astype(float)

    def mean(self, col: str, mask=None) -> np.ndarray:
        """Weighted mean of `col` over the masked rows (NaN values are ignored)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.total(col, mask) / self.count(mask, col)


@dataclass(frozen=True)
class BootResult:
    point: float
    lo: float
    hi: float
    alpha: float
    n_reps: int
    n_undefined: int  # replicates where the statistic was undefined (0/0); excluded
    degenerate: bool  # every defined replicate equal (e.g. zero events): not a valid CI
    n_clusters: dict = field(default_factory=dict)  # stratum -> number of clusters
    reps: np.ndarray = field(default_factory=lambda: np.array([]), repr=False, compare=False)
    method: str = "cluster bootstrap (percentile)"

    def p_value(self, null: float, alternative: str = "greater") -> float:
        return boot_pvalue(self.reps, null, alternative)

    def to_json(self) -> dict:
        return {"point": _r(self.point), "lo": _r(self.lo), "hi": _r(self.hi), "alpha": self.alpha,
                "n_reps": self.n_reps, "n_undefined": self.n_undefined, "degenerate": self.degenerate,
                "n_clusters": self.n_clusters, "method": self.method}


def _r(x, nd: int = 6):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(xf):
        return None
    if math.isinf(xf):
        return "inf" if xf > 0 else "-inf"
    return round(xf, nd)


def boot_pvalue(reps: np.ndarray, null: float, alternative: str = "greater") -> float:
    """One-sided bootstrap p-value by percentile inversion, with the +1 correction.

    alternative='greater' tests H0: theta <= null (p = share of replicates at or below the null);
    'less' tests H0: theta >= null; 'two' doubles the smaller tail (capped at 1)."""
    r = np.asarray(reps, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return NAN
    lo_tail = (1 + np.sum(r <= null)) / (r.size + 1)
    hi_tail = (1 + np.sum(r >= null)) / (r.size + 1)
    if alternative == "greater":
        return float(lo_tail)
    if alternative == "less":
        return float(hi_tail)
    return float(min(1.0, 2 * min(lo_tail, hi_tail)))


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[WeightedView], np.ndarray], *,
                      cluster: str = "brand", strata: str | Sequence[str] | None = None,
                      reps: int = 4000, seed: int = 0, alpha: float = 0.05) -> BootResult:
    """Nonparametric cluster bootstrap with a percentile interval.

    Clusters (e.g. brands) are resampled with replacement; with `strata` (e.g. model) the
    clusters of each stratum are resampled within that stratum, so every replicate keeps the
    per-stratum number of clusters (equal-weight pooling over models stays well defined).
    `stat` receives a WeightedView and must return one value per replicate (array) using its
    weighted primitives; it is called twice (original sample, then all replicates at once), so
    arbitrary statistics are cheap. Clusters are ordered by their key, not by row order, so the
    result depends only on the data and the seed."""
    if df is None or len(df) == 0:
        return BootResult(NAN, NAN, NAN, alpha, 0, 0, True)
    scols = [] if strata is None else ([strata] if isinstance(strata, str) else list(strata))
    keys = df[cluster].astype(str)
    for c in reversed(scols):
        keys = df[c].astype(str) + "\x1f" + keys
    uniq = sorted(set(keys))
    codes = pd.Categorical(keys, categories=uniq).codes.astype(np.int64)
    C = len(uniq)
    stratum_of = [k.rsplit("\x1f", 1)[0] if scols else "" for k in uniq]

    point_arr = np.asarray(stat(WeightedView(df, codes, np.ones((1, C)))), dtype=float).reshape(-1)
    point = float(point_arr[0]) if point_arr.size else NAN

    rng = np.random.default_rng(seed)
    W = np.zeros((reps, C))
    n_clusters: dict[str, int] = {}
    for s in sorted(set(stratum_of)):
        idx = np.array([i for i, st in enumerate(stratum_of) if st == s])
        n_clusters[s or "all"] = int(idx.size)
        W[:, idx] = rng.multinomial(idx.size, np.full(idx.size, 1.0 / idx.size), size=reps)
    vals = np.asarray(stat(WeightedView(df, codes, W)), dtype=float).reshape(-1)
    if vals.size == 1 and reps > 1:  # a statistic that ignores the weights
        vals = np.repeat(vals, reps)
    nan = np.isnan(vals)
    valid = vals[~nan]
    if valid.size == 0:
        return BootResult(point, NAN, NAN, alpha, reps, int(nan.sum()), True, n_clusters, valid)
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2], method="inverted_cdf")
    degenerate = bool(np.all(valid == valid[0]))
    return BootResult(point, float(lo), float(hi), alpha, reps, int(nan.sum()), degenerate, n_clusters, valid)


# ---------------------------------------------------------------- heterogeneity across models

def freeman_halton_rx2(ks: Sequence[int], ns: Sequence[int], *, max_tables: int = 5_000_000,
                       mc_reps: int = 200_000, seed: int = 0) -> tuple[float, str]:
    """Exact test of homogeneity of r proportions (Freeman-Halton extension of Fisher's exact test
    to an r x 2 table): conditional on both margins, the tables are multivariate hypergeometric;
    p = P(tables at most as probable as the observed one). Exact by full enumeration (vectorised
    over the first r-1 rows) when at most `max_tables` tables exist, otherwise a seeded Monte Carlo
    over the same conditional distribution. Returns (p, method)."""
    ks = [int(k) for k in ks]
    ns = [int(n) for n in ns]
    if len(ks) < 2 or sum(ns) == 0:
        return 1.0, "fewer than two groups"
    K, N = sum(ks), sum(ns)
    if K == 0 or K == N:
        return 1.0, "exact (no variation)"
    from scipy.special import gammaln

    def logp(table: np.ndarray) -> np.ndarray:
        # log prod C(n_i, k_i) / C(N, K) for each row of `table` (one table per row)
        nn = np.asarray(ns, dtype=float)
        t = table.astype(float)
        lc = gammaln(nn + 1) - gammaln(t + 1) - gammaln(nn - t + 1)
        return lc.sum(axis=1) - (gammaln(N + 1) - gammaln(K + 1) - gammaln(N - K + 1))

    obs = logp(np.asarray([ks]))[0]
    count = 1
    for n in ns[:-1]:
        count *= (n + 1)
    if count <= max_tables:
        grids = np.meshgrid(*[np.arange(n + 1) for n in ns[:-1]], indexing="ij")
        first = np.stack([g.ravel() for g in grids], axis=1)
        last = K - first.sum(axis=1)
        ok = (last >= 0) & (last <= ns[-1])
        tables = np.column_stack([first[ok], last[ok]])
        lp = logp(tables)
        p = float(np.exp(lp[lp <= obs + 1e-7]).sum())
        return min(1.0, p), "exact (Freeman-Halton, full enumeration)"
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_hypergeometric(np.asarray(ns), K, size=mc_reps)
    lp = logp(draws)
    p = float((1 + np.sum(lp <= obs + 1e-7)) / (mc_reps + 1))
    return p, f"Monte Carlo Freeman-Halton ({mc_reps} draws, seed {seed})"
