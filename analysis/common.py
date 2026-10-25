"""Independent paper analyses over released evidence and explicit retained aggregates."""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path

from scipy.stats import beta, binomtest, norm

REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("OB_OUTPUT_ROOT", REPO / "outputs")).resolve()
RESULTS = Path(os.environ.get("OB_EVIDENCE_ROOT", OUTPUT / "input" / "full")).resolve()
REPORT = Path(os.environ.get("OB_REPORT_ROOT", OUTPUT / "report")).resolve()
EXPECTED = REPO / "expected"
TABLES = REPORT / "tables"
GEN = OUTPUT / "generated"
DERIVED = GEN / "derived"
FIGURES = OUTPUT / "figures"
RETAINED_TABLES = {"browser_use", "browser_use_llm"}

# Roster in the order used by every table and figure: Qwen3.5 by size, then Gemma, then Ministral by size.
MODELS = ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "qwen3.5-35b-a3b", "gemma-4-12b", "ministral-3-8b", "ministral-3-14b"]
THINK = ["qwen3.5-9b-think", "qwen3.5-35b-a3b-think"]
SHORT = {
    "qwen3.5-2b": "q2b", "qwen3.5-4b": "q4b", "qwen3.5-9b": "q9b", "qwen3.5-35b-a3b": "q35b",
    "gemma-4-12b": "g12b", "ministral-3-8b": "m8b", "ministral-3-14b": "m14b",
    "qwen3.5-9b-think": "q9bt", "qwen3.5-35b-a3b-think": "q35bt",
}
DISPLAY = {
    "qwen3.5-2b": "Qwen3.5-2B", "qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B", "gemma-4-12b": "Gemma-4-12B", "ministral-3-8b": "Ministral-3-8B",
    "ministral-3-14b": "Ministral-3-14B", "qwen3.5-9b-think": "Qwen3.5-9B (thinking)",
    "qwen3.5-35b-a3b-think": "Qwen3.5-35B-A3B (thinking)",
}
FAMILY = {m: ("qwen" if m.startswith("qwen") else "gemma" if m.startswith("gemma") else "ministral") for m in SHORT}

A1_E2 = ["typosquat", "combosquat", "idn"]            # E2 and E2-S lookalike classes (no subdomain)
A1_E3 = ["typosquat", "combosquat", "idn", "subdomain"]  # E1, E3, E5 lookalike classes
A2 = ["a2_iframe", "a2_action"]
DEFENSES_E3 = ["D1", "D2", "D3", "D3info", "D3push", "D5r", "D6r"]
EXP_DIRS = ["e1_default", "e2_information", "e2s_screenshot", "e3_enforcement", "e3_utility", "e5_parity", "e6_thinking"]
SEED_QUOTES = 20261016
Z95 = norm.ppf(0.975)


# data access
@lru_cache(maxsize=None)
def episodes(exp: str) -> tuple:
    """Complete episode records of one experiment directory, as a tuple of dicts."""
    out = []
    with open(RESULTS / exp / "episodes.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("status") == "complete":
                out.append(r)
    return tuple(out)


def transcripts(exp: str):
    """Yield transcript records (episode_id, steps[...]) of one experiment directory."""
    with gzip.open(RESULTS / exp / "transcripts.jsonl.gz", "rt") as fh:
        for line in fh:
            yield json.loads(line)


@lru_cache(maxsize=None)
def table(name: str) -> tuple:
    root = EXPECTED / "report" / "tables" if name in RETAINED_TABLES else TABLES
    with open(root / f"{name}.csv", newline="") as fh:
        return tuple(csv.DictReader(fh))


@lru_cache(maxsize=None)
def claims() -> dict:
    return json.loads((REPORT / "claims.json").read_text())


def hyp(hid: str) -> dict:
    return claims()["by_environment"]["realistic"]["hypotheses"][hid]


def f(r: dict, key: str):
    """Shortcut for the factor of an episode record."""
    return r["factors"][key]


def sel(recs, **kw):
    """Filter episode records by factor values; a list value means membership."""
    out = []
    for r in recs:
        ok = True
        for k, v in kw.items():
            x = r["factors"].get(k)
            if isinstance(v, (list, tuple, set)):
                if x not in v:
                    ok = False
                    break
            elif x != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def kn(recs, outcome: str) -> tuple[int, int]:
    return sum(1 for r in recs if r["outcome"].get(outcome)), len(recs)


# statistics
def wilson(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return max(0.0, c - h), min(1.0, c + h)


def wilson_lower_one_sided(k: float, n: float, alpha: float) -> float:
    """One-sided (1 - alpha) Wilson lower bound."""
    return wilson(k, n, conf=1 - 2 * alpha)[0]


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def newcombe(k1: int, n1: int, k2: int, n2: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid score interval (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, conf)
    l2, u2 = wilson(k2, n2, conf)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    m = only_a + only_b
    return 1.0 if m == 0 else float(binomtest(only_a, m, 0.5).pvalue)


def union(*intervals) -> tuple[float, float]:
    los = [x[0] for x in intervals if x is not None and not _nan(x[0])]
    his = [x[1] for x in intervals if x is not None and not _nan(x[1])]
    return (min(los) if los else math.nan, max(his) if his else math.nan)


def _nan(x) -> bool:
    try:
        return x is None or x == "" or math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def num(x):
    """CSV cell to float (NaN for empty)."""
    return math.nan if _nan(x) else float(x)


# formatting
def _q(x: float, places: int) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def pct_places(p: float) -> int:
    """Integer percent unless the rate is below 10% and above 0, then one decimal."""
    v = 100 * p
    return 1 if 0 < v < 9.95 else 0


def point_ci_places(p: float, lo: float, hi: float) -> tuple[int, int]:
    """Decimals for a rate and its interval: integers, except one decimal for small rates and for intervals of a
    zero or small rate (so that 0/1,008 prints as 0% [0, 0.4] rather than 0% [0, 0])."""
    point = 0 if p in (0.0, 1.0) else pct_places(p)
    ci = 1 if (100 * hi < 9.95 or 0 < 100 * p < 9.95) else 0
    return point, max(point, ci)


def fmt_pct(p: float, places: int | None = None, sign: bool = True) -> str:
    if _nan(p):
        raise ValueError("NaN rate")
    places = pct_places(p) if places is None else places
    s = str(_q(100 * p, places))
    return s + (r"\%" if sign else "")


def fmt_pct_ci(lo: float, hi: float, places: int) -> str:
    return "[" + fmt_pct(lo, places, False) + ",~" + fmt_pct(hi, places, False) + "]"


def fmt_dec(x: float, places: int = 2, signed: bool = False) -> str:
    if _nan(x):
        raise ValueError("NaN value")
    s = str(_q(x, places))
    if s.startswith("-"):
        s = "$-$" + s[1:]
    elif signed and float(s) > 0:
        s = "+" + s
    return s


def fmt_dec_ci(lo: float, hi: float, places: int = 2) -> str:
    return "[" + fmt_dec(lo, places) + ",~" + fmt_dec(hi, places) + "]"


def fmt_pp(d: float, places: int | None = None) -> str:
    """Signed difference of rates in percentage points."""
    v = 100 * d
    places = (1 if abs(v) < 9.95 and v != 0 else 0) if places is None else places
    s = str(_q(v, places))
    if s.startswith("-"):
        return "$-$" + s[1:]
    return ("+" + s) if float(s) > 0 else s


def fmt_pp_ci(lo: float, hi: float, places: int) -> str:
    def one(v):
        s = str(_q(100 * v, places))
        return "$-$" + s[1:] if s.startswith("-") else s
    return "[" + one(lo) + ",~" + one(hi) + "]"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return r"$p<0.001$"
    digits = max(2, -int(math.floor(math.log10(p))) + 1)
    return r"$p=" + str(_q(p, digits)) + "$"


def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_derived(name: str) -> list[dict]:
    with open(DERIVED / f"{name}.csv", newline="") as fh:
        return list(csv.DictReader(fh))
