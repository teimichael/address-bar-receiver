"""The machine-readable pre-registration (configs/prereg.yaml), read at analysis time.

Every threshold, class list, cell, contrast, Holm family and bootstrap setting used by
`analysis/claims.py` comes from this file; nothing is hard-coded in the analysis. The file's
sha256 is recorded next to every claim so a report can be matched to the exact rules it applied
(and, via the run manifests' `prereg_sha256`, to the rules in force when the data were produced).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREREG = REPO_ROOT / "configs" / "prereg.yaml"
# Every confirmatory config sets `environment: realistic` (no infrastructure name the agent can read
# says "test"); used only if the pre-registration does not name its environment itself.
DEFAULT_CONFIRMATORY_ENVIRONMENT = "realistic"

UNION_RULE = "union_cluster_bootstrap_and_episode_level"
CLUSTER_BOOTSTRAP = "cluster_bootstrap"
INTERVAL_RULES = (UNION_RULE, CLUSTER_BOOTSTRAP)

_TEST_RE = re.compile(r"H0:\s*[^<>=]*?\s*(<=|>=|<|>)\s*(-?[0-9]*\.?[0-9]+)")
_EXCL_RE = re.compile(r">\s*([0-9]*\.?[0-9]+)\s*%\s*exclusions")
_GATE_RE = re.compile(r"wilson_lower_(\d+)")


@dataclass(frozen=True)
class Prereg:
    raw: dict
    path: str
    sha256: str

    # ------------------------------------------------------------ global settings
    @property
    def version(self):
        return self.raw.get("version")

    @property
    def frozen(self) -> bool:
        return bool(self.raw.get("frozen", False))

    @property
    def alpha(self) -> float:
        return float(self.raw.get("alpha", 0.05))

    @property
    def ci_level(self) -> float:
        return float(self.raw.get("ci", 0.95))

    @property
    def ci_alpha(self) -> float:
        """Two-sided error rate of every reported confidence interval."""
        return round(1.0 - self.ci_level, 10)

    @property
    def bootstrap(self) -> dict:
        b = dict(self.raw.get("bootstrap") or {})
        return {"cluster": b.get("cluster", "brand"), "reps": int(b.get("reps", 4000)),
                "seed": int(b.get("seed", 0))}

    @property
    def interval_rule(self) -> str:
        """How a confirmatory interval is formed (`interval_rule:`). v2.1:
        `union_cluster_bootstrap_and_episode_level` = (min lower, max upper) of the brand-cluster
        bootstrap interval and the episode-level interval of the same estimand. Absent (v2 before
        the amendment): the bootstrap interval, exact/Katz only when the bootstrap is degenerate."""
        rule = str(self.raw.get("interval_rule") or CLUSTER_BOOTSTRAP)
        if rule not in INTERVAL_RULES:
            raise ValueError(f"unsupported interval_rule {rule!r} (known: {', '.join(INTERVAL_RULES)})")
        return rule

    @property
    def pilot_seen(self) -> list[str]:
        return list(self.raw.get("pilot_seen") or [])

    @property
    def confirmatory_environment(self) -> str:
        """The stimulus environment the confirmatory claims apply to (`stimulus_environment:` in
        prereg.yaml; the confirmatory configs' environment otherwise). Claims are computed per
        environment and never pooled over environments."""
        return str(self.raw.get("stimulus_environment") or self.raw.get("confirmatory_environment")
                   or DEFAULT_CONFIRMATORY_ENVIRONMENT)

    @property
    def amendments(self) -> list[dict]:
        """Post-pilot, pre-freeze amendments (`amendments:`), each with what had been seen."""
        return [dict(a) for a in (self.raw.get("amendments") or []) if isinstance(a, dict)]

    # ------------------------------------------------------------ classes / hypotheses
    def classes(self, ref) -> list[str]:
        """Resolve a class reference: a named list in `classes:` (e.g. 'deceptive'), an explicit
        list, or a single class name."""
        named = self.raw.get("classes") or {}
        if isinstance(ref, (list, tuple)):
            out: list[str] = []
            for r in ref:
                out += self.classes(r)
            return out
        if isinstance(ref, str) and ref in named:
            return list(named[ref])
        return [str(ref)]

    @property
    def hypotheses(self) -> dict[str, dict]:
        return dict(self.raw.get("hypotheses") or {})

    def hyp(self, hid: str) -> dict:
        return dict(self.hypotheses.get(hid) or {})

    @property
    def families(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in (self.raw.get("holm_families") or {}).items()}

    def family_of(self, hid: str) -> str | None:
        for fam, ids in self.families.items():
            if hid in ids:
                return fam
        return self.hyp(hid).get("family")

    @property
    def gate(self) -> dict:
        return dict(self.raw.get("competence_gate") or {})

    def gate_level(self) -> float:
        """CI level of the gate statistic, parsed from its name (wilson_lower_90 -> 0.90)."""
        m = _GATE_RE.search(str(self.gate.get("statistic", "")))
        if not m:
            raise ValueError(f"unsupported competence-gate statistic {self.gate.get('statistic')!r}")
        return int(m.group(1)) / 100.0

    def test_null(self, hid: str) -> tuple[str, float] | None:
        """(operator, value) of the hypothesis' `test: "one-sided H0: X <= v ..."` line."""
        t = self.hyp(hid).get("test")
        if not t:
            return None
        m = _TEST_RE.search(str(t))
        return (m.group(1), float(m.group(2))) if m else None

    @property
    def exclusion_threshold(self) -> float:
        """Per-cell exclusion-rate flag, parsed from `exclusions.instrument` ("> 5% exclusions")."""
        text = str((self.raw.get("exclusions") or {}).get("instrument", ""))
        m = _EXCL_RE.search(text)
        return float(m.group(1)) / 100.0 if m else 0.05

    def descriptive(self) -> list[str]:
        return list(self.raw.get("descriptive") or [])

    def summary(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "version": self.version, "frozen": self.frozen,
                "drafted": self.raw.get("drafted"), "alpha": self.alpha, "ci": self.ci_level,
                "bootstrap": self.bootstrap, "interval_rule": self.interval_rule,
                "confirmatory_environment": self.confirmatory_environment,
                "amendments": [a.get("id") for a in self.amendments]}


def load_prereg(path: str | Path | None = None) -> Prereg:
    p = Path(path) if path else DEFAULT_PREREG
    data = p.read_bytes()
    raw = yaml.safe_load(data.decode())
    if not isinstance(raw, dict) or "hypotheses" not in raw:
        raise ValueError(f"{p} is not a pre-registration file (no `hypotheses:`)")
    try:
        shown = str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        shown = str(p)
    pr = Prereg(raw=raw, path=shown, sha256=hashlib.sha256(data).hexdigest())
    _ = pr.interval_rule  # an unknown rule is an unusable pre-registration, never silently ignored
    return pr
