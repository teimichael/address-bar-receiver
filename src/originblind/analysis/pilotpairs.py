"""Paired pilot checks: a NON-pre-registered pilot run that repeats some plan items of a
pre-registered experiment with ONE setting changed, compared item by item with that experiment.

Declared checks (`PAIRED_PILOT_CHECKS`, variant experiment -> base experiment):

  e3_legend vs e3_enforcement   configs/pilot/pv2r_e3_legend.yaml re-runs the D3info/D3push cells of
                                pv2r_e3_enforcement (same master seed -> the same items, brands,
                                lookalike hosts and canaries; episode ids identical) with
                                `push_legend: true`: the system prompt describes the pushed ORIGIN
                                FACTS/CHECK section exactly as the D3 tool is described (prereg
                                amendment v2.1, change "push_legend"). It measures that change before
                                freezing; it is descriptive and never enters a confirmatory claim.

Method (identical to the lab-vs-realistic check in analysis/envcheck, whose machinery it uses):
records are matched by episode_id within one stimulus environment (complete records only, first
complete per id); the base experiment is restricted to the variant's cells (models, arms, defenses,
flows, drivers). Per model x arm x defense x class unit (exact | A1 | A2 | deceptive | each class):
completion on exact (R_bound), leak elsewhere, abstention everywhere; the paired difference
variant - base with a 95% brand-cluster bootstrap interval (pooled rows: equal-weight mean over
models, brands resampled within model), the discordant pairs and the exact McNemar p.

The design identity is verified, not assumed: the two run configurations must agree on everything
except name, models and defenses (master seed included); the keys that differ (e.g. push_legend)
are reported as the manipulated setting(s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from originblind.analysis.envcheck import GROUP_UNITS, _md, pair_frames, paired_table, paired_tex, spec_diff, spec_of
from originblind.analysis.load import ENVIRONMENT_ORDER, ExperimentData, order_values

PAIRED_PILOT_CHECKS: tuple[tuple[str, str], ...] = (("e3_legend", "e3_enforcement"),)
BASE, VARIANT = "base", "variant"  # column suffixes of the two conditions (difference = variant - base)
RESTRICT = ("model", "arm", "defense", "flow", "driver")
DESIGN_FREE = ("models", "defenses")  # may differ: the variant re-runs a subset of the base's cells

__all__ = ["PAIRED_PILOT_CHECKS", "PairedPilot", "paired_pilot_checks", "pilotpairs_markdown", "write_pilotpairs"]


@dataclass
class PairedPilot:
    experiment: str  # the variant (non-pre-registered) experiment
    base: str  # the pre-registered experiment it repeats
    environment: str
    variant_dirs: list[str]
    base_dirs: list[str]
    identity: list[dict]
    manipulated: dict  # setting -> {"base": value, "variant": value}
    pairs: pd.DataFrame
    paired: pd.DataFrame
    unpaired: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"pilotpair_{self.experiment}" + ("" if self.environment == "realistic" else f"_{self.environment}")

    def to_json(self) -> dict:
        prim = self.paired[self.paired["primary"]] if len(self.paired) else self.paired
        return {"experiment": self.experiment, "base": self.base, "environment": self.environment,
                "difference": f"{VARIANT} ({self.experiment}) - {BASE} ({self.base})",
                "variant_dirs": self.variant_dirs, "base_dirs": self.base_dirs, "identity": self.identity,
                "manipulated": self.manipulated, "n_pairs": int(len(self.pairs)),
                "unpaired": self.unpaired.to_dict(orient="records"), "paired_rows": int(len(self.paired)),
                "primary_rows": prim.to_dict(orient="records"), "notes": self.notes}


def _dirs(exps: dict[str, ExperimentData], experiment: str, env: str) -> list[ExperimentData]:
    return [e for _, e in sorted(exps.items()) if e.experiment == experiment and env in e.environments]


def _identity(variants: list[ExperimentData], bases: list[ExperimentData]) -> tuple[list[dict], dict]:
    rows, manipulated = [], {}
    for ev in variants:
        for eb in bases:
            sv, sb = spec_of(ev), spec_of(eb)
            if sv is None or sb is None:
                rows.append({"dirs": [eb.name, ev.name], "identical_design": None, "differing_keys": [],
                             "note": "a run configuration is missing from manifest.json"})
                continue
            diff = spec_diff(sb, sv, drop=DESIGN_FREE)
            for k in diff:
                manipulated.setdefault(k, {"base": sb.get(k), "variant": sv.get(k)})
            rows.append({"dirs": [eb.name, ev.name], "master_seed_equal": sv.get("master_seed_hex") ==
                         sb.get("master_seed_hex"), "differing_keys": diff,
                         "variant_models": list(sv.get("models") or []), "variant_defenses": list(sv.get("defenses") or [])})
    return rows, manipulated


def paired_pilot_checks(df: pd.DataFrame, exps: dict[str, ExperimentData], classes: dict[str, list[str]], *,
                        alpha: float = 0.05, reps: int = 4000, seed: int = 0,
                        model_order: list[str] | None = None) -> list[PairedPilot]:
    """Every declared check whose variant experiment is in `df` (per stimulus environment)."""
    out: list[PairedPilot] = []
    if df is None or df.empty:
        return out
    for exp, base in PAIRED_PILOT_CHECKS:
        envs = order_values(set(df.loc[df["experiment"] == exp, "environment"]), ENVIRONMENT_ORDER)
        for env in envs:
            v = df[(df["experiment"] == exp) & (df["environment"] == env)]
            b = df[(df["experiment"] == base) & (df["environment"] == env)]
            notes: list[str] = []
            for col in RESTRICT:
                b = b[b[col].isin(set(v[col]))]
            identity, manipulated = _identity(_dirs(exps, exp, env), _dirs(exps, base, env))
            if not manipulated:
                notes.append("no run-configuration key differs between the two experiments (beyond models/defenses)")
            if any(i.get("master_seed_equal") is False for i in identity):
                notes.append("master seeds differ: the matched episode ids are not the same stimuli")
            if b.empty:
                notes.append(f"no {base} episodes in the variant's cells in environment {env}: nothing to pair")
                pairs = pd.DataFrame()
            else:
                pairs = pair_frames(b, v, BASE, VARIANT, ["episode_id"])
                pairs.insert(0, "experiment", exp)
            paired = (paired_table(pairs, classes, alpha=alpha, reps=reps, seed=seed, model_order=model_order,
                                   a=BASE, b=VARIANT, seed_label=f"pilotpair::{exp}::{env}")
                      if len(pairs) else pd.DataFrame())
            rows = []
            for m in sorted(set(v["model"]) | set(b["model"])):
                ib = set(b.loc[b["model"] == m, "episode_id"])
                iv = set(v.loc[v["model"] == m, "episode_id"])
                rows.append({"model": m, f"complete_{BASE}": len(ib), f"complete_{VARIANT}": len(iv),
                             "pairs": len(ib & iv), f"only_{BASE}": len(ib - iv), f"only_{VARIANT}": len(iv - ib)})
            out.append(PairedPilot(exp, base, env, [e.name for e in _dirs(exps, exp, env)],
                                   [e.name for e in _dirs(exps, base, env)], identity, manipulated, pairs, paired,
                                   pd.DataFrame(rows), notes))
    return out


def _f(x, nd: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if math.isnan(v) else f"{v:.{nd}f}"


def _show(v) -> str:
    if v is None:
        return "absent (default)"
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


def pilotpairs_markdown(checks: list[PairedPilot]) -> str:
    from originblind.analysis.tables import fmt_kn

    L: list[str] = []
    for c in checks:
        man = "; ".join(f"`{k}`: {_show(v['base'])} -> {_show(v['variant'])}"
                        for k, v in sorted(c.manipulated.items())) or "--"
        L += [f"## Paired pilot check: `{c.experiment}` vs `{c.base}` (environment `{c.environment}`; not "
              "pre-registered, descriptive)", "",
              f"The variant ({', '.join(c.variant_dirs)}) repeats plan items of the base ({', '.join(c.base_dirs)}) "
              f"with {man}. Records matched by episode_id; difference = variant - base, 95% brand-cluster "
              f"bootstrap; exact two-sided McNemar on the discordant pairs. Pairs: {len(c.pairs)}.", ""]
        for i in c.identity:
            L.append(f"- design identity ({' vs '.join(i['dirs'])}): master seed equal = {i.get('master_seed_equal')}, "
                     f"differing keys {i.get('differing_keys')}")
        for n in c.notes:
            L.append(f"- note: {n}")
        L += ["", _md(c.unpaired), ""]
        if len(c.paired):
            prim = c.paired[c.paired["primary"] & c.paired["unit"].isin(GROUP_UNITS)]
            ab = c.paired[(~c.paired["primary"]) & c.paired["unit"].isin(GROUP_UNITS)].set_index(
                ["model", "arm", "defense", "unit"])
            rows = []
            for _, r in prim.iterrows():
                key = (r["model"], r["arm"], r["defense"], r["unit"])
                a = ab.loc[key] if key in ab.index else None
                rows.append({"model": r["model"], "defense": r["defense"], "classes": r["unit"],
                             "outcome": {"r_bound": "completion"}.get(r["outcome"], r["outcome"]), "pairs": r["n_pairs"],
                             BASE: fmt_kn(r[f"k_{BASE}"], r["n_pairs"]), VARIANT: fmt_kn(r[f"k_{VARIANT}"], r["n_pairs"]),
                             "diff [95%]": f"{_f(r['diff'])} [{_f(r['diff_lo'])}, {_f(r['diff_hi'])}]",
                             "disc. base/variant": f"{r[f'only_{BASE}']}/{r[f'only_{VARIANT}']}",
                             "McNemar p": f"{r['mcnemar_p']:.3g}",
                             "abstain base/variant": "--" if a is None else f"{int(a[f'k_{BASE}'])}/{int(a[f'k_{VARIANT}'])}"})
            L += ["Primary outcome per class group (completion on exact, leak elsewhere; every class and abstention: "
                  f"`tables/{c.name}.csv`):", "", _md(pd.DataFrame(rows)), ""]
    return "\n".join(L)


def write_pilotpairs(checks: list[PairedPilot], out_tables) -> list[str]:
    from originblind.analysis.tables import write_table

    files: list[str] = []
    for c in checks:
        files += [p.name for p in write_table(c.paired, out_tables / c.name, tex=paired_tex(c.paired, BASE, VARIANT),
                                              comment=f"paired pilot check {c.experiment} vs {c.base} "
                                                      f"({c.environment}); variant - base; 95% brand-cluster "
                                                      "bootstrap; exact McNemar")]
        files += [p.name for p in write_table(c.unpaired, out_tables / f"{c.name}_unpaired")]
    return files
