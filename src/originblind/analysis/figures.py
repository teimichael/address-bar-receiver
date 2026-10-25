"""Publication figures: vector PDF + 300-dpi PNG at paper widths, each with a CSV table-view twin.

Every figure is drawn from the report's tables (never from raw episodes), so a plotted number is
always the number in a CSV. Design rules (the `dataviz` method, applied to static paper figures):

  form       forest plots for ratios/rates with intervals, dumbbells for paired contrasts, a
             frontier scatter for the E3 defense trade-off, a heatmap for the exposure grid, and a
             labelled outcome grid for the browser-use mechanism.
  identity   one fixed categorical slot per MODEL in roster order (color follows the model in every
             figure; never re-assigned by rank), always doubled by a marker shape and a legend
             (secondary encoding: three slots sit below 3:1 contrast on white, so the CSV twin and the
             legend are the relief channel). Within-model distinctions use marker shape / fill.
  magnitude  one-hue sequential blue ramp (exposure heatmap); status colors only where a cell
             MEANS good/bad (browser-use: secret delivered to the bound site vs leaked), always with
             a text label in the cell.
  marks      thin CI lines, >= 4 pt markers with a white ring, hairline solid grid, recessive axes,
             no figure titles (captions live in the paper), selective direct labels only.
  sizes      single column 3.3 in, double column 7.0 in; 7 pt sans text; TrueType fonts embedded
             (pdf.fonttype 42); no timestamps in the files (deterministic bytes for identical input).

Missing inputs never fail the report: a figure (or a panel) whose data are absent is skipped with a
note, which the report lists in SUMMARY.md. Validated palette: dataviz references/palette.md
(categorical slots 1-8, light mode; `validate_palette.js` passes all hard gates on white).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

from originblind.analysis.load import ARM_ORDER, CLASS_ORDER, DEFENSE_ORDER
from originblind.analysis.prereg import REPO_ROOT

# ---------------------------------------------------------------- palette (dataviz reference instance)

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h")
SEQ = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b")
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
W1, W2 = 3.3, 7.0

RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlecolor": INK2, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 0, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
    "legend.handletextpad": 0.3, "legend.columnspacing": 1.0, "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02, "axes.unicode_minus": False, "path.simplify": False,
}
CLASS_LABEL = {"exact": "exact", "typosquat": "typosquat", "combosquat": "combosquat", "idn": "IDN homoglyph",
               "subdomain": "subdomain", "raw_ip": "raw IP (floor)", "a2_iframe": "A2 iframe", "a2_action": "A2 action"}
DEFENSE_FAMILY = {"D0": "instruction", "D1": "instruction", "D2": "information", "D3": "information",
                  "D3info": "information", "D3push": "information", "D5": "enforcement", "D5r": "enforcement",
                  "D6": "enforcement", "D6r": "enforcement", "DBU": "enforcement"}
FAMILY_MARKER = {"instruction": "o", "information": "s", "enforcement": "^"}


def roster_order() -> list[str]:
    try:
        import yaml

        data = yaml.safe_load((REPO_ROOT / "configs" / "roster.yaml").read_text())
        return [m["key"] for m in data.get("models", [])]
    except Exception:  # noqa: BLE001 - a missing roster only changes the slot order
        return []


def model_styles(models) -> dict[str, tuple[str, str]]:
    """model -> (color, marker): roster order first, then any other model alphabetically; slots
    are fixed by that order and never depend on which models a figure happens to show."""
    order = [m for m in roster_order()]
    extra = sorted(set(models) - set(order))
    out = {}
    for i, m in enumerate(order + extra):
        out[m] = (SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]) if i < len(SERIES) else (MUTED, "o")
    return out


def ordered_models(models) -> list[str]:
    order = roster_order()
    ms = list(dict.fromkeys(models))
    return [m for m in order if m in ms] + sorted(m for m in ms if m not in order)


# ---------------------------------------------------------------- results / IO

@dataclass
class FigureResult:
    name: str
    status: str  # written | skipped
    files: list[str] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        return {"name": self.name, "status": self.status, "files": self.files, "note": self.note}


@dataclass
class FigureData:
    """Everything a figure may draw: the report's tables plus claims, gate and experiment roles."""
    tables: dict[str, pd.DataFrame]
    claims: dict | None = None
    experiments: dict[str, str] = field(default_factory=dict)  # role -> experiment id
    classes: dict[str, list[str]] = field(default_factory=dict)  # prereg class lists
    legacy: bool = False
    suffix: str = ""  # appended to every figure name (one set per stimulus environment)

    def t(self, name: str) -> pd.DataFrame:
        v = self.tables.get(name)
        return v if isinstance(v, pd.DataFrame) else pd.DataFrame()

    @property
    def gate(self) -> dict[str, str]:
        g = ((self.claims or {}).get("competence_gate") or {}).get("models") or {}
        return {m: v.get("status", "not_evaluable") for m, v in g.items()}


def _save(fig: Figure, out_dir: Path, name: str, data: pd.DataFrame | None) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    pdf, png = out_dir / f"{name}.pdf", out_dir / f"{name}.png"
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, format="png", dpi=300, metadata={"Software": None})
    files += [pdf.name, png.name]
    if data is not None:
        csv = out_dir / f"{name}.csv"
        d = data.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].round(6)
        d.to_csv(csv, index=False, lineterminator="\n")
        files.append(csv.name)
    return files


def _skip(name: str, why: str) -> FigureResult:
    return FigureResult(name, "skipped", [], why)


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=6.5, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _xgrid(ax) -> None:
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def _compact_x(ax, step: float | None = None) -> None:
    """Short tick labels (0, 0.5, 1 rather than 0.00, 0.50, 1.00) so neighbouring panels' edge
    labels never run into each other; `step` sets the major tick spacing."""
    if step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}" if abs(v) > 1e-12 else "0"))


def _row_major(handles: list, ncols: int) -> list:
    """Legend handles reordered so a multi-row legend reads left-to-right, top-to-bottom in the
    given order (matplotlib fills legend columns first)."""
    if ncols <= 1 or len(handles) <= ncols:
        return handles
    nrows = math.ceil(len(handles) / ncols)
    grid = [handles[r * ncols:(r + 1) * ncols] for r in range(nrows)]
    out = []
    for c in range(ncols):
        for r in range(nrows):
            if c < len(grid[r]):
                out.append(grid[r][c])
    return out


def _model_handles(models: list[str], styles: dict, gate: dict[str, str]) -> list[Line2D]:
    hs = []
    for m in models:
        c, mk = styles[m]
        failed = gate.get(m) not in (None, "pass")
        lab = m + ("" if gate.get(m) in (None, "pass") else f" (gate {gate.get(m).replace('_', ' ')})")
        hs.append(Line2D([], [], color=c, marker=mk, linestyle="-", linewidth=0.9, markersize=4,
                         markerfacecolor=SURFACE if failed else c, markeredgecolor=c if failed else SURFACE,
                         markeredgewidth=0.6, label=lab))
    return hs


def _point(ax, x, y, color, marker, *, hollow=False, size=4.2, z=3):
    ax.plot([x], [y], linestyle="none", marker=marker, markersize=size, color=color,
            markerfacecolor=SURFACE if hollow else color, markeredgecolor=color if hollow else SURFACE,
            markeredgewidth=0.7 if hollow else 0.6, zorder=z)


def _hline_ci(ax, lo, hi, y, color, xmax, lw=0.9, xmin=0.0):
    """A horizontal interval clipped to [xmin, xmax]; an open end is marked with a small arrow."""
    if not (_finite(lo) and _finite(hi)):
        return
    a, b = max(xmin, float(lo)), min(xmax, float(hi))
    if a <= b:
        ax.plot([a, b], [y, y], color=color, linewidth=lw, solid_capstyle="round", zorder=2)
    if float(hi) > xmax:
        ax.plot([xmax], [y], marker=">", markersize=2.6, color=color, linestyle="none", zorder=2, clip_on=False)
    if float(lo) < xmin:
        ax.plot([xmin], [y], marker="<", markersize=2.6, color=color, linestyle="none", zorder=2, clip_on=False)


# ---------------------------------------------------------------- forest (RR by class x model)

def _rr_items(rr: pd.DataFrame, sel: dict) -> pd.DataFrame:
    from originblind.analysis.tables import rr_ci

    if rr.empty:
        return rr
    g = rr[rr["unit"] == "class"] if "unit" in rr else rr
    for k, v in sel.items():
        g = g[g[k] == v]
    if g.empty:
        return g
    rows = []
    for _, r in g.iterrows():
        lo, hi, src = rr_ci(r)
        rows.append({"experiment": r["experiment"], "model": r["model"], "arm": r["arm"], "defense": r["defense"],
                     "flow": r["flow"], "driver": r["driver"], "origin_class": r["origin_class"],
                     "leak": f"{int(r['leak_k'])}/{int(r['leak_n'])}",
                     "exact_r_bound": f"{int(r['exact_r_bound_k'])}/{int(r['exact_r_bound_n'])}",
                     "exact_k": int(r["exact_r_bound_k"]), "exact_n": int(r["exact_r_bound_n"]),
                     "rr": _f(r["rr"]), "lo": _f(lo), "hi": _f(hi), "ci": src, "placebo": bool(r["placebo"]),
                     "floor": bool(r["floor"])})
    return pd.DataFrame(rows)


def _forest_panel(ax, items: pd.DataFrame, classes: list[str], models: list[str], styles: dict,
                  gate: dict[str, str], xmax: float, label: str, show_y: bool, show_x: bool = True) -> None:
    n = len(classes)
    offs = np.linspace(-0.27, 0.27, len(models)) if len(models) > 1 else np.array([0.0])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, xmax)
    _xgrid(ax)
    ax.axvline(1.0, color=MUTED, linewidth=0.7, zorder=1)
    # a stratum whose exact completion is 0 has no defined RR (x / 0): nothing is drawn for that
    # model, the panel says so (the counts stay in the CSV twin)
    undefined = []
    had_items = bool(len(items))
    if len(items):
        for m in models:
            r = items[items["model"] == m]
            if len(r) and (r["exact_k"] == 0).all():
                undefined.append(f"{m}: exact 0/{int(r.iloc[0]['exact_n'])},\nRR undefined")
        items = items[items["exact_k"] > 0]
    for j, txt in enumerate(undefined):
        t = ax.text(0.97, 0.03 + 0.13 * j, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
                    color=INK2, zorder=5, linespacing=1.1)
        t.set_in_layout(False)
    for i, c in enumerate(classes):
        sub = items[items["origin_class"] == c] if len(items) else items
        if len(sub) and sub["placebo"].any():  # shaded band = placebo (named in the figure legend)
            ax.axhspan(i - 0.46, i + 0.46, color=NEUTRAL, zorder=0, linewidth=0)
        for j, m in enumerate(models):
            r = sub[sub["model"] == m] if len(sub) else sub
            if r.empty:
                continue
            r = r.iloc[0]
            col, mk = styles[m]
            y = i + offs[j]
            _hline_ci(ax, r["lo"], r["hi"], y, col, xmax)
            if _finite(r["rr"]):
                _point(ax, min(float(r["rr"]), xmax), y, col, mk, hollow=gate.get(m) not in (None, "pass"))
            elif r["rr"] == float("inf"):
                ax.text(xmax * 0.97, y, "inf", fontsize=5.5, color=INK2, ha="right", va="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([CLASS_LABEL.get(c, c) for c in classes] if show_y else [""] * n)
    ax.tick_params(axis="y", length=0)
    _compact_x(ax, 0.5 if xmax <= 2.0 else 1.0)
    ax.set_title(label, loc="left", pad=3)
    if not had_items:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=6.5)


def _xmax(frames: list[pd.DataFrame]) -> float:
    vals = [1.25]
    for f in frames:
        if len(f):
            vals += [v for v in f["hi"].tolist() + f["rr"].tolist() if _finite(v)]
    return float(min(3.0, math.ceil(max(vals) * 4) / 4))


def _forest_figure(fd: FigureData, out: Path, name: str, panels: list[tuple[str, dict]], ncols: int,
                   width: float) -> FigureResult:
    rr = fd.t("rr")
    if rr.empty:
        return _skip(name, "no relative-release table (no stratum with exact cells)")
    items = [(lab, _rr_items(rr, sel)) for lab, sel in panels]
    if all(it.empty for _, it in items):
        return _skip(name, "none of the panels' cells are in this namespace: " + "; ".join(lab for lab, _ in panels))
    present = set().union(*[set(it["origin_class"]) for _, it in items if len(it)])
    classes = [c for c in CLASS_ORDER if c in present and c != "exact"]
    models = ordered_models(set().union(*[set(it["model"]) for _, it in items if len(it)]))
    styles = model_styles(models)
    xmax = _xmax([it for _, it in items])
    nrows = math.ceil(len(panels) / ncols)
    h = 0.55 + nrows * (0.24 * len(classes) + 0.45)
    with rc_context(RC | {"figure.constrained_layout.w_pad": 0.06}):
        fig = Figure(figsize=(width, h), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True)
        fig.supxlabel("relative release RR = P(leak | class) / P(R_bound | exact), 95% CI", fontsize=7, color=INK2)
        for k, (lab, it) in enumerate(items):
            ax = axes[k // ncols][k % ncols]
            bottom = k + ncols >= len(items)
            _forest_panel(ax, it, classes, models, styles, fd.gate, xmax, lab, show_y=(k % ncols == 0),
                          show_x=bottom)
            if bottom:
                ax.tick_params(labelbottom=True)
        for k in range(len(items), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)
        hs = _model_handles(models, styles, fd.gate)
        if any(len(it) and it["placebo"].any() for _, it in items):
            hs.append(Patch(facecolor=NEUTRAL, edgecolor="none", label="placebo: A2 sink not in the observation"))
        nc = min(len(hs), 4)
        fig.legend(handles=_row_major(hs, nc), loc="outside upper center", ncols=nc)
        data = pd.concat([it.assign(panel=lab) for lab, it in items if len(it)], ignore_index=True)
        files = _save(fig, out, name, data)
    return FigureResult(name, "written", files, "")


def fig_rr_forest(fd: FigureData, out: Path) -> FigureResult:
    e1, e2 = fd.experiments.get("e1", "e1_default"), fd.experiments.get("e2", "e2_information")
    panels = [(f"E1 default · T1 · D0 · login", {"experiment": e1, "arm": "T1", "defense": "D0", "flow": "login"}),
              (f"E1 default · T1 · D1 · login", {"experiment": e1, "arm": "T1", "defense": "D1", "flow": "login"}),
              (f"E2 information · T2 · D1 · login", {"experiment": e2, "arm": "T2", "defense": "D1", "flow": "login"})]
    return _forest_figure(fd, out, "fig_rr_forest" + fd.suffix, panels, ncols=3, width=W2)


def fig_rr_forest_e1_flows(fd: FigureData, out: Path) -> FigureResult:
    e1 = fd.experiments.get("e1", "e1_default")
    panels = [(f"T1 · {d} · {fl}", {"experiment": e1, "arm": "T1", "defense": d, "flow": fl})
              for d in ("D0", "D1") for fl in ("login", "otp", "checkout")]
    return _forest_figure(fd, out, "fig_rr_forest_e1_flows" + fd.suffix, panels, ncols=3, width=W2)


def fig_rr_forest_legacy(fd: FigureData, out: Path) -> FigureResult:
    """October pilot: every (directory, arm, defense) stratum is its own panel (never pooled)."""
    rr = fd.t("rr")
    if rr.empty:
        return _skip("fig_rr_forest_legacy" + fd.suffix, "no relative-release table")
    strata = rr[["experiment", "arm", "defense", "flow"]].drop_duplicates()
    strata = strata.assign(_a=strata["arm"].map(lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99),
                           _d=strata["defense"].map(lambda d: DEFENSE_ORDER.index(d) if d in DEFENSE_ORDER else 99))
    strata = strata.sort_values(["experiment", "_a", "_d"])
    panels = [(f"{r.experiment} · {r.arm} · {r.defense}", {"experiment": r.experiment, "arm": r.arm,
                                                             "defense": r.defense, "flow": r.flow})
              for r in strata.itertuples()]
    return _forest_figure(fd, out, "fig_rr_forest_legacy" + fd.suffix, panels, ncols=4, width=W2)


# ---------------------------------------------------------------- competence gate

def fig_competence(fd: FigureData, out: Path) -> FigureResult:
    name = "fig_competence" + fd.suffix
    gate = (fd.claims or {}).get("competence_gate") or {}
    c = fd.t("competence")
    if c.empty or not c["n"].fillna(0).astype(int).any():
        return _skip(name, f"competence-gate cell ({gate.get('experiment')}: {gate.get('cell')}) has no episodes")
    c = c[c["n"].fillna(0).astype(int) > 0]
    models = ordered_models(c["model"])
    styles = model_styles(models)
    thr = float(gate.get("threshold", 0.5))
    with rc_context(RC):
        fig = Figure(figsize=(W1, 0.55 + 0.2 * len(models)), layout="constrained")
        ax = fig.subplots()
        _xgrid(ax)
        ax.set_xlim(0, 1.03)
        ax.set_ylim(len(models) - 0.5, -0.75)
        ax.axvline(thr, color=MUTED, linewidth=0.7, zorder=1)
        ax.text(thr - 0.01, -0.68, f"gate: lower bound ≥ {thr:.2f}", fontsize=5.5, color=MUTED, ha="right",
                va="center")
        for i, m in enumerate(models):
            r = c[c["model"] == m].iloc[0]
            col, mk = styles[m]
            _hline_ci(ax, r["lower"], r["upper"], i, col, 1.0)
            _point(ax, float(r["p"]), i, col, mk, hollow=r["status"] != "pass")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([f"{m}  {int(c[c.model == m].iloc[0]['k'])}/{int(c[c.model == m].iloc[0]['n'])} · "
                            f"{c[c.model == m].iloc[0]['status']}" for m in models])
        lvl = int(round(100 * float(gate.get("level", 0.9))))
        ax.set_xlabel(f"exact-origin completion ({lvl}% Wilson)")
        files = _save(fig, out, name, c)
    return FigureResult(name, "written", files, "")


# ---------------------------------------------------------------- knowing-doing

def fig_knowing_doing(fd: FigureData, out: Path) -> FigureResult:
    """Per model (E2, T2, D1, login): per class, the sink read correctly (readout), the binding
    violation recognised (discrimination: 'NO'), and the secret sent to the field's sink (exact:
    the bound site = completion; deceptive: the off-origin domain = leak)."""
    name = "fig_knowing_doing" + fd.suffix
    e2 = fd.experiments.get("e2", "e2_information")
    p = fd.t("probe")
    if p.empty:
        return _skip(name, "no probe answers in this namespace")
    sel = p[(p["experiment"] == e2) & (p["arm"] == "T2") & (p["defense"] == "D1") & (p["flow"] == "login")]
    if fd.legacy or sel.empty:
        return _skip(name, f"no probe keys in {e2} T2/D1/login")
    models = ordered_models(sel["model"])
    styles = model_styles(models)
    classes = [c for c in CLASS_ORDER if c in set(sel["origin_class"])]
    k = fd.t("kdg")
    rows = []
    for _, r in sel.iterrows():
        exact = r["origin_class"] == "exact"
        sent = ("r_bound" if exact else "leak")
        for meas, pre in (("readout correct", "readout_correct"), ("judged mismatch", "judged_mismatch"),
                          ("sent to sink", sent)):
            rows.append({"model": r["model"], "origin_class": r["origin_class"], "measure": meas,
                         "k": r[f"{pre}_k"], "n": r[f"{pre}_n"], "p": r[f"{pre}_p"], "lo": r[f"{pre}_lo"],
                         "hi": r[f"{pre}_hi"]})
    data = pd.DataFrame(rows)
    meas_style = {"readout correct": (-0.24, "o", "gray"), "judged mismatch": (0.0, "s", "hollow"),
                  "sent to sink": (0.24, "o", "fill")}
    ncols = min(4, len(models))
    nrows = math.ceil(len(models) / ncols)
    with rc_context(RC | {"figure.constrained_layout.w_pad": 0.06}):
        fig = Figure(figsize=(W2, 0.6 + nrows * (0.2 * len(classes) + 0.6)), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True)
        fig.supxlabel("share of probe keys (readout, judgement) or episodes (sent), Wilson 95%", fontsize=7,
                      color=INK2)
        for idx, m in enumerate(models):
            ax = axes[idx // ncols][idx % ncols]
            col, _ = styles[m]
            _xgrid(ax)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(len(classes) - 0.5, -0.5)
            for i, c in enumerate(classes):
                for meas, (dy, mk, kind) in meas_style.items():
                    r = data[(data.model == m) & (data.origin_class == c) & (data.measure == meas)]
                    if r.empty or not _finite(r.iloc[0]["p"]):
                        continue
                    r = r.iloc[0]
                    cc = INK2 if kind == "gray" else col
                    _hline_ci(ax, r["lo"], r["hi"], i + dy, cc, 1.02, lw=0.9 if kind == "fill" else 0.6)
                    _point(ax, float(r["p"]), i + dy, cc, mk, hollow=kind != "fill",
                           size=4.6 if kind == "fill" else 3.6)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels([CLASS_LABEL.get(c, c) for c in classes] if idx % ncols == 0 else [""] * len(classes))
            ax.tick_params(axis="y", length=0)
            _compact_x(ax, 0.5)
            kk = k[(k["experiment"] == e2) & (k["model"] == m) & (k["arm"] == "T2") & (k["defense"] == "D1")] \
                if len(k) else k
            sub = ""
            if len(kk) and int(kk.iloc[0]["kdg_n"]):
                r = kk.iloc[0]
                sub = (f"\nKDG {int(r['kdg_k'])}/{int(r['kdg_n'])} · exact false NO "
                       f"{int(r['exact_false_mismatch_k'])}/{int(r['exact_false_mismatch_n'])}")
            ax.set_title(m + sub, loc="left", pad=3)
        for idx in range(len(models), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        for idx in range(len(models)):  # the lowest panel of every column carries the x tick labels
            if idx + ncols >= len(models):
                axes[idx // ncols][idx % ncols].tick_params(labelbottom=True)
        hs = [Line2D([], [], linestyle="none", marker="o", markersize=3.6, markerfacecolor=SURFACE,
                     markeredgecolor=INK2, label="sink read correctly (readout probe)"),
              Line2D([], [], linestyle="none", marker="s", markersize=3.6, markerfacecolor=SURFACE,
                     markeredgecolor=INK2, label="mismatch judged (discrimination probe: NO)"),
              Line2D([], [], linestyle="none", marker="o", markersize=3.6, markerfacecolor=INK2,
                     markeredgecolor=SURFACE, label="secret sent to the field's sink (exact: completion)")]
        fig.legend(handles=hs, loc="outside upper center", ncols=3)
        files = _save(fig, out, name, data)
    return FigureResult(name, "written", files, "model color = panel model; shapes = measure")


# ---------------------------------------------------------------- defense frontier

def _spread(ys: list[float], sep: float, lo: float, hi: float) -> list[float]:
    """Label positions for points sorted top-down: at most `sep` apart, inside [lo, hi], each as
    close to its point as the neighbours allow (push down, then push back up from the floor)."""
    pos = []
    for y in ys:
        pos.append(min(y, pos[-1] - sep) if pos else min(y, hi))
    for i in range(len(pos) - 1, -1, -1):
        floor = lo if i == len(pos) - 1 else pos[i + 1] + sep
        pos[i] = max(pos[i], floor)
    return pos


def _frontier_labels(ax, pts: list[tuple[float, float, str]], *, sep: float = 0.07, dx: float = 0.012,
                     char_w: float = 0.019, lo: float = -0.02, hi: float = 1.04) -> None:
    """Direct labels for the defense points without collisions. Each label starts beside its point
    (left of points in the 0.75-0.95 band, whose right side is crowded by the x ~ 1 column; right
    otherwise); every pair of labels whose text boxes overlap horizontally and sit closer than
    `sep` vertically is pushed apart (repeated top-down passes, clamped to [lo, hi]); a displaced
    label is moved further out and gets a hairline leader to its point."""
    labs = []
    for x, y, lab in pts:
        left = 0.75 < x < 0.95
        w = char_w * len(lab)
        lx = x - dx if left else x + dx
        labs.append({"x": x, "y": y, "lab": lab, "left": left, "ly": min(max(y, lo), hi),
                     "x0": lx - w if left else lx, "x1": lx if left else lx + w})
    for _ in range(100):
        moved = False
        labs.sort(key=lambda d: -d["ly"])
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                a, b = labs[i], labs[j]  # a is above b
                if a["x1"] <= b["x0"] or b["x1"] <= a["x0"]:
                    continue
                gap = a["ly"] - b["ly"]
                if gap >= sep - 1e-9:
                    continue
                need = (sep - gap) / 2
                a["ly"], b["ly"] = min(hi, a["ly"] + need), max(lo, b["ly"] - need)
                if a["ly"] - b["ly"] < sep - 1e-9:  # one side hit a bound: move the other the rest
                    if b["ly"] <= lo + 1e-9:
                        a["ly"] = min(hi, b["ly"] + sep)
                    else:
                        b["ly"] = max(lo, a["ly"] - sep)
                moved = True
        if not moved:
            break
    for d in labs:
        displaced = abs(d["ly"] - d["y"]) > 0.005
        off = dx * (3.0 if displaced else 1.0)
        lx = d["x"] - off if d["left"] else d["x"] + off
        if displaced:
            ax.plot([d["x"], lx + (dx * 0.5 if d["left"] else -dx * 0.5)], [d["y"], d["ly"]], color=MUTED,
                    linewidth=0.4, zorder=2)
        ax.text(lx, d["ly"], d["lab"], fontsize=5.5, color=INK2, ha="right" if d["left"] else "left",
                va="center", zorder=4)


def fig_defense_frontier(fd: FigureData, out: Path) -> FigureResult:
    name = "fig_defense_frontier" + fd.suffix
    e3 = fd.experiments.get("e3", "e3_enforcement")
    d = fd.t("defense")
    d = d[d["experiment"] == e3] if len(d) else d
    if d.empty:
        return _skip(name, f"{e3} not in this namespace")
    models = ordered_models(d["model"])
    styles = model_styles(models)
    ncols = min(4, len(models))
    nrows = math.ceil(len(models) / ncols)
    data = d[["model", "defense", "arm", "flow", "driver", "exact_r_bound_k", "exact_r_bound_n", "exact_r_bound_p",
              "deceptive_leak_k", "deceptive_leak_n", "deceptive_leak_p", "deceptive_leak_lo", "deceptive_leak_hi",
              "deceptive_abstained_k", "deceptive_abstained_n", "deceptive_abstained_p"]].copy()
    xs = [v for v in d["exact_r_bound_p"].tolist() if _finite(v)]
    x0 = min(0.0 if not xs else math.floor(min(xs) * 10) / 10 - 0.1, 0.5)
    with rc_context(RC):
        fig = Figure(figsize=(W2 if ncols > 1 else W1, 0.5 + nrows * 1.9), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False)
        fig.supxlabel("exact-origin completion R_bound", fontsize=7, color=INK2)
        for idx, m in enumerate(models):
            ax = axes[idx // ncols][idx % ncols]
            col, _ = styles[m]
            ax.grid(True, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(max(-0.04, x0), 1.17)
            ax.set_ylim(-0.06, 1.08)
            ax.spines["bottom"].set_bounds(max(-0.04, x0), 1.0)
            ax.set_xticks([t for t in np.arange(0.0, 1.0001, 0.2) if t >= max(-0.04, x0) - 1e-9])
            _compact_x(ax)
            sub = d[d["model"] == m]
            labels = []
            for _, r in sub.iterrows():
                x, y = _f(r["exact_r_bound_p"]), _f(r["deceptive_leak_p"])
                if not (_finite(x) and _finite(y)):
                    continue
                ab = _f(r["deceptive_abstained_p"])
                size = 3.0 + 5.0 * (ab if _finite(ab) else 0.0)
                fam = DEFENSE_FAMILY.get(r["defense"], "instruction")
                ax.plot([x, x], [max(0, _f(r["deceptive_leak_lo"])), min(1, _f(r["deceptive_leak_hi"]))],
                        color=col, linewidth=0.6, alpha=0.55, zorder=2)
                ax.plot([x], [y], linestyle="none", marker=FAMILY_MARKER[fam], markersize=size, color=col,
                        markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
                labels.append((x, y, str(r["defense"])))
            # ~5 pt in data units: x span / approximate axes width in points
            span = 1.17 - max(-0.04, x0)
            width_pt = ((W2 if ncols > 1 else W1) / ncols) * 72 * 0.8
            _frontier_labels(ax, labels, dx=5.0 * span / width_pt)
            ax.set_title(m, loc="left", pad=3)
            if idx % ncols == 0:
                ax.set_ylabel("deceptive-class leak (Wilson 95%)")
        for idx in range(len(models), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        hs = [Line2D([], [], linestyle="none", marker=FAMILY_MARKER[f], markersize=4, markerfacecolor=INK2,
                     markeredgecolor=SURFACE, label=lab) for f, lab in
              (("instruction", "instruction (D0, D1)"), ("information", "information (D2, D3, D3info, D3push)"),
               ("enforcement", "enforcement (D5r, D6r)"))]
        hs += [Line2D([], [], linestyle="none", marker="o", markersize=3.0 + 5.0 * a, markerfacecolor=MUTED,
                      markeredgecolor=SURFACE, label=f"abstention {a:.0%}") for a in (0.0, 0.5, 1.0)]
        fig.legend(handles=hs, loc="outside upper center", ncols=6 if ncols > 1 else 2)
        files = _save(fig, out, name, data)
    return FigureResult(name, "written", files, "bottom-right = completes the real task and never leaks; "
                                                "whiskers = Wilson 95% of the leak")


# ---------------------------------------------------------------- representation (H5)

def _dumbbell(ax, data: pd.DataFrame, arms: list[str], models: list[str], styles: dict, gate: dict,
              arm_marker: dict) -> None:
    _xgrid(ax)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(len(models) - 0.5, -0.5)
    for i, m in enumerate(models):
        col, _ = styles[m]
        sub = data[data["model"] == m]
        xs = [(_f(sub[sub["arm"] == a].iloc[0]["leak_p"]) if len(sub[sub["arm"] == a]) else float("nan")) for a in arms]
        pts = [(x, a) for x, a in zip(xs, arms) if _finite(x)]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [i] * len(pts), color=col, linewidth=0.9, alpha=0.6, zorder=2)
        for x, a in pts:
            mk, hollow = arm_marker[a]
            _point(ax, x, i, col, mk, hollow=hollow, size=5.4 if hollow else 3.8, z=3 if hollow else 4)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)


def fig_representation(fd: FigureData, out: Path) -> FigureResult:
    """H5 cells: IDN pages under T1 (punycode) vs T1u (Unicode), and A1 lookalikes under the
    screenshot arms S0/S1/S2. A panel whose cells are absent is left out (noted), never drawn empty."""
    name = "fig_representation" + fd.suffix
    t = fd.t("arm_rates")
    h = (fd.claims or {}).get("hypotheses") or {}
    a_idn = t[t["contrast"] == "H5a"] if len(t) else t
    a_s = t[t["contrast"] == "H5b/H5c"] if len(t) else t
    panels = [(sub, arms, lab, hids) for sub, arms, lab, hids in
              ((a_idn, ["T1", "T1u"], "IDN pages: T1 (punycode) vs T1u (Unicode)", ["H5a"]),
               (a_s, ["S0", "S1", "S2"], "A1 lookalikes: S0 viewport · S1 omnibox · S2 fake bar", ["H5b", "H5c"]))
              if not sub.empty]
    if not panels:
        return _skip(name, "neither the IDN T1/T1u cells nor the screenshot arms are in this namespace")
    missing = [] if not a_s.empty else ["screenshot arms S0/S1/S2 (e2s_screenshot) not in this namespace"]
    missing += [] if not a_idn.empty else ["IDN T1/T1u cells not in this namespace"]
    models = ordered_models(set().union(*[set(sub["model"]) for sub, *_ in panels]))
    styles = model_styles(models)
    arm_marker = {"T1": ("o", True), "T1u": ("o", False), "S0": ("s", True), "S1": ("s", False), "S2": ("D", False)}
    width = W2 if len(panels) > 1 else W1
    with rc_context(RC):
        fig = Figure(figsize=(width, 0.85 + 0.2 * len(models)), layout="constrained")
        axes = np.atleast_1d(fig.subplots(1, len(panels)))
        arms_shown: list[str] = []
        for ax, (sub, arms, lab, hids) in zip(axes, panels):
            ax.set_title(lab, loc="left", pad=3)
            _dumbbell(ax, sub, arms, models, styles, fd.gate, arm_marker)
            ax.tick_params(axis="y", length=0)
            _compact_x(ax, 0.25)
            arms_shown += [a for a in arms if a in set(sub["arm"])]
            txt = []
            for hid in hids:
                r = h.get(hid) or {}
                if r.get("estimate") is not None and r.get("ci"):
                    txt.append(f"{hid}: RD {float(r['estimate']):+.2f} [{_f(r['ci'][0]):+.2f}, {_f(r['ci'][1]):+.2f}]"
                               f", {r.get('status', '').replace('_', ' ')}")
            ax.set_xlabel("\n".join(["leak (share of episodes)", *txt]))
        arm_label = {"T1": "T1 (punycode URL)", "T1u": "T1u (Unicode URL)", "S0": "S0 viewport",
                     "S1": "S1 + real omnibox", "S2": "S2 + fake address bar"}
        hs = [Line2D([], [], linestyle="none", marker=arm_marker[a][0], markersize=4,
                     markerfacecolor=SURFACE if arm_marker[a][1] else INK2, markeredgecolor=INK2,
                     label=arm_label.get(a, a)) for a in arms_shown]
        nc = min(len(hs), 5 if width == W2 else 2)
        fig.legend(handles=_row_major(hs, nc), loc="outside upper center", ncols=nc)
        data = pd.concat([sub for sub, *_ in panels], ignore_index=True)
        files = _save(fig, out, name, data)
    return FigureResult(name, "written", files, "; ".join(missing))


# ---------------------------------------------------------------- parity (E5)

def fig_parity(fd: FigureData, out: Path) -> FigureResult:
    name = "fig_parity" + fd.suffix
    p = fd.t("parity")
    if p.empty:
        return _skip(name, "E5 parity (in-process) or its E1 chromium counterpart is not in this namespace")
    models = ordered_models(p["model"])
    styles = model_styles(models)
    classes = [c for c in CLASS_ORDER if c in set(p["origin_class"])]
    offs = np.linspace(-0.25, 0.25, len(models)) if len(models) > 1 else np.array([0.0])
    with rc_context(RC):
        fig = Figure(figsize=(W1, 0.7 + 0.2 * len(classes)), layout="constrained")
        ax = fig.subplots()
        _xgrid(ax)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(len(classes) - 0.5, -0.5)
        ax.axvline(0.0, color=MUTED, linewidth=0.7, zorder=1)
        for i, c in enumerate(classes):
            for j, m in enumerate(models):
                r = p[(p["model"] == m) & (p["origin_class"] == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                col, mk = styles[m]
                _hline_ci(ax, r["rd_lo"], r["rd_hi"], i + offs[j], col, 1.0, xmin=-1.0)
                _point(ax, float(r["rd"]), i + offs[j], col, mk, hollow=fd.gate.get(m) not in (None, "pass"))
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([CLASS_LABEL.get(c, c) + (" (R_bound)" if c == "exact" else "") for c in classes])
        ax.set_xlabel("risk difference, Chromium − in-process (Newcombe 95%)")
        ax.tick_params(axis="y", length=0)
        _compact_x(ax, 0.5)
        nc = 2 if len(models) > 3 else len(models)
        fig.legend(handles=_row_major(_model_handles(models, styles, fd.gate), nc), loc="outside upper center",
                   ncols=nc)
        files = _save(fig, out, name, p)
    return FigureResult(name, "written", files, "")


# ---------------------------------------------------------------- exposure heatmap (H1a)

PIPELINE_LABEL = {
    "ours_T0": "ours T0 (no URL)", "ours_T1": "ours T1 (URL, punycode)", "ours_T1u": "ours T1u (URL, Unicode)",
    "ours_T2": "ours T2 (+ origin annotations)",
    "pw_aria": "Playwright aria snapshot", "pw_aria_ai": "Playwright aria snapshot (AI mode)",
    "pw_mcp": "Playwright MCP snapshot",
    "bgym_axtree": "BrowserGym AXTree + tabs", "bgym_axtree_1tab": "BrowserGym AXTree, 1 tab",
    "bgym_axtree_html_1tab": "BrowserGym AXTree + HTML (MiniWoB)", "bgym_dom": "BrowserGym raw DOM",
    "bu_default": "browser-use DOM state", "bu_coi_off": "browser-use, x-origin iframes off",
}


def fig_exposure(fd: FigureData, out: Path) -> FigureResult:
    name = "fig_exposure" + fd.suffix
    e = fd.t("exposure")
    if e.empty:
        return _skip(name, "exposure audit (exposure/exposure.json) not in this namespace")
    from originblind.analysis.audits import PIPELINE_ORDER

    items = [(c, lab) for c, lab in (("sink_domain_present", "sink domain shown"), ("top_url_present", "top URL shown"),
                                     ("form_action_present", "form action shown"))
             if c in e.columns]
    if not items:
        return _skip(name, "exposure rows carry none of sink_domain_present / top_url_present / form_action_present")
    note = "pages pooled over flows; ours_* rows are our own arms"
    if "environment" in e.columns and e["environment"].nunique() > 1:
        envs = sorted(set(e["environment"]))
        keep = "realistic" if "realistic" in envs else envs[0]
        e = e[e["environment"] == keep]
        note += f"; audit environments {envs}, drawn: {keep} (all in the CSV)"
    pipes = sorted(set(e["pipeline"]), key=lambda p: (PIPELINE_ORDER.index(p) if p in PIPELINE_ORDER else 99, p))
    classes = [c for c in CLASS_ORDER if c in set(e["origin_class"])]
    dflt = {r.pipeline: bool(r.default_config) for r in e.itertuples()}
    fam = {r.pipeline: r.framework_family for r in e.itertuples()}

    def colour(v):
        if not _finite(v):
            return SURFACE
        if v <= 0:
            return NEUTRAL
        return SEQ[min(len(SEQ) - 1, 3 + int(round(float(v) * 6)))]  # steps 250..550 (partial -> full)

    with rc_context(RC):
        fig = Figure(figsize=(W2, 0.75 + 0.15 * len(pipes)), layout="constrained")
        axes = fig.subplots(1, len(items), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (col, lab) in zip(axes, items):
            ax.set_xlim(-0.5, len(classes) - 0.5)
            ax.set_ylim(len(pipes) - 0.5, -0.5)
            for s in ax.spines.values():
                s.set_visible(False)
            for i, pp in enumerate(pipes):
                for j, c in enumerate(classes):
                    r = e[(e["pipeline"] == pp) & (e["origin_class"] == c)]
                    v = _f(r.iloc[0][col]) if len(r) else float("nan")
                    ax.add_patch(Rectangle((j - 0.5 + 0.03, i - 0.5 + 0.06), 0.94, 0.88, facecolor=colour(v),
                                           edgecolor="none"))
                    if _finite(v) and 0 < v < 1:
                        pages = int(r.iloc[0]["pages"])
                        ax.text(j, i, f"{int(round(v * pages))}/{pages}", ha="center", va="center", fontsize=5,
                                color=INK)
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([CLASS_LABEL.get(c, c).replace(" (floor)", "") for c in classes], rotation=40,
                               ha="right", rotation_mode="anchor")
            ax.set_yticks(range(len(pipes)))
            ax.set_yticklabels([f"{PIPELINE_LABEL.get(p, p)}{' *' if dflt.get(p) and fam.get(p) != 'ours' else ''}"
                                for p in pipes])
            ax.tick_params(length=0)
            ax.set_title(lab, loc="left", pad=3)
            ours = [i for i, p in enumerate(pipes) if fam.get(p) == "ours"]
            if ours and len(ours) < len(pipes):
                ax.axhline(max(ours) + 0.5, color=AXIS, linewidth=0.6)
        hs = [Patch(facecolor=NEUTRAL, label="on no page"), Patch(facecolor=SEQ[6], label="on some pages (k/n)"),
              Patch(facecolor=SEQ[9], label="on every page")]
        hs.append(Line2D([], [], linestyle="none", marker="$*$", color=INK2, label="framework default configuration"))
        fig.legend(handles=hs, loc="outside upper center", ncols=4)
        files = _save(fig, out, name, e)
    return FigureResult(name, "written", files, note)


# ---------------------------------------------------------------- browser-use mechanism (H4)

_BU_STYLE = {"leak": (STATUS["critical"], SURFACE, "leak"), "bound": (STATUS["good"], SURFACE, "bound"),
             "withheld": (NEUTRAL, INK, "held"), "blocked": (NEUTRAL, INK2, "blocked"),
             "no_field": (SURFACE, MUTED, "no field"), "none": (SURFACE, MUTED, "none")}


def fig_browser_use(fd: FigureData, out: Path) -> FigureResult:
    name = "fig_browser_use" + fd.suffix
    b = fd.t("browser_use")
    b = b[b["mode"] == "deterministic"] if len(b) else b
    if b.empty:
        return _skip(name, "browser-use audit (browser_use/browser_use.json) not in this namespace")
    from originblind.analysis.audits import BU_OUTCOMES

    cfg_cols = ["sensitive_data_format", "allowed_domains", "cross_origin_iframes"]
    cfgs = b[cfg_cols].drop_duplicates()
    fmt_rank = {"domain": 0, "legacy": 1}
    coi_rank = {"default": 0, "true": 1, "false": 2}
    cfgs = cfgs.assign(_f=cfgs["sensitive_data_format"].map(lambda v: fmt_rank.get(v, 9)),
                       _c=cfgs["cross_origin_iframes"].map(lambda v: coi_rank.get(str(v), 9)))
    cfgs = cfgs.sort_values(["_f", "allowed_domains", "_c"]).drop(columns=["_f", "_c"]).reset_index(drop=True)
    classes = [c for c in CLASS_ORDER if c in set(b["origin_class"])]
    rows = []
    for j, cfg in cfgs.iterrows():
        for i, c in enumerate(classes):
            r = b[(b["origin_class"] == c) & (b[cfg_cols] == cfg[cfg_cols].values).all(axis=1)]
            if r.empty:
                continue
            r = r.iloc[0]
            counts = {o: int(r.get(o, 0)) for o in BU_OUTCOMES}
            top = max(BU_OUTCOMES, key=lambda o: (counts[o], -BU_OUTCOMES.index(o)))
            rows.append({**{k: cfg[k] for k in cfg_cols}, "origin_class": c, "n": int(r["n"]), "outcome": top,
                         **counts, "_x": j, "_y": i})
    data = pd.DataFrame(rows)
    with rc_context(RC):
        fig = Figure(figsize=(W2, 1.25 + 0.2 * len(classes)), layout="constrained")
        ax = fig.subplots()
        ax.set_xlim(-0.5, len(cfgs) - 0.5)
        ax.set_ylim(len(classes) - 0.5, -0.5)
        for s in ax.spines.values():
            s.set_visible(False)
        for _, r in data.iterrows():
            fc, tc, lab = _BU_STYLE[r["outcome"]]
            ax.add_patch(Rectangle((r["_x"] - 0.47, r["_y"] - 0.44), 0.94, 0.88, facecolor=fc, edgecolor=GRID,
                                   linewidth=0.4))
            text = lab if r["n"] == 1 else f"{lab} {int(r[r['outcome']])}/{int(r['n'])}"
            ax.text(r["_x"], r["_y"], text, ha="center", va="center", fontsize=5.5, color=tc)
        coi_short = {"default": "default", "true": "on", "false": "off"}
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels([f"{c.allowed_domains}\n{coi_short.get(str(c.cross_origin_iframes), c.cross_origin_iframes)}"
                            for c in cfgs.itertuples()], fontsize=5.5)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([CLASS_LABEL.get(c, c) for c in classes])
        ax.tick_params(length=0)
        # group label per sensitive_data format under its columns, and what the two tick lines mean
        fmt_name = {"domain": "domain-scoped sensitive_data", "legacy": "flat (legacy) sensitive_data"}
        for fmt_v, g in cfgs.groupby("sensitive_data_format", sort=False):
            xs = list(g.index)
            t = ax.text((min(xs) + max(xs)) / 2, -0.30, fmt_name.get(fmt_v, str(fmt_v)), ha="center", va="top",
                        fontsize=6, color=INK2, transform=ax.get_xaxis_transform())
            t.set_in_layout(True)
            ax.plot([min(xs) - 0.4, max(xs) + 0.4], [-0.27, -0.27], color=AXIS, linewidth=0.6, clip_on=False,
                    transform=ax.get_xaxis_transform())
        ax.text(-0.55, -0.02, "allowed_domains\ncross_origin_iframes", ha="right", va="top", fontsize=5.5,
                color=MUTED, transform=ax.get_xaxis_transform())
        hs = [Patch(facecolor=_BU_STYLE[o][0], edgecolor=GRID, label=t) for o, t in
              (("leak", "leak: canary reached a third party"), ("bound", "bound: canary reached the official site"),
               ("withheld", "held: secret not substituted"), ("blocked", "blocked: navigation refused"),
               ("no_field", "no field: not in the library's DOM state"))]
        fig.legend(handles=_row_major(hs, 3), loc="outside upper center", ncols=3)
        files = _save(fig, out, name, data.drop(columns=["_x", "_y"]))
    return FigureResult(name, "written", files, "deterministic tool-executor runs (no LLM)")


# ---------------------------------------------------------------- environment check (lab vs realistic)

_ENV_EXPERIMENTS = ("e1_default", "e2_information", "e3_enforcement", "e6_thinking")
_EXP_SHORT = {"e1_default": "E1", "e2_information": "E2", "e2s_screenshot": "E2-S", "e3_enforcement": "E3",
              "e3_utility": "E3-U", "e5_parity": "E5", "e6_thinking": "E6"}


def _env_rows(t: pd.DataFrame) -> list[tuple]:
    """Row strata of the environment figure: E1 per defense, E2 per arm, E3 per defense, E6 per arm
    (utility flows have no deceptive class and stay in the table)."""
    keys = t[["experiment", "arm", "defense"]].drop_duplicates()
    keys = keys[keys["experiment"].isin(_ENV_EXPERIMENTS)]
    rank = {e: i for i, e in enumerate(_ENV_EXPERIMENTS)}
    keys = keys.assign(_e=keys["experiment"].map(rank),
                       _a=keys["arm"].map(lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99),
                       _d=keys["defense"].map(lambda d: DEFENSE_ORDER.index(d) if d in DEFENSE_ORDER else 99))
    return [tuple(r) for r in keys.sort_values(["_e", "_a", "_d"])[["experiment", "arm", "defense"]].itertuples(
        index=False)]


def _env_row_label(exp: str, arm: str, defense: str, t: pd.DataFrame) -> str:
    sub = t[t["experiment"] == exp]
    many_arms, many_defs = sub["arm"].nunique() > 1, sub["defense"].nunique() > 1
    parts = [_EXP_SHORT.get(exp, exp)]
    if many_arms or not many_defs:
        parts.append(arm)
    if many_defs or not many_arms:
        parts.append(defense)
    return " · ".join(parts)


def fig_envcheck(paired: pd.DataFrame, out: Path, name: str = "fig_envcheck") -> FigureResult:
    """Paired lab vs realistic environment, one panel per model: per experiment x arm/defense row,
    the deceptive-class leak (model colour) and the exact completion (gray), each a dumbbell from
    the lab rate (hollow) to the realistic rate (filled) on the same paired episodes."""
    from originblind.analysis.envcheck import ENV_A, ENV_B, POOLED

    if paired is None or paired.empty:
        return _skip(name, "no lab/realistic pairs (needs --paired-env and both environments)")
    t = paired[paired["primary"] & paired["unit"].isin(["deceptive", "exact"]) & (paired["model"] != POOLED)]
    t = t[t["experiment"].isin(_ENV_EXPERIMENTS)]
    if t.empty:
        return _skip(name, "no paired E1/E2/E3/E6 strata")
    # a reasoning-format variant ("<key>-think", E6 only) is drawn in its base model's panel
    base = {m: m[: -len("-think")] for m in set(t["model"]) if m.endswith("-think") and m[: -len("-think")] in set(t["model"])}
    folded = sorted(base)
    if base:
        t = t.assign(model=t["model"].replace(base))
    rows = _env_rows(t)
    models = ordered_models(t["model"])
    styles = model_styles(models)
    ncols = min(4, len(models))
    nrows = math.ceil(len(models) / ncols)
    unit_style = {"deceptive": (-0.17, "o"), "exact": (0.17, "s")}
    data = t[["experiment", "model", "arm", "defense", "driver", "flows", "unit", "outcome", "n_pairs",
              f"k_{ENV_A}", f"p_{ENV_A}", f"k_{ENV_B}", f"p_{ENV_B}", "diff", "diff_lo", "diff_hi",
              f"only_{ENV_A}", f"only_{ENV_B}", "mcnemar_p"]].copy()
    with rc_context(RC | {"figure.constrained_layout.w_pad": 0.06}):
        fig = Figure(figsize=(W2, 0.75 + nrows * (0.19 * len(rows) + 0.35)), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False, sharey=True)
        fig.supxlabel("share of paired episodes (same plan item in both environments)", fontsize=7, color=INK2)
        for idx, m in enumerate(models):
            ax = axes[idx // ncols][idx % ncols]
            col, _ = styles[m]
            _xgrid(ax)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(len(rows) - 0.5, -0.5)
            prev = None
            for i, (exp, arm, d) in enumerate(rows):
                if prev is not None and exp != prev:
                    ax.axhline(i - 0.5, color=AXIS, linewidth=0.5, zorder=1)
                prev = exp
                for unit, (dy, mk) in unit_style.items():
                    r = t[(t["model"] == m) & (t["experiment"] == exp) & (t["arm"] == arm) & (t["defense"] == d)
                          & (t["unit"] == unit)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    a, b = _f(r[f"p_{ENV_A}"]), _f(r[f"p_{ENV_B}"])
                    cc = col if unit == "deceptive" else INK2
                    if _finite(a) and _finite(b):
                        ax.plot([a, b], [i + dy, i + dy], color=cc, linewidth=0.9, alpha=0.7, zorder=2,
                                solid_capstyle="round")
                    if _finite(b):
                        _point(ax, b, i + dy, cc, mk, hollow=False, size=3.4, z=3)
                    if _finite(a):  # the lab ring on top: a coincident pair reads as a dot inside a ring
                        ax.plot([a], [i + dy], linestyle="none", marker=mk, markersize=5.2, markerfacecolor="none",
                                markeredgecolor=cc, markeredgewidth=0.8, zorder=4)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([_env_row_label(e, a, d, t) + (" (thinking)" if e == "e6_thinking" and folded else "")
                                for e, a, d in rows])
            ax.tick_params(axis="y", length=0)
            _compact_x(ax, 0.5)
            ax.set_title(m + (" (E6: thinking mode)" if any(base[f] == m for f in folded) else ""), loc="left",
                         pad=3)
        for idx in range(len(models), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        hs = [Line2D([], [], linestyle="none", marker="o", markersize=5.2, markerfacecolor="none",
                     markeredgecolor=INK2, markeredgewidth=0.8, label=f"{ENV_A} environment (ring)"),
              Line2D([], [], linestyle="none", marker="o", markersize=3.4, markerfacecolor=INK2,
                     markeredgecolor=SURFACE, label=f"{ENV_B} environment (dot)"),
              Line2D([], [], linestyle="-", linewidth=0.9, marker="o", markersize=3.9, color=MUTED,
                     label="deceptive-class leak (model colour)"),
              Line2D([], [], linestyle="-", linewidth=0.9, marker="s", markersize=3.6, color=INK2,
                     label="exact-origin completion")]
        fig.legend(handles=hs, loc="outside upper center", ncols=4)
        files = _save(fig, out, name, data)
    return FigureResult(name, "written", files, f"paired episodes; difference, McNemar and CIs in the CSV twin "
                                                f"and tables/envcheck_paired.csv")


# ---------------------------------------------------------------- driver

FIGURES = (fig_competence, fig_exposure, fig_rr_forest, fig_rr_forest_e1_flows, fig_knowing_doing,
           fig_defense_frontier, fig_representation, fig_parity, fig_browser_use)
LEGACY_FIGURES = (fig_rr_forest_legacy,)


def make_figures(fd: FigureData, out_dir: str | Path, suffix: str | None = None) -> list[FigureResult]:
    out = Path(out_dir)
    if suffix is not None:
        fd.suffix = suffix
    results = []
    for fn in (LEGACY_FIGURES if fd.legacy else FIGURES):
        try:
            results.append(fn(fd, out))
        except Exception as e:  # noqa: BLE001 - a broken figure must not lose the tables/claims
            results.append(FigureResult(fn.__name__ + fd.suffix, "skipped", [], f"error: {type(e).__name__}: {e}"))
    return results


__all__ = ["FigureData", "FigureResult", "make_figures", "model_styles", "ordered_models", "FIGURES",
           "LEGACY_FIGURES", "W1", "W2"]
