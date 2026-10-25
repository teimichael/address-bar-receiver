"""Figure 2: the binding plane. Each model is placed by its release ratio on lookalike domains (x) and on the
injected cross-origin iframe (y); arrows go from the condition without the receiver signal to the condition with it.
Release ratio = deceptive release rate / legitimate completion rate in the same cell. Crosshairs at the arrow heads
are 95% intervals (union of Katz and brand-cluster bootstrap intervals). Data: generated/derived/binding_plane.csv."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import figstyle as S
from common import FIGURES, MODELS, read_derived

rows = read_derived("binding_plane")


def pt(exp, cond, model, group):
    for r in rows:
        if r["experiment"] == exp and r["model"] == model and r["group"] == group and \
                (r["arm"] if exp == "e2_information" else r["defense"]) == cond:
            return float(r["rr"]), float(r["lo"]), float(r["hi"])
    raise LookupError((exp, cond, model, group))


PANELS = [
    ("e2_information", "T1", "T2", "(a) Showing the receiver: URL only $\\rightarrow$ URL + receiver",
     {"qwen3.5-2b": (6, 4), "qwen3.5-4b": (5, -9), "qwen3.5-9b": (4, 4), "qwen3.5-35b-a3b": (6, -9),
      "gemma-4-12b": (-21, 5), "ministral-3-8b": (-21, -8), "ministral-3-14b": (6, 6)}),
    ("e3_enforcement", "D1", "D3push", "(b) Pushing a verdict: instruction only $\\rightarrow$ pushed verdict",
     {"qwen3.5-9b": (5, 6)}),
]

fig, axes = plt.subplots(1, 2, figsize=(S.TEXT_W, 2.3), sharey=True, gridspec_kw={"wspace": 0.08})
LIM = (-0.05, 1.33)
YLIM = (-0.05, 1.55)
for ax, (exp, c0, c1, title, offsets) in zip(axes, PANELS):
    # guide regions (visual aids only, not classifications)
    for (x0, y0, w, h, lab, xy, ha) in ((LIM[0], 0.7, 0.45, YLIM[1] - 0.7, "refuses\nlookalikes only", (-0.02, 1.5), "left"),
                                        (LIM[0], YLIM[0], 0.45, 0.45, "refuses\nboth", (0.38, 0.38), "right"),
                                        (0.7, 0.7, LIM[1] - 0.7, YLIM[1] - 0.7, "refuses\nneither", (1.3, 1.5), "right")):
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#f0efec", edgecolor="none", zorder=0))
        if exp == "e2_information":  # region captions once; panel (b) shares the same regions
            ax.text(*xy, lab, ha=ha, va="top", fontsize=6, color=S.MUTED, zorder=1, linespacing=0.95)
    ax.axhline(1, color=S.GRID, lw=0.5, zorder=0.5)
    ax.axvline(1, color=S.GRID, lw=0.5, zorder=0.5)
    for m in MODELS:
        x0, _, _ = pt(exp, c0, m, "a1")
        y0, _, _ = pt(exp, c0, m, "iframe")
        x1, xl, xh = pt(exp, c1, m, "a1")
        y1, yl, yh = pt(exp, c1, m, "iframe")
        col = S.COLOR[m]
        ax.plot([xl, xh], [y1, y1], color=col, lw=0.7, alpha=0.9, zorder=2, solid_capstyle="butt")
        ax.plot([x1, x1], [yl, yh], color=col, lw=0.7, alpha=0.9, zorder=2, solid_capstyle="butt")
        if abs(x1 - x0) + abs(y1 - y0) > 0.03:
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                         color=col, alpha=0.75, shrinkA=2.5, shrinkB=3.5, zorder=3))
        ax.scatter([x0], [y0], s=14, marker=S.MARKER[m], facecolor="white", edgecolor=col, linewidth=0.8, zorder=4)
        ax.scatter([x1], [y1], s=18, marker=S.MARKER[m], facecolor=col, edgecolor="white", linewidth=0.5, zorder=5)
        off = offsets.get(m)
        if off is not None:
            ax.annotate(S.LABEL[m], (x1, y1), xytext=off, textcoords="offset points", fontsize=6, color=S.INK,
                        va="center", zorder=6)
    if exp == "e3_enforcement":
        lead = dict(arrowstyle="-", lw=0.4, color=S.MUTED, shrinkA=0, shrinkB=2.5)
        for m, txt, xy in (("ministral-3-8b", "M-8B", (0.17, 0.30)), ("qwen3.5-2b", "Q-2B", (0.17, 0.21)),
                           ("qwen3.5-4b", "Q-4B", (0.17, 0.12)),
                           ("gemma-4-12b", "G-12B, M-14B, Q-35B", (0.17, 0.03))):
            x1, _, _ = pt(exp, c1, m, "a1")
            y1, _, _ = pt(exp, c1, m, "iframe")
            ax.annotate(txt, (x1, y1), xytext=xy, textcoords="data", fontsize=6, color=S.INK, va="center",
                        zorder=6, arrowprops=lead)
    ax.set_xlim(*LIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel("release ratio on lookalike domains")
axes[0].set_ylabel("release ratio on the injected iframe")
# legend: open = before, filled = after; marker shape = family
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=S.INK2, markersize=3.6,
            label="without the signal"),
     Line2D([], [], marker="o", ls="", markerfacecolor=S.INK2, markeredgecolor="white", markersize=4.2,
            label="with the signal (95\\% CI)"),
     Line2D([], [], marker="o", ls="", color=S.COLOR["qwen3.5-9b"], markersize=3.6, label="Qwen3.5"),
     Line2D([], [], marker="s", ls="", color=S.COLOR["gemma-4-12b"], markersize=3.6, label="Gemma-4"),
     Line2D([], [], marker="^", ls="", color=S.COLOR["ministral-3-14b"], markersize=3.8, label="Ministral-3")]
axes[1].legend(handles=h, loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2,
               labelspacing=0.25, bbox_to_anchor=(1.0, 0.0))
out = FIGURES / "fig_binding_plane.pdf"
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
