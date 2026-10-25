"""Figure 3: the defense ladder. Release on lookalike pages (n = 96 per cell), release on injected-receiver pages
(n = 48), and completion of the legitimate login (n = 48), for each defense (rows) and model (columns), all with the
URL shown (framework-default observation). Cell values are percentages; 95% Wilson intervals are in Appendix B.
A dagger marks cells that are zero by construction (the gate withholds the secret). Data: report tables/defense.csv."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import figstyle as S
from common import DEFENSES_E3, FIGURES, MODELS, table

ROWS = ["Instruction", "Annotations", "Check tool", "Pushed facts", "Pushed verdict", "Top-level gate",
        "Receiver gate"]
assert len(ROWS) == len(DEFENSES_E3)
by = {(r["model"], r["defense"]): r for r in table("defense")}


def matrix(prefix):
    a = np.zeros((len(DEFENSES_E3), len(MODELS)))
    for i, d in enumerate(DEFENSES_E3):
        for j, m in enumerate(MODELS):
            r = by[(m, d)]
            a[i, j] = int(r[f"{prefix}_k"]) / int(r[f"{prefix}_n"])
    return a


ORANGE = LinearSegmentedColormap.from_list("orange", ["#fbf1ea", "#f7c3a6", "#ef8a5c", "#d55a26", "#9a3a12",
                                                      "#5e2209"])
BLUE = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
PANELS = [("a1_leak", "(a) release to lookalike domains", ORANGE, {("D5r",), ("D6r",)}),
          ("a2_leak", "(b) release to injected receivers", ORANGE, {("D6r",)}),
          ("exact_r_bound", "(c) completion of the legitimate login", BLUE, set())]

fig, axes = plt.subplots(1, 3, figsize=(S.TEXT_W, 1.95), gridspec_kw={"wspace": 0.06})
for ax, (prefix, title, cmap, construct) in zip(axes, PANELS):
    a = matrix(prefix)
    ax.imshow(a, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i, d in enumerate(DEFENSES_E3):
        for j, m in enumerate(MODELS):
            v = a[i, j]
            rgb = cmap(v)[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            txt = f"{100 * v:.0f}" + (r"$^\dagger$" if (d,) in construct else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white" if lum < 0.5 else S.INK)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([S.LABEL[m] for m in MODELS], fontsize=6)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=2)
    ax.set_yticks(range(len(DEFENSES_E3)))
    ax.set_yticklabels(ROWS if ax is axes[0] else [], fontsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # thin surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DEFENSES_E3), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel(title, labelpad=3)
out = FIGURES / "fig_defense_heatmap.pdf"
FIGURES.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("wrote", out)
