"""Shared matplotlib style for the paper's figures: PGF backend through pdflatex with Linux Biolinum (the ACM
sans face), so figure text matches the paper and every font is embedded as Type 1."""
from __future__ import annotations

import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402

from common import MODELS  # noqa: E402

COLUMN_W, TEXT_W = 3.33, 7.0  # inches, ACM sigconf

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([r"\usepackage[T1]{fontenc}", r"\usepackage{libertine}",
                               r"\renewcommand{\familydefault}{\sfdefault}", r"\usepackage{amsmath}"]),
    "font.family": "sans-serif",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
})

# Lightness encodes model size within each family; direct labels also identify models.
FAMILY_STEPS = {
    "qwen": ["#86b6ef", "#3987e5", "#256abf", "#0d366b"],   # blue ramp steps 250, 400, 500, 700
    "gemma": ["#1baf7a"],
    "ministral": ["#f39a74", "#c44a1b"],                      # orange, light and dark steps
}
COLOR = {
    "qwen3.5-2b": FAMILY_STEPS["qwen"][0], "qwen3.5-4b": FAMILY_STEPS["qwen"][1],
    "qwen3.5-9b": FAMILY_STEPS["qwen"][2], "qwen3.5-35b-a3b": FAMILY_STEPS["qwen"][3],
    "gemma-4-12b": FAMILY_STEPS["gemma"][0],
    "ministral-3-8b": FAMILY_STEPS["ministral"][0], "ministral-3-14b": FAMILY_STEPS["ministral"][1],
}
MARKER = {m: ("o" if m.startswith("qwen") else "s" if m.startswith("gemma") else "^") for m in MODELS}
LABEL = {"qwen3.5-2b": "Q-2B", "qwen3.5-4b": "Q-4B", "qwen3.5-9b": "Q-9B", "qwen3.5-35b-a3b": "Q-35B",
         "gemma-4-12b": "G-12B", "ministral-3-8b": "M-8B", "ministral-3-14b": "M-14B"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
