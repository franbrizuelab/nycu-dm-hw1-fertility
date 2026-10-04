"""Matplotlib style for ACL figures (single column = 3.15 in, full width = 6.5 in)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (blue, orange, aqua) and a single-hue blue ramp for ordered data.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

COL_W, FULL_W = 3.15, 6.5

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "lines.linewidth": 1.5, "legend.frameon": False, "figure.dpi": 160,
})
