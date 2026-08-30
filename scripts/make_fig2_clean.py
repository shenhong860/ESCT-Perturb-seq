"""Clean-protocol performance figure for the v22 ESCT manuscript."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import env_path, experiments_dir

OUT = env_path("ESCT_FIG2_OUT",
               experiments_dir() / "figure2_performance_v22.pdf")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

methods = ["scGPT full", "scGPT MLP", "linearModel", "additive",
           "CAPRA (80ep)", "ESCT"]
pcc = [0.264, 0.446, 0.504, 0.531, 0.601, 0.667]
dir_acc = [0.620, 0.733, 0.742, 0.754, 0.775, 0.805]
l2 = [4.228, 3.168, 2.976, 2.990, 3.390, 2.487]
colors = ["#B0BEC5", "#90A4AE", "#78909C", "#607D8B", "#0288D1", "#D32F2F"]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, values, title, xlabel, xlim in [
    (axes[0], pcc, "PCC-top100", "PCC (higher is better)", (0, 0.8)),
    (axes[1], dir_acc, "Direction accuracy", "dir (higher is better)",
     (0.5, 0.9)),
    (axes[2], l2, "Mean L2", "L2 (lower is better)", (0, 5)),
]:
    y = np.arange(len(methods))
    ax.barh(y, values, color=colors[::-1], edgecolor="white", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(methods[::-1])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    for yi, v in zip(y, values):
        ax.text(v + (0.01 if xlim[0] >= 0.5 else 0.04), yi,
                f"{v:.3f}", va="center", fontsize=7)

ax = axes[3]
for m, p, l, c in zip(methods, pcc, l2, colors):
    ax.scatter(l, p, s=70, color=c, edgecolors="white", zorder=5)
    ax.annotate(m, (l, p), textcoords="offset points", xytext=(6, 4),
                fontsize=6)
ax.set_xlabel("Mean L2 (lower is better)")
ax.set_ylabel("PCC-top100 (higher is better)")
ax.set_title("PCC vs L2", loc="left", fontweight="bold")
ax.set_xlim(2.0, 4.6)
ax.set_ylim(0.2, 0.75)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("written:", OUT)
