import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, json, os

OUT = r"D:\guo\新课题_待定\submission"
os.makedirs(OUT, exist_ok=True)

# CNS-style settings
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Nature-style palette
C_BLUE = "#4477AA"
C_CYAN = "#66CCEE"
C_GREEN = "#228833"
C_YELLOW = "#CCBB44"
C_RED = "#EE6677"
C_PURPLE = "#AA3377"
C_GREY = "#BBBBBB"
C_DARK = "#333333"
BG = "#F8F8F8"

# ==============================
# FIGURE 1: Method overview
# ==============================
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.set_xlim(-0.5, 10.5); ax1.set_ylim(-0.5, 7.5); ax1.axis("off")
ax1.set_facecolor("white")

def draw_box(ax, x, y, w, h, title, subtitle="", fc="#E8F0FE", ec=C_BLUE, fs=8):
    b = FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x+w/2, y+h*0.65, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=C_DARK)
    if subtitle:
        ax.text(x+w/2, y+h*0.30, subtitle, ha="center", va="center", fontsize=fs-1.5, color="#666666")

def draw_arrow(ax, x0, y0, x1, y1, color=C_DARK, style="-|>"):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                arrowprops=dict(arrowstyle=style, lw=1.5, color=color, connectionstyle="arc3,rad=0.0"))

# Panel label
ax1.text(-0.3, 7.2, "a", fontsize=14, fontweight="bold", color=C_DARK)

# Section headers
ax1.text(2.0, 6.8, "Additive backbone", fontsize=9, fontweight="bold", color=C_BLUE, ha="center")
ax1.text(5.8, 6.8, "Interaction residual", fontsize=9, fontweight="bold", color=C_RED, ha="center")
ax1.text(9.0, 6.8, "W1 transport", fontsize=9, fontweight="bold", color=C_GREEN, ha="center")

# --- Input (left) ---
draw_box(ax1, -0.3, 3.0, 1.6, 1.0, "Control cells\n$x_0$", "mean = $\\mu_0$", fc="#E8F8E8", ec=C_GREEN)
draw_box(ax1, -0.3, 1.5, 1.6, 1.0, "Gene embeddings\n(scGPT $e_G$)", "512-d vector", fc="#FFF3E0", ec="#E65100")
draw_arrow(ax1, 1.3, 3.5, 1.8, 4.5)
draw_arrow(ax1, 1.3, 2.0, 1.8, 4.0)

# --- Backbone selection (3 branches) ---
draw_box(ax1, 1.8, 5.0, 1.8, 0.9, "Direct\n$\\Delta_G$", "gene seen", fc="#E3F2FD", ec=C_BLUE)
draw_box(ax1, 1.8, 3.7, 1.8, 0.9, "Retrieval\nKNN($e_G$)", "dev median $\\gamma_s$", fc="#E3F2FD", ec=C_CYAN)
draw_box(ax1, 1.8, 2.4, 1.8, 0.9, "Global mean\n$\\bar{\\Delta}$", "gate fail", fc="#FCE4EC", ec=C_RED)

# Connect branches to sum
for yy in [5.45, 4.15, 2.85]:
    draw_arrow(ax1, 3.65, yy, 4.3, 4.0)

# Sum junction
ax1.plot(4.45, 4.0, "o", ms=14, color=C_BLUE, zorder=5)
ax1.text(4.45, 4.0, "+", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
ax1.text(4.45, 3.4, "$\\Delta_{\\mathrm{anchor}}$", ha="center", fontsize=8, color=C_BLUE)

# --- Interaction residual ---
draw_box(ax1, 4.8, 3.5, 1.9, 1.0, "Interaction\nresidual $R(A,B)$",
         "$U[\\sigma(Wh_A){\\odot}\\sigma(Wh_B)]$", fc="#FFEBEE", ec=C_RED)
draw_arrow(ax1, 4.6, 4.0, 4.75, 4.0)

# Sum residual
ax1.plot(7.0, 4.0, "o", ms=14, color=C_RED, zorder=5)
ax1.text(7.0, 4.0, "+", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

# Final prediction
draw_box(ax1, 7.4, 3.5, 1.7, 1.0, "ESCT prediction\n$\\hat{\\Delta}(A,B)$", "",
         fc="#E8F5E9", ec=C_GREEN)
draw_arrow(ax1, 7.15, 4.0, 7.35, 4.0)

# --- W1 transport (bottom row) ---
draw_box(ax1, 7.4, 0.8, 1.7, 1.0, "W1 transport\n$\\mathcal{T}_\\theta(x_0)$",
         "cell-level synthesis", fc="#E0F2F1", ec=C_GREEN)
draw_arrow(ax1, 8.25, 3.4, 8.25, 1.85)
draw_box(ax1, 9.5, 0.8, 1.3, 1.0, "Predicted\ncells", "",
         fc="#E8F5E9", ec=C_GREEN)
draw_arrow(ax1, 9.15, 1.3, 9.45, 1.3)

# Labels for metrics
ax1.text(7.0, 2.8, "PCC / dir / L2", ha="center", fontsize=7, color="#666", style="italic")
ax1.text(9.0, 2.8, "MMD / pd", ha="center", fontsize=7, color="#666", style="italic")

plt.tight_layout()
fig1.savefig(f"{OUT}/figure1_method_v2.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig1.savefig(f"{OUT}/figure1_method_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# ==============================
# FIGURE 2: Performance comparison (4 panels)
# ==============================
fig2, axes2 = plt.subplots(1, 4, figsize=(14, 3.2))

methods = ["add", "linearModel", "ESCT v1", "ESCT v2.1", "scGPT ft"]
pcc_vals = [0.508, 0.518, 0.678, 0.701, 0.731]
dir_vals = [0.758, 0.751, 0.814, 0.828, 0.874]
l2_vals  = [3.142, 3.080, 2.571, 2.489, 4.010]
colors   = ["#BDBDBD", "#9E9E9E", "#64B5F6", "#1976D2", "#EF5350"]

# Panel a: PCC
ax = axes2[0]
bars = ax.barh(methods[::-1], pcc_vals[::-1], color=colors[::-1], edgecolor="white", height=0.6)
ax.set_xlabel("PCC-top100", fontsize=9)
ax.set_title("a  PCC (higher = better)", fontsize=9, fontweight="bold", loc="left")
for bar, v in zip(bars, pcc_vals[::-1]):
    ax.text(v+0.01, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlim(0, 0.85)

# Panel b: dir
ax = axes2[1]
bars = ax.barh(methods[::-1], dir_vals[::-1], color=colors[::-1], edgecolor="white", height=0.6)
ax.set_xlabel("Direction accuracy", fontsize=9)
ax.set_title("b  Direction (higher = better)", fontsize=9, fontweight="bold", loc="left")
for bar, v in zip(bars, dir_vals[::-1]):
    ax.text(v+0.005, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlim(0.7, 0.95)

# Panel c: L2
ax = axes2[2]
bars = ax.barh(methods[::-1], l2_vals[::-1], color=colors[::-1], edgecolor="white", height=0.6)
ax.set_xlabel("Mean L2 distance", fontsize=9)
ax.set_title("c  L2 (lower = better)", fontsize=9, fontweight="bold", loc="left")
for bar, v in zip(bars, l2_vals[::-1]):
    ax.text(v+0.08, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=7)
ax.set_xlim(0, 5)

# Panel d: PCC vs L2 scatter (trade-off)
ax = axes2[3]
for i, (m, p, l) in enumerate(zip(methods, pcc_vals, l2_vals)):
    ax.scatter(l, p, color=colors[i], s=80, zorder=5, edgecolors="white", linewidths=1)
    ax.annotate(m, (l, p), textcoords="offset points", xytext=(5,5), fontsize=6)
ax.set_xlabel("Mean L2 (lower = better)", fontsize=9)
ax.set_ylabel("PCC-top100 (higher = better)", fontsize=9)
ax.set_title("d  Direction-calibration trade-off", fontsize=9, fontweight="bold", loc="left")
ax.annotate("", xy=(2.5, 0.72), xytext=(3.8, 0.75),
            arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.5))
ax.text(3.2, 0.76, "ESCT\nimprovement", fontsize=6, color=C_GREEN, ha="center")

fig2.tight_layout()
fig2.savefig(f"{OUT}/figure2_performance_v2.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig2.savefig(f"{OUT}/figure2_performance_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# ==============================
# FIGURE 3: Epistasis discovery (3 panels)
# ==============================
j = json.load(open(r"D:\guo\新课题_待定\experiments\epistasis_discovery_norman.json"))
combos = [r["combo"] for r in j["per_combo"]]
ratios = [r["ratio"] for r in j["per_combo"]]
pccs = [r["pcc_add"] for r in j["per_combo"]]

# filter guide-identity artefacts
valid = [(c,r,p) for c,r,p in zip(combos,ratios,pccs) if "1+" not in c and "+1" not in c]
combos_f = [x[0] for x in valid]; ratios_f = [x[1] for x in valid]; pccs_f = [x[2] for x in valid]

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 3.8))

# Panel a: histogram
ax = axes3[0]
ax.hist(ratios_f, bins=18, color=C_BLUE, edgecolor="white", alpha=0.85)
ax.axvline(0.5, color=C_RED, ls="--", lw=1.5)
ax.text(0.52, ax.get_ylim()[1]*0.9, "additive/epistasis", fontsize=7, color=C_RED)
ax.set_xlabel("Observed residual ratio $\\epsilon$", fontsize=9)
ax.set_ylabel("Combinations", fontsize=9)
ax.set_title("a  Observed non-additivity", fontsize=9, fontweight="bold", loc="left")

# Panel b: scatter
ax = axes3[1]
ax.scatter(ratios_f, pccs_f, c=C_BLUE, s=25, alpha=0.7, edgecolors="white", linewidths=0.5)
ax.set_xlabel("Observed residual ratio $\\epsilon$", fontsize=9)
ax.set_ylabel("Additive baseline PCC", fontsize=9)
ax.set_title("b  Additive PCC vs $\\epsilon$", fontsize=9, fontweight="bold", loc="left")
# annotate top 3
top3 = sorted(zip(combos_f, ratios_f, pccs_f), key=lambda x:-x[1])[:3]
for c, r, p in top3:
    ax.annotate(c, (r, p), fontsize=6, textcoords="offset points", xytext=(4, 4))

# Panel c: top 15 epistatic pairs as horizontal bar
ax = axes3[2]
top15 = sorted(zip(combos_f, ratios_f), key=lambda x:-x[1])[:15]
names = [x[0] for x in top15][::-1]
vals = [x[1] for x in top15][::-1]
bar_colors = [C_RED if v > 0.8 else C_BLUE for v in vals]
ax.barh(names, vals, color=bar_colors, edgecolor="white", height=0.6)
ax.set_xlabel("Observed residual ratio $\\epsilon$", fontsize=9)
ax.set_title("c  Top 15 non-additive pairs", fontsize=9, fontweight="bold", loc="left")
for i, v in enumerate(vals):
    ax.text(v+0.02, i, f"{v:.2f}", va="center", fontsize=6)

fig3.tight_layout()
fig3.savefig(f"{OUT}/figure3_epistasis_v2.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig3.savefig(f"{OUT}/figure3_epistasis_v2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

print("ALL 3 CNS-STYLE FIGURES WRITTEN TO", OUT)
