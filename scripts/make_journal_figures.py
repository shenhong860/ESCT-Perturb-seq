"""Journal-style ESCT figures (Fig 2 performance, Fig 3 non-additivity).

Figure contract
---------------
Fig 2 claim: ESCT+W1 improves additive baselines on mean-level and
distributional metrics across Norman, Replogle_exp6, and Wessels under
strict gene splits.
Evidence chain:
  a  paired per-combo PCC, additive vs ESCT+W1, three datasets (primary)
  b  per-dataset mean +/- 95% CI for PCC, dir, L2, MMD (generality)
  c  official scPerturBench pearson_distance paired (orthogonal metric)
  d  Norman same-split baseline dot plot (control against alternatives)
Archetype: quantitative grid; hero panel a.

Fig 3 claim: observed non-additivity is pervasive and ESCT gains are not
concentrated in high-epsilon combinations.
Evidence chain:
  a  distribution of observed non-additivity (epsilon)
  b  epsilon vs additive PCC with regression and marginals
  c  top-15 non-additive pairs as a dot plot

Export contract: SVG, PDF, PNG and TIFF at 600 dpi; Arial; editable text.
Backend: Python/matplotlib only.
"""

import json

from paths import env_path, experiments_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


EXPERIMENTS = experiments_dir()
CAPRA_JSON = env_path("ESCT_CAPRA_JSON",
                       experiments_dir() / "capra_clean_protocol.json")
OUT = env_path("ESCT_FIGURE_DIR", experiments_dir() / "figures_journal")
OUT.mkdir(parents=True, exist_ok=True)

DATASET_LABEL = {
    "norman": "Norman",
    "replogle": "Replogle_exp6",
    "wessels": "Wessels",
}
DATASET_COLOR = {
    "norman": "#0F4D92",
    "replogle": "#42949E",
    "wessels": "#B64342",
}
BLUE = "#0F4D92"
RED = "#B64342"
GREY = "#767676"
LIGHT_GREY = "#CFCECE"


def mean_ci(values):
    """Return (mean, half-width of 95% CI) for an array of values."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float(values.mean()), 0.0
    se = values.std(ddof=1) / np.sqrt(n)
    tcrit = st.t.ppf(0.975, n - 1)
    return float(values.mean()), float(tcrit * se)


def save_fig(fig, name):
    """Write one figure/panel as SVG, PDF, PNG and TIFF."""
    for ext in (".svg", ".pdf", ".png", ".tiff"):
        path = OUT / (name + ext)
        if ext in (".png", ".tiff"):
            fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=600)
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        print("written:", path)


def panel_label(ax, label, x=-0.10, y=1.04, fontsize=8):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="bottom", ha="left")


def load_w1_rows(name):
    path = EXPERIMENTS / f"esct_w1_devgamma_clean_{name}.json"
    return json.load(open(path, encoding="utf-8"))["rows"]


def load_official_pd(name):
    path = EXPERIMENTS / f"esct_w1_devgamma_clean_{name}_official_pd.json"
    return json.load(open(path, encoding="utf-8"))["per_combo"]


def as_float_list(value):
    if isinstance(value, str):
        return [float(x) for x in json.loads(value)]
    return [float(x) for x in value]


def baseline_rows():
    """Return per-method list of [pcc, dir, l2] rows for Norman."""
    rows = {}

    esct = [as_float_list(r["w1"]) for r in load_w1_rows("norman")]
    rows["ESCT+W1"] = esct

    capra_path = CAPRA_JSON if CAPRA_JSON.exists() else \
        EXPERIMENTS / "capra_clean_protocol.json"
    if capra_path.exists():
        capra = json.load(open(capra_path, encoding="utf-8"))
        rows["CAPRA"] = [as_float_list(r["metrics"]) for r in capra["rows"]]

    lm = json.load(open(EXPERIMENTS / "linearModel_clean_protocol.json",
                        encoding="utf-8"))
    rows["linearModel"] = [as_float_list(r["metrics"]) for r in lm["rows"]]

    mlp = json.load(open(EXPERIMENTS / "scgpt_mlp_clean_protocol.json",
                         encoding="utf-8"))
    rows["scGPT-MLP"] = [as_float_list(r["metrics"]) for r in mlp["rows"]]

    ft = json.load(open(EXPERIMENTS / "scgpt_transformer_finetune_clean.json",
                        encoding="utf-8"))
    ft_rows = []
    for split in ft["per_seed"]:
        for r in split.get("rows", []):
            ft_rows.append([float(r["pcc"]), float(r["dir"]), float(r["l2"])])
    rows["scGPT-full"] = ft_rows
    return rows


def marginal_hist(ax, values, color, position, limits):
    """Add a marginal histogram on top (position='top') or right."""
    inset = ax.inset_axes(position, transform=ax.transAxes)
    if position[1] > 0.5:
        inset.hist(values, bins=18, color=color, alpha=0.55, density=True,
                   edgecolor="white", linewidth=0.4)
        inset.set_ylim(*limits)
        inset.set_xticks([])
        inset.set_yticks([])
    else:
        inset.hist(values, bins=18, color=color, alpha=0.55, density=True,
                   edgecolor="white", linewidth=0.4, orientation="horizontal")
        inset.set_xlim(*limits)
        inset.set_xticks([])
        inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)


def fig2_panel_a(ax):
    ax.set_title("Paired per-combo PCC", loc="left", fontweight="bold",
                 fontsize=7)
    for name in ("norman", "replogle", "wessels"):
        rows = load_w1_rows(name)
        x = [as_float_list(r["add"])[0] for r in rows]
        y = [as_float_list(r["w1"])[0] for r in rows]
        color = DATASET_COLOR[name]
        ax.scatter(x, y, s=14, color=color, alpha=0.72, linewidths=0,
                   label=DATASET_LABEL[name], zorder=3)
    lo, hi = ax.get_xlim()
    lim = (min(lo, hi), max(lo, hi, 1.0))
    ax.plot(lim, lim, color=GREY, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Additive PCC (top-100)")
    ax.set_ylabel("ESCT+W1 PCC (top-100)")
    ax.legend(loc="upper left", fontsize=6, markerscale=1.2)


def make_fig2_panel_a_standalone():
    """Panel a with properly separated marginal distributions and legend."""
    fig = plt.figure(figsize=(4.8, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.06, wspace=0.06, left=0.11, right=0.97,
                          top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    for name in ("norman", "replogle", "wessels"):
        rows = load_w1_rows(name)
        x = [as_float_list(r["add"])[0] for r in rows]
        y = [as_float_list(r["w1"])[0] for r in rows]
        color = DATASET_COLOR[name]
        ax.scatter(x, y, s=14, color=color, alpha=0.72, linewidths=0,
                   label=DATASET_LABEL[name], zorder=3)
        ax_top.hist(x, bins=18, color=color, alpha=0.55, density=True,
                    edgecolor="white", linewidth=0.4)
        ax_right.hist(y, bins=18, color=color, alpha=0.55, density=True,
                      orientation="horizontal", edgecolor="white",
                      linewidth=0.4)

    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Additive PCC (top-100)")
    ax.set_ylabel("ESCT+W1 PCC (top-100)")
    ax.set_title("Paired per-combo PCC", loc="left", fontweight="bold",
                 fontsize=7)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    fig.text(0.03, 0.95, "a", fontweight="bold", fontsize=8)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975),
               fontsize=6.5, frameon=False)
    save_fig(fig, "figure2_journal_panel_a")
    plt.close(fig)


def fig2_panel_b(ax_grid):
    ax_grid.axis("off")
    metrics = [
        ("PCC", 0, 0, False),
        ("Direction", 1, 1, False),
        ("Mean L2", 2, 2, True),
        ("MMD", None, 3, True),
    ]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for pos, (label, row_metric, col_metric, lower_better) in zip(
            positions, metrics):
        sub = ax_grid.figure.add_axes([
            ax_grid.get_position().x0 + pos[1] * 0.245 + 0.015,
            ax_grid.get_position().y0 + (1 - pos[0]) * 0.43 - 0.02,
            0.225, 0.34,
        ])
        sub.set_title(label, loc="left", fontweight="bold", fontsize=6)
        x_cats = [0.0, 0.42]
        for name in ("norman", "replogle", "wessels"):
            rows = load_w1_rows(name)
            if row_metric is None:
                add_vals = [float(r["add_mmd"]) for r in rows]
                esct_vals = [float(r["esct_mmd"]) for r in rows]
            else:
                add_vals = [as_float_list(r["add"])[row_metric] for r in rows]
                esct_vals = [as_float_list(r["w1"])[row_metric] for r in rows]
            color = DATASET_COLOR[name]
            for xi, vals in ((x_cats[0], add_vals), (x_cats[1], esct_vals)):
                m, ci = mean_ci(vals)
                sub.errorbar(xi, m, yerr=ci, fmt="o", ms=3.4, lw=0.8,
                             capsize=2, color=color, zorder=3)
        sub.set_xticks(x_cats)
        sub.set_xticklabels(["Additive", "ESCT+W1"], fontsize=5.5)
        sub.tick_params(labelsize=5.5)
        sub.set_xlim(-0.18, 0.60)
        if lower_better:
            sub.yaxis.set_inverted(True)


def fig2_panel_c(ax):
    ax.set_title("Official pearson_distance", loc="left", fontweight="bold",
                 fontsize=7)
    for name in ("norman", "replogle", "wessels"):
        rows = load_official_pd(name)
        x = [float(r["add"]) for r in rows]
        y = [float(r["esct"]) for r in rows]
        color = DATASET_COLOR[name]
        ax.scatter(x, y, s=14, color=color, alpha=0.72, linewidths=0,
                   label=DATASET_LABEL[name], zorder=3)
    lo, hi = 0.0, 1.15
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Additive pearson_distance")
    ax.set_ylabel("ESCT+W1 pearson_distance")
    ax.legend(loc="upper left", fontsize=6, markerscale=1.2)


def make_fig2_panel_c_standalone():
    """Panel c with separated marginal distributions and legend above."""
    fig = plt.figure(figsize=(4.8, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.06, wspace=0.06, left=0.11, right=0.97,
                          top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    for name in ("norman", "replogle", "wessels"):
        rows = load_official_pd(name)
        x = [float(r["add"]) for r in rows]
        y = [float(r["esct"]) for r in rows]
        color = DATASET_COLOR[name]
        ax.scatter(x, y, s=14, color=color, alpha=0.72, linewidths=0,
                   label=DATASET_LABEL[name], zorder=3)
        ax_top.hist(x, bins=18, color=color, alpha=0.55, density=True,
                    edgecolor="white", linewidth=0.4)
        ax_right.hist(y, bins=18, color=color, alpha=0.55, density=True,
                      orientation="horizontal", edgecolor="white",
                      linewidth=0.4)

    lo, hi = 0.0, 1.15
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Additive pearson_distance")
    ax.set_ylabel("ESCT+W1 pearson_distance")
    ax.set_title("Official pearson_distance", loc="left", fontweight="bold",
                 fontsize=7)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    fig.text(0.03, 0.95, "c", fontweight="bold", fontsize=8)
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975),
               fontsize=6.5, frameon=False)
    save_fig(fig, "figure2_journal_panel_c")
    plt.close(fig)


def fig2_panel_d(ax_grid):
    ax_grid.axis("off")
    methods = baseline_rows()
    method_order = ["ESCT+W1", "CAPRA", "linearModel", "scGPT-MLP",
                    "scGPT-full"]
    metric_names = ["PCC", "Direction", "Mean L2"]
    colors = {"ESCT+W1": RED}
    for idx, metric in enumerate(range(3)):
        sub = ax_grid.figure.add_axes([
            ax_grid.get_position().x0 + idx * 0.245 + 0.015,
            ax_grid.get_position().y0 + 0.28,
            0.225, 0.50,
        ])
        sub.set_title(metric_names[idx], loc="left", fontweight="bold",
                      fontsize=6)
        y_pos = np.arange(len(method_order))[::-1]
        for yi, method in zip(y_pos, method_order):
            if method not in methods:
                continue
            vals = np.array([row[metric] for row in methods[method]])
            m, ci = mean_ci(vals)
            color = colors.get(method, GREY)
            sub.errorbar(m, yi, xerr=ci, fmt="o", ms=4.0, lw=0.8, capsize=2,
                         color=color, zorder=3)
        sub.set_yticks(y_pos)
        sub.set_yticklabels(method_order, fontsize=5.5)
        sub.tick_params(labelsize=5.5)
        sub.set_xlim(0, 1) if metric < 2 else sub.set_xlim(1.5, 5.0)


def make_fig2():
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.35],
                          height_ratios=[1.0, 0.86], hspace=0.42, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    panel_label(ax_c, "c")
    panel_label(ax_d, "d")
    fig2_panel_a(ax_a)
    fig2_panel_b(ax_b)
    fig2_panel_c(ax_c)
    fig2_panel_d(ax_d)
    save_fig(fig, "figure2_journal")
    plt.close(fig)

    make_fig2_panel_a_standalone()

    make_fig2_panel_c_standalone()

    for name, draw, label, size in [
        ("figure2_journal_panel_b", fig2_panel_b, "b", (5.4, 4.0)),
        ("figure2_journal_panel_d", fig2_panel_d, "d", (5.4, 3.6)),
    ]:
        fig = plt.figure(figsize=size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        panel_label(ax, label, x=0.005, y=0.99)
        draw(ax)
        save_fig(fig, name)
        plt.close(fig)


def fig3_data():
    eps = json.load(open(EXPERIMENTS / "epistasis_discovery_norman.json",
                         encoding="utf-8"))
    combos, ratios, pcc_add = [], [], []
    for r in eps["per_combo"]:
        combo = r["combo"]
        if "1+" in combo or "+1" in combo:
            continue
        combos.append(combo)
        ratios.append(float(r["ratio"]))
        pcc_add.append(float(r["pcc_add"]))
    return combos, np.array(ratios), np.array(pcc_add)


def fig3_panel_a(ax):
    _, ratios, _ = fig3_data()
    ax.hist(ratios, bins=18, density=True, color=BLUE, alpha=0.55,
            edgecolor="white", linewidth=0.4)
    x = np.linspace(0, ratios.max() * 1.02, 300)
    kde = st.gaussian_kde(ratios)
    ax.plot(x, kde(x), color=BLUE, lw=1.2)
    ax.axvline(np.median(ratios), color=RED, lw=0.8, ls="--")
    ax.text(np.median(ratios) + 0.03, ax.get_ylim()[1] * 0.9,
            f"median {np.median(ratios):.2f}", fontsize=6, color=RED)
    ax.set_xlabel("Observed non-additivity  ε")
    ax.set_ylabel("Density")


def fig3_panel_b(ax):
    _, ratios, pcc_add = fig3_data()
    ax.scatter(ratios, pcc_add, s=16, color=BLUE, alpha=0.7, linewidths=0,
               zorder=3)
    res = st.linregress(ratios, pcc_add)
    xs = np.linspace(ratios.min(), ratios.max(), 100)
    ax.plot(xs, res.intercept + res.slope * xs, color=RED, lw=1.2)
    ax.text(0.97, 0.04, f"r = {res.rvalue:.2f}", transform=ax.transAxes,
            ha="right", fontsize=6)
    ax.set_xlabel("Observed non-additivity  ε")
    ax.set_ylabel("Additive baseline PCC")


def make_fig3_panel_b_standalone():
    """Panel b with separated marginal distributions and regression."""
    fig = plt.figure(figsize=(4.8, 4.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.06, wspace=0.06, left=0.11, right=0.97,
                          top=0.90, bottom=0.10)
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    _, ratios, pcc_add = fig3_data()
    ax.scatter(ratios, pcc_add, s=16, color=BLUE, alpha=0.7, linewidths=0,
               zorder=3)
    res = st.linregress(ratios, pcc_add)
    xs = np.linspace(ratios.min(), ratios.max(), 100)
    ax.plot(xs, res.intercept + res.slope * xs, color=RED, lw=1.2)
    ax.text(0.97, 0.04, f"r = {res.rvalue:.2f}", transform=ax.transAxes,
            ha="right", fontsize=6)
    ax_top.hist(ratios, bins=18, color="#42949E", alpha=0.55, density=True,
                edgecolor="white", linewidth=0.4)
    ax_right.hist(pcc_add, bins=18, color="#0F4D92", alpha=0.55,
                  density=True, orientation="horizontal",
                  edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Observed non-additivity  ε")
    ax.set_ylabel("Additive baseline PCC")
    ax.set_title("Additive PCC vs ε", loc="left", fontweight="bold",
                 fontsize=7)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.set_yticks([])
    ax_right.set_xticks([])
    fig.text(0.03, 0.95, "b", fontweight="bold", fontsize=8)
    save_fig(fig, "figure3_journal_panel_b")
    plt.close(fig)


def fig3_panel_c(ax):
    combos, ratios, _ = fig3_data()
    order = np.argsort(-ratios)[:15]
    names = [combos[i] for i in order][::-1]
    vals = ratios[order][::-1]
    colors = [RED if v > 0.8 else BLUE for v in vals]
    ax.scatter(vals, np.arange(len(vals)), s=16, color=colors, zorder=3)
    ax.axvline(0.8, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=5.5)
    ax.set_xlabel("Observed non-additivity  ε")
    for xi, yi in zip(vals, np.arange(len(vals))):
        ax.text(xi + 0.02, yi, f"{xi:.2f}", va="center", fontsize=5.5)


def make_fig3():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, label in zip(axes, "abc"):
        panel_label(ax, label)
    fig3_panel_a(axes[0])
    fig3_panel_b(axes[1])
    fig3_panel_c(axes[2])
    fig.tight_layout()
    save_fig(fig, "figure3_journal")
    plt.close(fig)

    make_fig3_panel_b_standalone()
    panels = [
        ("figure3_journal_panel_a", fig3_panel_a),
        ("figure3_journal_panel_c", fig3_panel_c),
    ]
    for name, draw in panels:
        fig, ax = plt.subplots(figsize=(3.0, 2.5))
        draw(ax)
        fig.tight_layout()
        save_fig(fig, name)
        plt.close(fig)


if __name__ == "__main__":
    make_fig2()
    make_fig3()
    print("journal figures written to", OUT)
