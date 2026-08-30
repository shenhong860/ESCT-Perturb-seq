"""ESCT gain vs observed non-additivity ratio on Norman cat0."""

import json

import numpy as np
import scipy.stats as st
from scipy.stats import pearsonr

from paths import env_path, experiments_dir

W1 = env_path("ESCT_W1_JSON",
              experiments_dir() / "esct_w1_devgamma_clean_norman.json")
EPS = env_path("ESCT_EPS_JSON",
               experiments_dir() / "epistasis_discovery_norman.json")
OUT = env_path("ESCT_EPS_STRAT_OUT",
               experiments_dir() / "epsilon_stratification.json")


def main():
    rows = json.load(open(W1))["rows"]
    eps_map = {r["combo"]: r["ratio"] for r in json.load(open(EPS))["per_combo"]}

    acc = {}
    for r in rows:
        c = r["combo"]
        if c not in eps_map:
            continue
        d = acc.setdefault(c, {"add": [], "esct": [], "l2add": [], "l2esct": []})
        d["add"].append(r["add"][0])
        d["esct"].append(r["w1"][0])
        d["l2add"].append(r["add"][2])
        d["l2esct"].append(r["w1"][2])

    combos = []
    for c, d in acc.items():
        combos.append({
            "combo": c,
            "epsilon": eps_map[c],
            "gain_pcc": float(np.mean(d["esct"]) - np.mean(d["add"])),
            "gain_l2": float(np.mean(d["l2add"]) - np.mean(d["l2esct"])),
        })
    combos.sort(key=lambda x: x["epsilon"])

    eps = np.array([x["epsilon"] for x in combos])
    gain_pcc = np.array([x["gain_pcc"] for x in combos])
    gain_l2 = np.array([x["gain_l2"] for x in combos])
    med = float(np.median(eps))
    low = [x for x in combos if x["epsilon"] <= med]
    high = [x for x in combos if x["epsilon"] > med]

    def group_stats(group, key):
        vals = np.array([x[key] for x in group])
        return float(vals.mean()), float(vals.std())

    result = {
        "n_combos": len(combos),
        "median_epsilon": med,
        "corr_epsilon_gain_pcc": float(pearsonr(eps, gain_pcc)[0]),
        "corr_epsilon_gain_l2": float(pearsonr(eps, gain_l2)[0]),
        "low_epsilon": {
            "n": len(low),
            "gain_pcc": group_stats(low, "gain_pcc"),
            "gain_l2": group_stats(low, "gain_l2"),
        },
        "high_epsilon": {
            "n": len(high),
            "gain_pcc": group_stats(high, "gain_pcc"),
            "gain_l2": group_stats(high, "gain_l2"),
        },
        "mannwhitney_gain_pcc": float(st.mannwhitneyu(
            [x["gain_pcc"] for x in low], [x["gain_pcc"] for x in high]).pvalue),
        "mannwhitney_gain_l2": float(st.mannwhitneyu(
            [x["gain_l2"] for x in low], [x["gain_l2"] for x in high]).pvalue),
        "per_combo": combos,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_combo"},
                     indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
