"""Paired clean-protocol baseline comparison for the Norman cat0 set."""

import json

import numpy as np
import scipy.stats as st

from paths import REPO_ROOT, env_path, experiments_dir

ESCT = env_path("ESCT_JSON", experiments_dir() / "esct_devgamma_clean_norman.json")
W1 = env_path("ESCT_W1_JSON",
              experiments_dir() / "esct_w1_devgamma_clean_norman.json")
LM = env_path("ESCT_LINEAR_JSON",
              experiments_dir() / "linearModel_clean_protocol.json")
MLP = env_path("ESCT_SCGPT_MLP_JSON",
               experiments_dir() / "scgpt_mlp_clean_protocol.json")
SCGPT = env_path("ESCT_SCGPT_FT_JSON",
                 experiments_dir() / "scgpt_transformer_finetune_clean.json")
CAPRA = env_path("ESCT_CAPRA_JSON",
                 experiments_dir() / "capra_clean_protocol.json")
OUT = env_path("ESCT_SUMMARY_OUT",
               REPO_ROOT / "outputs" / "clean_baseline_comparison.json")


def rows_with_seed(path):
    obj = json.load(open(path))
    out = []
    if "rows" in obj and obj["rows"] and "seed" in obj["rows"][0]:
        return [(str(r["seed"]), r["combo"], r["metrics"]
                 if "metrics" in r else None) for r in obj["rows"]]
    rows = obj["rows"]
    offset = 0
    for seed, vec in obj["vectors_cat0"].items():
        for r in rows[offset:offset + len(vec)]:
            out.append((str(seed), r["combo"], r["esct"]))
        offset += len(vec)
    return out


def scgpt_rows():
    obj = json.load(open(SCGPT))
    out = []
    for split in obj["per_seed"]:
        for r in split["rows"]:
            out.append((str(split["seed"]), r["combo"],
                        [r["pcc"], r["dir"], r["l2"]]))
    return out


def main():
    esct = {(s, c): m for s, c, m in rows_with_seed(ESCT)}
    lm = {(s, c): m for s, c, m in rows_with_seed(LM)}
    mlp = {(s, c): m for s, c, m in rows_with_seed(MLP)}
    capra = {(s, c): m for s, c, m in rows_with_seed(CAPRA)}
    sc = {(s, c): m for s, c, m in scgpt_rows()}
    keys = [k for k in esct
            if k in lm and k in mlp and k in capra and k in sc]
    result = {"n": len(keys), "means": {}}
    names = {"esct": esct, "linearModel": lm, "CAPRA": capra,
             "scGPT-MLP": mlp, "scGPT-full": sc}
    for name, table in names.items():
        arr = np.array([table[k] for k in keys])
        result["means"][name] = [float(arr[:, j].mean()) for j in range(3)]
    result["paired_vs_esct"] = {}
    for name, table in names.items():
        if name == "esct":
            continue
        for j, metric in enumerate(["pcc", "dir", "mean_l2"]):
            a = np.array([esct[k][j] for k in keys])
            b = np.array([table[k][j] for k in keys])
            diff = b - a
            result["paired_vs_esct"].setdefault(name, {})[metric] = {
                "esct": float(a.mean()), "other": float(b.mean()),
                "p": float(st.wilcoxon(a, b).pvalue) if np.any(diff) else 1.0,
                "esct_win": int(np.sum((a > b) if metric != "mean_l2"
                                       else (a < b))),
            }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
