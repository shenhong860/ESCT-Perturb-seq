"""Summarize Norman ablation table with paired tests against full ESCT+W1."""

import json

import numpy as np
import scipy.stats as st

from paths import env_path, experiments_dir

FULL = env_path("ESCT_W1_JSON",
                experiments_dir() / "esct_w1_devgamma_clean_norman.json")
NO_RET = env_path("ESCT_ABLATION_NO_RET_JSON",
                  experiments_dir() / "ablation_w1_norman_noretrieval.json")
NO_RES = env_path("ESCT_ABLATION_NO_RES_JSON",
                  experiments_dir() / "ablation_w1_norman_noresidual.json")
MEAN = env_path("ESCT_NORMAN_JSON",
                experiments_dir() / "esct_devgamma_clean_norman.json")
MEAN_MMD = env_path("ESCT_ABLATION_MEAN_MMD_JSON",
                    experiments_dir() / "ablation_mean_only_mmd.json")
OUT = env_path("ESCT_ABLATION_SUMMARY_OUT",
               experiments_dir() / "ablation_summary.json")


def w1_map(path, metric_field="w1"):
    obj = json.load(open(path))
    out = {}
    rows = obj["rows"]
    offset = 0
    for seed, vec in obj["vectors_cat0"].items():
        count = len(vec)
        for r in rows[offset:offset + count]:
            out[(str(seed), r["combo"])] = {"metrics": r[metric_field],
                                            "mmd": r["esct_mmd"],
                                            "add_mmd": r["add_mmd"]}
        offset += count
    if offset != len(rows):
        raise RuntimeError(f"{path}: row/vector mismatch")
    return out


def mean_map(path):
    obj = json.load(open(path))
    out = {}
    rows = obj["rows"]
    offset = 0
    for seed, vec in obj["vectors_cat0"].items():
        count = len(vec)
        for r in rows[offset:offset + count]:
            out[(str(seed), r["combo"])] = r
        offset += count
    if offset != len(rows):
        raise RuntimeError(f"{path}: row/vector mismatch")
    return out


def main():
    full = w1_map(FULL)
    no_ret = w1_map(NO_RET)
    no_res = w1_map(NO_RES)
    mean_rows = mean_map(MEAN)
    mean_mmd = {(str(r["seed"]), r["combo"]): r["mmd"]
                for r in json.load(open(MEAN_MMD))["per_combo"]}
    keys = [k for k in full if k in no_ret and k in no_res and k in mean_rows
            and k in mean_mmd]

    variants = {
        "full_esct_w1": {"metrics": [full[k]["metrics"] for k in keys],
                         "mmd": [full[k]["mmd"] for k in keys]},
        "minus_retrieval": {"metrics": [no_ret[k]["metrics"] for k in keys],
                            "mmd": [no_ret[k]["mmd"] for k in keys]},
        "minus_residual": {"metrics": [no_res[k]["metrics"] for k in keys],
                           "mmd": [no_res[k]["mmd"] for k in keys]},
        "minus_transport": {"metrics": [mean_rows[k]["esct"] for k in keys],
                            "mmd": [mean_mmd[k] for k in keys]},
        "additive": {"metrics": [mean_rows[k]["add"] for k in keys],
                     "mmd": [full[k]["add_mmd"] for k in keys]},
    }

    result = {"n": len(keys), "variants": {}}
    for name, table in variants.items():
        arr = np.array(table["metrics"])
        result["variants"][name] = {
            "pcc": float(arr[:, 0].mean()),
            "dir": float(arr[:, 1].mean()),
            "l2": float(arr[:, 2].mean()),
            "mmd": float(np.mean(table["mmd"])),
        }

    result["paired_vs_full"] = {}
    for name, table in variants.items():
        if name == "full_esct_w1":
            continue
        row = {}
        for j, metric in enumerate(["pcc", "dir", "l2"]):
            a = np.array([full[k]["metrics"][j] for k in keys])
            b = np.array([table["metrics"][i][j] for i in range(len(keys))])
            diff = b - a
            row[metric] = {
                "full": float(a.mean()), "ablated": float(b.mean()),
                "p": float(st.wilcoxon(a, b).pvalue) if np.any(diff) else 1.0,
                "full_win": int(np.sum((a > b) if metric != "l2"
                                       else (a < b))),
            }
        a = np.array([full[k]["mmd"] for k in keys])
        b = np.array(table["mmd"])
        diff = b - a
        row["mmd"] = {
            "full": float(a.mean()), "ablated": float(b.mean()),
            "p": float(st.wilcoxon(a, b).pvalue) if np.any(diff) else 1.0,
            "full_win": int(np.sum(a < b)),
        }
        result["paired_vs_full"][name] = row

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
