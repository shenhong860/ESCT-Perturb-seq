"""Official scPerturBench pearson_distance for leakage-clean ESCT reruns."""

import json
import pickle
import sys

import anndata as ad
import numpy as np
import scipy.stats as st
from scipy.stats import pearsonr

from paths import dataset_paths, deg_dir


CONFIG = {
    "norman": {
        "h5ad": dataset_paths("norman")[0],
        "deg": deg_dir() / "DEG_hvg5000_Norman.pkl",
    },
    "replogle": {
        "h5ad": dataset_paths("replogle")[0],
        "deg": deg_dir() / "DEG_hvg5000_Replogle_exp6.pkl",
    },
    "wessels": {
        "h5ad": dataset_paths("wessels")[0],
        "deg": deg_dir() / "DEG_hvg5000_Wessels.pkl",
    },
}
TOP_DEG = 100


def main():
    name, result_path, out_path = sys.argv[1:]
    cfg = CONFIG[name.lower()]
    a = ad.read_h5ad(cfg["h5ad"], backed="r")
    genes = list(a.var_names)
    a.file.close()
    result = json.load(open(result_path))
    with open(cfg["deg"], "rb") as f:
        deg = pickle.load(f)

    rows = []
    for seed, combos in result["vectors_cat0"].items():
        for combo, vec in combos.items():
            if combo not in deg:
                continue
            keep = np.array([i for i, g in enumerate(genes)
                             if g in set(deg[combo].index[:TOP_DEG])], dtype=int)
            if len(keep) < 2:
                continue
            true = np.asarray(vec["true"])[keep]
            add = np.asarray(vec["add"])[keep]
            esct = np.asarray(vec["esct"])[keep]
            if min(np.std(true), np.std(add), np.std(esct)) < 1e-12:
                continue
            pd_add = 1.0 - float(pearsonr(add, true)[0])
            pd_esct = 1.0 - float(pearsonr(esct, true)[0])
            rows.append({"seed": int(seed), "combo": combo,
                         "n_deg": int(len(keep)), "add": pd_add,
                         "esct": pd_esct})

    add = np.array([r["add"] for r in rows])
    esct = np.array([r["esct"] for r in rows])
    diff = esct - add
    summary = {
        "dataset": name,
        "n": len(rows),
        "mean_add": float(add.mean()),
        "mean_esct": float(esct.mean()),
        "mean_diff": float(diff.mean()),
        "p": float(st.wilcoxon(esct, add).pvalue) if len(rows) > 1 else None,
        "esct_win": int(np.sum(diff < 0)),
        "per_combo": rows,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_combo"},
                     indent=2))
    print("written:", out_path)


if __name__ == "__main__":
    main()
