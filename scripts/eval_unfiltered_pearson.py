"""Unfiltered all-gene Pearson delta comparison for clean ESCT vectors."""

import json

import numpy as np
import scipy.stats as st
from scipy.stats import pearsonr

from paths import env_path, experiments_dir

DATASETS = {
    "norman": env_path("ESCT_NORMAN_JSON",
                       experiments_dir() / "esct_devgamma_clean_norman.json"),
    "replogle": env_path("ESCT_REPLOGLE_JSON",
                         experiments_dir() / "esct_devgamma_clean_replogle.json"),
    "wessels": env_path("ESCT_WESSELS_JSON",
                        experiments_dir() / "esct_devgamma_clean_wessels.json"),
}
OUT = env_path("ESCT_UNFILTERED_PEARSON_OUT",
               experiments_dir() / "unfiltered_pearson.json")


def one_dataset(name, path):
    obj = json.load(open(path))
    add, esct = [], []
    rows = []
    for seed, vecs in obj["vectors_cat0"].items():
        for combo, v in vecs.items():
            true = np.asarray(v["true"], dtype=float)
            a = np.asarray(v["add"], dtype=float)
            e = np.asarray(v["esct"], dtype=float)
            if min(np.std(true), np.std(a), np.std(e)) < 1e-12:
                continue
            ra = float(pearsonr(a, true)[0])
            re = float(pearsonr(e, true)[0])
            add.append(ra)
            esct.append(re)
            rows.append({"seed": int(seed), "combo": combo,
                         "add": ra, "esct": re})
    add = np.array(add)
    esct = np.array(esct)
    diff = esct - add
    return {
        "dataset": name,
        "n": len(diff),
        "mean_add": float(add.mean()),
        "mean_esct": float(esct.mean()),
        "mean_diff": float(diff.mean()),
        "p": float(st.wilcoxon(esct, add).pvalue) if np.any(diff) else 1.0,
        "esct_win": int(np.sum(diff > 0)),
        "per_combo": rows,
    }


def main():
    result = {name: one_dataset(name, path)
              for name, path in DATASETS.items()}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk != "per_combo"} for k, v in result.items()},
                     indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
