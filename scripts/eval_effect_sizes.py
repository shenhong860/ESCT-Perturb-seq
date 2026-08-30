"""Cohen's d and per-split seed summaries for clean ESCT results."""

import json

import numpy as np


DATASETS = {
    "norman": r"D:\guo\新课题_待定\experiments\esct_devgamma_clean_norman.json",
    "replogle": r"D:\guo\新课题_待定\experiments\esct_devgamma_clean_replogle.json",
    "wessels": r"D:\guo\新课题_待定\experiments\esct_devgamma_clean_wessels.json",
}
OUT = r"E:\guo\新课题_待定\experiments\effect_sizes.json"


def seeded_rows(obj):
    rows = obj["rows"]
    offset = 0
    out = []
    for seed, vec in obj["vectors_cat0"].items():
        count = len(vec)
        for r in rows[offset:offset + count]:
            out.append((int(seed), r["combo"], r["add"], r["esct"]))
        offset += count
    if offset != len(rows):
        raise RuntimeError("row/vector mismatch")
    return out


def main():
    result = {}
    for name, path in DATASETS.items():
        obj = json.load(open(path))
        rows = seeded_rows(obj)
        add = np.array([r[2] for r in rows])
        esct = np.array([r[3] for r in rows])
        diffs = esct - add
        cohens = {
            "pcc": float(diffs[:, 0].mean() / (diffs[:, 0].std(ddof=1) + 1e-12)),
            "dir": float(diffs[:, 1].mean() / (diffs[:, 1].std(ddof=1) + 1e-12)),
            "l2": float(diffs[:, 2].mean() / (diffs[:, 2].std(ddof=1) + 1e-12)),
        }
        seed_summary = {}
        for seed in sorted({r[0] for r in rows}):
            vals = np.array([r[3] for r in rows if r[0] == seed])
            seed_summary[str(seed)] = [float(vals[:, j].mean())
                                       for j in range(3)]
        arr = np.array(list(seed_summary.values()))
        result[name] = {
            "n": len(rows),
            "cohens_d": cohens,
            "seed_means_esct_mean": [float(arr[:, j].mean())
                                     for j in range(3)],
            "seed_means_esct_std": [float(arr[:, j].std(ddof=1))
                                    for j in range(3)],
            "seed_means": seed_summary,
        }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk != "seed_means"} for k, v in result.items()},
                     indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
