"""scPerturBench linearModel.r-equivalent baseline on the clean protocol."""

import json

import numpy as np
from sklearn.decomposition import PCA

from run_esct_devgamma_clean import (
    DEV_FRAC, TRAIN_FRAC, load_dataset, metrics, split_terms,
)


RIDGE = 0.1
PCA_DIM = 10
N_SPLITS = 12
OUT = r"D:\guo\新课题_待定\experiments\linearModel_clean_protocol.json"


def main():
    X, pert, emb = load_dataset("norman")
    dim = int(emb[next(iter(emb))].shape[0])
    labels = sorted(set(pert) - {"control"})
    singles = [x for x in labels if len(split_terms(x)) == 1]
    combos = [x for x in labels if len(split_terms(x)) == 2
              and all(g in emb for g in split_terms(x))]
    all_genes = sorted({g for x in singles + combos for g in split_terms(x)
                        if g in emb})
    pert_idx = {x: np.where(pert == x)[0] for x in labels}
    ctrl_idx = np.where(pert == "control")[0]
    ctrl_mean = np.asarray(X[ctrl_idx].mean(0)).ravel()

    def sparse_mean(idx):
        return np.asarray(X[idx].mean(0)).ravel()

    delta = {x: sparse_mean(pert_idx[x]) - ctrl_mean for x in singles + combos}
    all_rows = []
    for seed in range(1, N_SPLITS + 1):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(all_genes).tolist()
        n_tr = int(len(all_genes) * TRAIN_FRAC)
        n_dev = int(len(all_genes) * DEV_FRAC)
        train_set = set(perm[:n_tr])
        dev_set = set(perm[n_tr:n_tr + n_dev])
        test_set = set(perm[n_tr + n_dev:])

        train_singles = [x for x in singles
                         if split_terms(x)[0] in train_set]
        train_combos = [c for c in combos
                        if all(g in train_set for g in split_terms(c))]
        train_conds = train_singles + train_combos
        if not train_conds:
            continue
        Y = np.stack([delta[c] for c in train_conds], axis=1).astype(np.float64)
        center = Y.mean(axis=1)
        Yc = Y - center[:, None]
        pca = PCA(n_components=min(PCA_DIM, len(train_conds) - 1),
                  random_state=seed).fit(Y.T)
        E = pca.components_.T.astype(np.float64)
        d = E.shape[1]
        P = np.stack([np.sum([emb[g] for g in split_terms(c)], axis=0)
                      for c in train_conds], axis=1).astype(np.float64)
        Ep = E.T @ E + RIDGE * np.eye(d)
        Pp = P @ P.T + RIDGE * np.eye(P.shape[0])
        K = np.linalg.solve(Ep, E.T @ Yc) @ P.T
        K = np.linalg.solve(Pp, K.T).T

        def predict(c):
            p = np.sum([emb[g] for g in split_terms(c)], axis=0)
            return E @ K @ p + center

        cat0 = [c for c in combos
                if any(g in test_set for g in split_terms(c))
                and all(g not in train_set for g in split_terms(c))]
        for c in cat0:
            pred = predict(c)
            all_rows.append({"seed": seed, "combo": c,
                             "metrics": metrics(pred, delta[c])})
        print(f"seed {seed}: n={len(cat0)}", flush=True)

    arr = np.array([r["metrics"] for r in all_rows])
    result = {"n": len(all_rows), "mean": {
        "pcc": float(arr[:, 0].mean()), "dir": float(arr[:, 1].mean()),
        "l2": float(arr[:, 2].mean())}, "rows": all_rows}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["mean"], indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
