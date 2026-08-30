"""MLP-head scGPT baseline on the clean unified Norman protocol.

The response head is trained on scGPT gene-pair embeddings to predict the
interaction residual on top of the clean additive anchor.  No test labels or
development labels enter the fit.
"""

import json

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn

from run_esct_devgamma_clean import (
    DEV_FRAC, TRAIN_FRAC, load_dataset, metrics, split_terms,
)


EPOCHS = 50
LR = 1e-4
HIDDEN = 256
N_SPLITS = 12
OUT = r"D:\guo\新课题_待定\experiments\scgpt_mlp_clean_protocol.json"


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

    true_delta = {c: sparse_mean(pert_idx[c]) - ctrl_mean for c in combos}
    all_rows = []
    for seed in range(1, N_SPLITS + 1):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(all_genes).tolist()
        n_tr = int(len(all_genes) * TRAIN_FRAC)
        n_dev = int(len(all_genes) * DEV_FRAC)
        train_set = set(perm[:n_tr])
        dev_set = set(perm[n_tr:n_tr + n_dev])
        test_set = set(perm[n_tr + n_dev:])

        single_map = {}
        for x in singles:
            gs = split_terms(x)
            if len(gs) == 1 and gs[0] in train_set:
                single_map[gs[0]] = sparse_mean(pert_idx[x]) - ctrl_mean
        global_delta = (np.mean(list(single_map.values()), axis=0)
                        if single_map else np.zeros(X.shape[1]))

        def anchor(c):
            return np.sum([single_map.get(g, global_delta)
                           for g in split_terms(c)], axis=0)

        train_combos = [c for c in combos
                        if all(g in train_set for g in split_terms(c))
                        and all(g in single_map for g in split_terms(c))]
        cat0 = [c for c in combos
                if any(g in test_set for g in split_terms(c))
                and all(g not in train_set for g in split_terms(c))]
        if not train_combos or not cat0:
            continue

        X_in = np.stack([np.concatenate([emb[g] for g in split_terms(c)])
                         for c in train_combos]).astype(np.float32)
        Y = np.stack([true_delta[c] - anchor(c) for c in train_combos]
                     ).astype(np.float32)
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(2 * dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, X.shape[1]),
        )
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        Xt = torch.as_tensor(X_in)
        Yt = torch.as_tensor(Y)
        for epoch in range(EPOCHS):
            opt.zero_grad()
            loss = loss_fn(model(Xt), Yt)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            for c in cat0:
                pair = np.concatenate([emb[g] for g in split_terms(c)]
                                      ).astype(np.float32)
                pred = (anchor(c) +
                        model(torch.as_tensor(pair[None])).numpy()[0])
                all_rows.append({"seed": seed, "combo": c,
                                 "metrics": metrics(pred, true_delta[c])})
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
