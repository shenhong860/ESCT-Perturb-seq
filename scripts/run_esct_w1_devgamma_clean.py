"""Leakage-clean ESCT+W1 rerun on the unified three-dataset protocol."""

import json
import os
import sys

import numpy as np
import scipy.stats as st
import torch
from scipy.optimize import linear_sum_assignment

from model_v2 import DCW1TV2
from run_esct_devgamma_clean import (
    KNN_K, KNN_TAU, LAMBDAS, N_SPLITS, DEV_FRAC, TRAIN_FRAC,
    dir_acc, load_dataset, metrics, norm, pcc, split_terms,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 100
MODEL_SEEDS = [0, 1]
ITERS = 500
BATCH = 128
DIR_WEIGHT = 2.0
MMD_WEIGHT = 10.0
N_CELLS = 200
MAX_TRAIN_CELLS = 300
NO_RETRIEVAL = os.environ.get("ABLATION_NO_RETRIEVAL", "0") == "1"
NO_RESIDUAL = os.environ.get("ABLATION_NO_RESIDUAL", "0") == "1"


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def dense_sample(X, idx, n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(idx, min(n, len(idx)), replace=False)
    block = X[idx]
    return np.asarray(block.todense() if hasattr(block, "todense")
                      else block, dtype=np.float32)


def sparse_mean(X, idx):
    return np.asarray(X[idx].mean(0)).ravel()


def rbf_mmd(x, y, sigma=2.0):
    x = torch.as_tensor(x, device=DEVICE, dtype=torch.float32)
    y = torch.as_tensor(y, device=DEVICE, dtype=torch.float32)
    xx = torch.exp(-torch.cdist(x, x) / (2 * sigma * sigma))
    yy = torch.exp(-torch.cdist(y, y) / (2 * sigma * sigma))
    xy = torch.exp(-torch.cdist(x, y) / (2 * sigma * sigma))
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def vector_norm(v):
    return np.sqrt(np.sum(v * v))


def ridge_solve(Phi, Y, reg):
    """Solve ridge normal equations with PyTorch to avoid mixed BLAS crashes."""
    Pt = torch.as_tensor(Phi, device=DEVICE, dtype=torch.float64)
    Yt = torch.as_tensor(Y, device=DEVICE, dtype=torch.float64)
    A = Pt.T @ Pt + reg * torch.eye(Pt.shape[1], device=DEVICE,
                                     dtype=torch.float64)
    B = Pt.T @ Yt
    return torch.linalg.solve(A, B).cpu().numpy()


def torch_matvec(A, v):
    product = (torch.as_tensor(A, device=DEVICE, dtype=torch.float64) @
               torch.as_tensor(v, device=DEVICE, dtype=torch.float64))
    return product.cpu().numpy()


def run_split(X, pert, emb, dim, seed):
    labels = sorted(set(pert) - {"control"})
    singles = [x for x in labels if len(split_terms(x)) == 1]
    combos = [x for x in labels if len(split_terms(x)) == 2
              and all(g in emb for g in split_terms(x))]
    all_genes = sorted({g for x in singles + combos for g in split_terms(x)
                        if g in emb})
    rng = np.random.RandomState(seed)
    perm = rng.permutation(all_genes).tolist()
    n_tr = int(len(all_genes) * TRAIN_FRAC)
    n_dev = int(len(all_genes) * DEV_FRAC)
    train_set = set(perm[:n_tr])
    dev_set = set(perm[n_tr:n_tr + n_dev])
    test_set = set(perm[n_tr + n_dev:])

    pert_idx = {x: np.where(pert == x)[0] for x in labels}
    ctrl_idx = np.where(pert == "control")[0]
    ctrl_mean = sparse_mean(X, ctrl_idx)
    ctrl_cells = dense_sample(X, ctrl_idx, N_CELLS, 42)

    single_map = {}
    for x in singles:
        gs = split_terms(x)
        if len(gs) == 1 and gs[0] in train_set and gs[0] in emb:
            single_map[gs[0]] = sparse_mean(X, pert_idx[x]) - ctrl_mean
    global_delta = (np.mean(list(single_map.values()), axis=0)
                    if single_map else np.zeros(X.shape[1]))

    train_genes = sorted(single_map)
    bank_emb = (np.stack([norm(emb[g]) for g in train_genes])
                if train_genes else np.zeros((0, dim)))
    bank_delta = (np.stack([single_map[g] for g in train_genes])
                  if train_genes else np.zeros((0, X.shape[1])))
    dev_sims = [float(np.max(torch_matvec(bank_emb, norm(emb[g]))))
                for g in sorted(dev_set)
                if g not in train_set and g in emb and len(train_genes)]
    gamma = float(np.median(dev_sims)) if dev_sims and not NO_RETRIEVAL else 1.0

    def retrieve(g):
        if g in single_map:
            return single_map[g], 1.0
        if not train_genes:
            return global_delta, -1.0
        sims = torch_matvec(bank_emb, norm(emb[g]))
        top1 = float(sims.max())
        nbr = np.argsort(-sims)[:KNN_K]
        weights = np.exp(KNN_TAU * sims[nbr])
        weights /= weights.sum() + 1e-12
        anchor = (weights[:, None] * bank_delta[nbr]).sum(0)
        return (anchor if top1 >= gamma else global_delta), top1

    def anchor(c):
        return np.sum([retrieve(g)[0] for g in split_terms(c)], axis=0)

    true_delta = {c: sparse_mean(X, pert_idx[c]) - ctrl_mean for c in combos}
    train_combos = [c for c in combos
                    if all(g in train_set for g in split_terms(c))]
    dev_combos = [c for c in combos
                  if any(g in dev_set for g in split_terms(c))
                  and not any(g in test_set for g in split_terms(c))]

    def phi(c):
        ha, hb = [norm(emb[g]) for g in split_terms(c)]
        return np.concatenate([ha * hb, ha, hb])

    Phi = (np.stack([phi(c) for c in train_combos]) if train_combos
           else np.zeros((0, 3 * dim)))
    Y = (np.stack([true_delta[c] - anchor(c) for c in train_combos])
         if train_combos else np.zeros((0, X.shape[1])))
    d = Phi.shape[1]
    best_lambda, best_error = LAMBDAS[0], np.inf
    if len(dev_combos) >= 3:
        Phi_dev = np.stack([phi(c) for c in dev_combos])
        Y_dev = np.stack([true_delta[c] - anchor(c) for c in dev_combos])
        rng_cv = np.random.RandomState(2000 + seed)
        folds = np.array_split(rng_cv.permutation(len(dev_combos)), 3)
        for lam in LAMBDAS:
            errors = []
            for fold in folds:
                keep = np.array([i for i in range(len(dev_combos))
                                 if i not in set(fold)])
                Xfit = np.vstack([Phi, Phi_dev[keep]])
                Yfit = np.vstack([Y, Y_dev[keep]])
                beta = ridge_solve(Xfit, Yfit, lam)
                for i in fold:
                    pred = anchor(dev_combos[i]) + torch_matvec(Phi_dev[i], beta)
                    residual = pred - true_delta[dev_combos[i]]
                    errors.append(float(np.sum(residual * residual)))
            err = float(np.mean(errors))
            if err < best_error:
                best_lambda, best_error = lam, err
    if len(Phi):
        Phi_all = (np.vstack([Phi, np.stack([phi(c) for c in dev_combos])])
                   if dev_combos else Phi)
        Y_all = (np.vstack([Y, np.stack([true_delta[c] - anchor(c)
                                         for c in dev_combos])])
                 if dev_combos else Y)
        beta = ridge_solve(Phi_all, Y_all, best_lambda)
    else:
        beta = np.zeros((d, X.shape[1]))

    # Ridge direction prior for genes without observed single-gene responses.
    tr_idx = [i for i, g in enumerate(all_genes) if g in single_map]
    Xt = np.stack([emb[all_genes[i]] for i in tr_idx]).astype(np.float64)
    Yt = np.stack([single_map[all_genes[i]] for i in tr_idx]).astype(np.float64)
    Dt = np.stack([norm(single_map[all_genes[i]].astype(np.float64))
                   for i in tr_idx])
    W_dir = ridge_solve(Xt, Dt, 10.0)

    def direction(g):
        if g in single_map:
            return norm(single_map[g].astype(np.float64))
        return norm(torch_matvec(W_dir.T, emb[g]))

    def combo_direction(c):
        return norm(np.sum([direction(g) for g in split_terms(c)], axis=0))

    def combo_direction_tensor(c):
        return torch.as_tensor(combo_direction(c), device=DEVICE,
                               dtype=torch.float32)

    def condition_feature(c):
        return np.sum([emb[g] for g in split_terms(c)], axis=0).astype(np.float32)

    train_singles = [x for x in singles
                     if split_terms(x)[0] in train_set
                     and split_terms(x)[0] in single_map]
    single_cells = {x: dense_sample(X, pert_idx[x], MAX_TRAIN_CELLS,
                                    500 + seed + i)
                    for i, x in enumerate(train_singles)}
    combo_cells = {c: dense_sample(X, pert_idx[c], MAX_TRAIN_CELLS,
                                   700 + seed + i)
                   for i, c in enumerate(train_combos)}
    flow_cells = {}
    for model_seed in MODEL_SEEDS:
        torch.manual_seed(seed * 100 + model_seed)
        np.random.seed(seed * 100 + model_seed)
        model = DCW1TV2(X.shape[1], dim, DEVICE, seed=model_seed, w1=True,
                        dir_weight=DIR_WEIGHT, mmd_weight=MMD_WEIGHT,
                        use_res_gate=False, dist_loss="mmd", res_eps=0.01)
        model.freeze_residual(True)
        pool = ([(x, "s") for x in train_singles] +
                [(c, "c") for c in train_combos])
        if not pool:
            continue
        for it in range(ITERS):
            if it == 100:
                model.freeze_residual(False)
            key, kind = pool[np.random.choice(len(pool))]
            if kind == "s":
                x1 = single_cells[key]
                g = split_terms(key)[0]
                feature = emb[g].astype(np.float32)
                prior = direction(g)
                target = single_map[g]
            else:
                x1 = combo_cells[key]
                feature = condition_feature(key)
                prior = combo_direction(key)
                target = true_delta[key]
            indices = np.random.choice(len(ctrl_cells),
                                       min(len(x1), BATCH), replace=True)
            x0 = ctrl_cells[indices]
            indices = np.random.choice(len(x1), min(len(x1), BATCH),
                                       replace=True)
            x1_batch = x1[indices]
            a = torch.as_tensor(x0, device=DEVICE, dtype=torch.float32)
            b = torch.as_tensor(x1_batch, device=DEVICE, dtype=torch.float32)
            row, col = linear_sum_assignment(torch.cdist(a, b).cpu().numpy())
            feature_t = torch.as_tensor(np.tile(feature, (len(row), 1)),
                                        device=DEVICE, dtype=torch.float32)
            target_norm = norm(np.asarray(target, dtype=np.float64))
            dir_target = torch.as_tensor(target_norm, device=DEVICE,
                                         dtype=torch.float32)
            dir_prior = torch.as_tensor(prior, device=DEVICE,
                                        dtype=torch.float32)
            model.train_step(a[row], b[col], feature_t, dir_target, dir_prior)
        flow_cells[model_seed] = {
            c: model.predict(ctrl_cells, condition_feature(c),
                             combo_direction_tensor(c)).cpu().numpy()
            for c in combos
        }

    rows = []
    vectors = {}
    test_combos = [c for c in combos
                   if any(g in test_set for g in split_terms(c))
                   and all(g not in train_set for g in split_terms(c))]
    for c in test_combos:
        fc = np.mean([flow_cells[s][c] for s in MODEL_SEEDS
                      if s in flow_cells], axis=0)
        fd = fc.mean(0) - ctrl_mean
        av = np.asarray(anchor(c), dtype=np.float64)
        es_delta = (av if NO_RESIDUAL
                    else av + torch_matvec(phi(c), beta))
        es_cells = fc + (es_delta - fd)
        true = true_delta[c]
        vectors[c] = {"true": true.tolist(), "add": av.tolist(),
                      "esct": es_delta.tolist(), "w1": es_cells.mean(0).tolist()}
        true_cells = dense_sample(X, pert_idx[c], N_CELLS,
                                  900 + seed * 100 + len(test_combos))
        rows.append({
            "combo": c,
            "add": metrics(av, true),
            "esct": metrics(es_delta, true),
            "w1": metrics(es_cells.mean(0) - ctrl_mean, true),
            "add_mmd": rbf_mmd(ctrl_cells + av, true_cells),
            "esct_mmd": rbf_mmd(es_cells, true_cells),
        })
    calibration = {"gamma": gamma, "lambda": best_lambda,
                   "n_train_bank": len(train_genes),
                   "n_dev_sims": len(dev_sims), "n_cat0": len(rows)}
    return calibration, rows, vectors


def main():
    name, out_path = sys.argv[1:]
    X, pert, emb = load_dataset(name)
    dim = int(emb[next(iter(emb))].shape[0])
    rows, vectors, calibrations = [], {}, {}
    for seed in range(1, N_SPLITS + 1):
        cal, split_rows, split_vectors = run_split(X, pert, emb, dim, seed)
        rows.extend(split_rows)
        vectors[str(seed)] = split_vectors
        calibrations[str(seed)] = cal
        print(f"{name} W1 split {seed}: gamma={cal['gamma']:.4f} "
              f"lambda={cal['lambda']} n={cal['n_cat0']}", flush=True)

    result = {"dataset": name, "n": len(rows), "calibrations": calibrations,
              "vectors_cat0": vectors, "rows": rows,
              "gamma_rule": "median dev top1",
              "ablation_no_retrieval": NO_RETRIEVAL,
              "ablation_no_residual": NO_RESIDUAL}
    methods = ["add", "esct", "w1"]
    for j, metric in enumerate(["pcc", "dir", "mean_l2"]):
        vals = {m: np.array([r[m][j] for r in rows]) for m in methods}
        result[metric] = {m: float(vals[m].mean()) for m in methods}
        for m in ["esct", "w1"]:
            hi_better = metric != "mean_l2"
            a, b = vals[m], vals["add"]
            result[metric][m + "_vs_add"] = {
                "p": float(st.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0,
                "win": int(np.sum((a > b) if hi_better else (a < b))),
            }
    result["mmd"] = {
        "add": float(np.mean([r["add_mmd"] for r in rows])),
        "w1": float(np.mean([r["esct_mmd"] for r in rows])),
        "p": float(st.wilcoxon([r["esct_mmd"] for r in rows],
                               [r["add_mmd"] for r in rows]).pvalue),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=json_default)
    print(json.dumps({k: result[k] for k in ["dataset", "n", "pcc", "dir",
                                             "mean_l2", "mmd"]}, indent=2))
    print("written:", out_path)


if __name__ == "__main__":
    main()
