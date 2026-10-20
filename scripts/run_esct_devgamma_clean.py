"""Leakage-clean ESCT v2.1 rerun with development-only gate calibration.

The interaction residual uses scGPT embeddings only.  The gate threshold is
recomputed independently in every gene split as the median top-1 cosine from
development-split genes to the training single-gene bank.  Test expression and
test labels are never used for fitting, feature construction, or calibration.
"""

import json
import os
import pickle
import sys

import anndata as ad
import numpy as np
import scipy.stats as st
from scipy.stats import pearsonr

from paths import dataset_paths


TOP_K = 100
N_SPLITS = 12
TRAIN_FRAC = 0.8
DEV_FRAC = 0.1
KNN_K = 5
KNN_TAU = 12.0
LAMBDAS = [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return str(o)


def norm(v):
    # np.linalg.norm can select a conflicting BLAS backend in mixed
    # torch/anndata environments; this element-wise reduction avoids it.
    return v / (np.sqrt(np.sum(v * v)) + 1e-8)


def sparse_mean(X, idx):
    return np.asarray(X[idx].mean(0)).ravel()


def pcc(pred, true):
    de = np.argsort(-np.abs(true))[:TOP_K]
    if np.std(pred[de]) < 1e-12 or np.std(true[de]) < 1e-12:
        return 0.0
    return float(pearsonr(pred[de], true[de])[0])


def dir_acc(pred, true):
    de = np.argsort(-np.abs(true))[:TOP_K]
    return float(np.mean(np.sign(pred[de]) == np.sign(true[de])))


def metrics(pred, true):
    return [pcc(pred, true), dir_acc(pred, true),
            float(np.sqrt(np.sum((pred - true) ** 2)))]


def load_dataset(name):
    path, emb_path = dataset_paths(name)
    a = ad.read_h5ad(path)
    if name.lower() == "norman":
        if "guide_identity" in a.obs:
            guides = a.obs["guide_identity"].astype(str).values

            def genes_from_guide(g):
                parts = g.replace("__", "_").split("_")
                return sorted({x for x in parts if not x.startswith("NegCtrl")})

            pert = np.array(["+".join(genes_from_guide(g)) if genes_from_guide(g)
                             else "control" for g in guides])
        elif "perturbation" in a.obs:
            # The released scPerturb Norman file stores the cleaned target
            # annotation directly, whereas the derived HVG file retains guide IDs.
            pert = a.obs["perturbation"].astype(str).values
        else:
            raise KeyError("Norman data require guide_identity or perturbation")
    else:
        pert = a.obs["perturbation"].astype(str).values

    with open(emb_path, "rb") as f:
        raw_emb = pickle.load(f)
    emb = {g: np.asarray(v, dtype=np.float64)
           for g, v in zip(raw_emb["genes"], raw_emb["emb"])}
    return a.X, pert, emb


def split_terms(label):
    return sorted(set(x for x in str(label).split("+") if x and x != "control"))


def run_split(X, pert, emb, dim, gene_seed):
    labels = sorted(set(pert) - {"control"})
    singles = [x for x in labels if len(split_terms(x)) == 1]
    combos = [x for x in labels if len(split_terms(x)) == 2
              and all(g in emb for g in split_terms(x))]
    all_genes = sorted({g for x in singles + combos for g in split_terms(x)
                        if g in emb})
    if not all_genes:
        raise RuntimeError("no embeddable perturbation genes")

    rng = np.random.RandomState(gene_seed)
    perm = rng.permutation(all_genes).tolist()
    n_tr = int(len(all_genes) * TRAIN_FRAC)
    n_dev = int(len(all_genes) * DEV_FRAC)
    train_set = set(perm[:n_tr])
    dev_set = set(perm[n_tr:n_tr + n_dev])
    test_set = set(perm[n_tr + n_dev:])

    pert_idx = {x: np.where(pert == x)[0] for x in labels}
    ctrl_idx = np.where(pert == "control")[0]
    ctrl_mean = sparse_mean(X, ctrl_idx)

    single_map = {}
    for x in singles:
        gs = split_terms(x)
        if len(gs) == 1 and gs[0] in train_set and gs[0] in emb:
            single_map[gs[0]] = sparse_mean(X, pert_idx[x]) - ctrl_mean
    global_delta = (np.mean(list(single_map.values()), axis=0)
                    if single_map else np.zeros(X.shape[1]))

    train_genes = sorted(single_map)
    if train_genes:
        bank_emb = np.stack([norm(emb[g]) for g in train_genes])
        bank_delta = np.stack([single_map[g] for g in train_genes])
    else:
        bank_emb = np.zeros((0, dim))
        bank_delta = np.zeros((0, X.shape[1]))

    def top1_sim(g):
        if not train_genes:
            return -1.0
        return float(bank_emb @ norm(emb[g]))

    # Development-only calibration: median of held-out dev genes' top-1 cosine.
    dev_sims = [float(np.max(bank_emb @ norm(emb[g]))) for g in sorted(dev_set)
                if g not in train_set and g in emb and len(train_genes)]
    gamma = float(np.median(dev_sims)) if dev_sims else 1.0

    def retrieve(g):
        if g in single_map:
            return single_map[g], 1.0
        if not train_genes:
            return global_delta, -1.0
        sims = bank_emb @ norm(emb[g])
        top1 = float(sims.max())
        nbr = np.argsort(-sims)[:KNN_K]
        weights = np.exp(KNN_TAU * sims[nbr])
        weights /= weights.sum() + 1e-12
        anchor = (weights[:, None] * bank_delta[nbr]).sum(0)
        return (anchor if top1 >= gamma else global_delta), top1

    def anchor(c):
        vals = [retrieve(g)[0] for g in split_terms(c)]
        return np.sum(vals, axis=0)

    true_delta = {c: sparse_mean(X, pert_idx[c]) - ctrl_mean for c in combos}
    train_combos = [c for c in combos
                    if all(g in train_set for g in split_terms(c))]
    dev_combos = [c for c in combos
                  if any(g in dev_set for g in split_terms(c))
                  and not any(g in test_set for g in split_terms(c))]

    def phi(c):
        ha, hb = [norm(emb[g]) for g in split_terms(c)]
        return np.concatenate([ha * hb, ha, hb])

    if train_combos:
        Phi = np.stack([phi(c) for c in train_combos])
        Y = np.stack([true_delta[c] - anchor(c) for c in train_combos])
    else:
        Phi = np.zeros((0, 3 * dim))
        Y = np.zeros((0, X.shape[1]))
    d = Phi.shape[1]

    best_lambda, best_error = LAMBDAS[0], np.inf
    if dev_combos and len(dev_combos) >= 3:
        Phi_dev = np.stack([phi(c) for c in dev_combos])
        Y_dev = np.stack([true_delta[c] - anchor(c) for c in dev_combos])
        rng_cv = np.random.RandomState(2000 + gene_seed)
        folds = np.array_split(rng_cv.permutation(len(dev_combos)), 3)
        for lam in LAMBDAS:
            errors = []
            for fold in folds:
                keep = np.array([i for i in range(len(dev_combos))
                                 if i not in set(fold)])
                Xfit = np.vstack([Phi, Phi_dev[keep]])
                Yfit = np.vstack([Y, Y_dev[keep]])
                beta = np.linalg.solve(Xfit.T @ Xfit + lam * np.eye(d),
                                       Xfit.T @ Yfit)
                for i in fold:
                    pred = anchor(dev_combos[i]) + Phi_dev[i] @ beta
                    errors.append(float(np.linalg.norm(
                        pred - true_delta[dev_combos[i]]) ** 2))
            err = float(np.mean(errors))
            if err < best_error:
                best_lambda, best_error = lam, err
    else:
        best_lambda = LAMBDAS[0]

    if len(Phi):
        Phi_all = np.vstack([Phi, np.stack([phi(c) for c in dev_combos])]) \
            if dev_combos else Phi
        Y_all = np.vstack([Y, np.stack([true_delta[c] - anchor(c)
                                        for c in dev_combos])]) \
            if dev_combos else Y
        beta = np.linalg.solve(Phi_all.T @ Phi_all + best_lambda * np.eye(d),
                               Phi_all.T @ Y_all)
    else:
        beta = np.zeros((d, X.shape[1]))

    result_rows = []
    vectors = {}
    for c in combos:
        if not (any(g in test_set for g in split_terms(c))
                and all(g not in train_set for g in split_terms(c))):
            continue
        av = np.asarray(anchor(c), dtype=np.float64)
        rv = (phi(c) @ beta).astype(np.float64)
        ev = av + rv
        tv = true_delta[c]
        vectors[c] = {"true": tv.tolist(), "add": av.tolist(),
                      "anchor": av.tolist(), "esct": ev.tolist()}
        result_rows.append({"combo": c, "add": metrics(av, tv),
                            "esct": metrics(ev, tv)})

    summary = {"gamma": gamma, "lambda": best_lambda,
               "n_train_genes": len(train_set), "n_dev_genes": len(dev_set),
               "n_test_genes": len(test_set), "n_train_bank": len(train_genes),
               "n_dev_sims": len(dev_sims), "n_cat0": len(result_rows)}
    return summary, result_rows, vectors


def main():
    if len(sys.argv) != 3:
        print("usage: python run_esct_devgamma_clean.py DATASET OUT_JSON")
        return
    name, out_path = sys.argv[1:]
    X, pert, emb = load_dataset(name)
    dim = int(emb[next(iter(emb))].shape[0])
    all_rows = []
    vectors = {}
    calibrations = {}
    for seed in range(1, N_SPLITS + 1):
        cal, rows, vec = run_split(X, pert, emb, dim, seed)
        all_rows.extend(rows)
        vectors[str(seed)] = vec
        calibrations[str(seed)] = cal
        print(f"{name} split {seed}: gamma={cal['gamma']:.4f} "
              f"lambda={cal['lambda']} n_cat0={cal['n_cat0']}", flush=True)

    add = np.array([r["add"] for r in all_rows])
    esct = np.array([r["esct"] for r in all_rows])
    result = {"dataset": name, "n": len(all_rows), "gamma_rule": "median dev top1",
              "calibrations": calibrations, "vectors_cat0": vectors,
              "rows": all_rows}
    for j, metric in enumerate(["pcc", "dir", "mean_l2"]):
        a, e = add[:, j], esct[:, j]
        result[metric] = {
            "add": float(a.mean()), "esct": float(e.mean()),
            "diff": float(np.mean(e - a)),
            "p": float(st.wilcoxon(e, a).pvalue) if np.any(e != a) else 1.0,
            "win": int(np.sum((e > a) if metric != "mean_l2" else (e < a))),
        }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=json_default)
    print(json.dumps({k: result[k] for k in ["dataset", "n", "pcc", "dir",
                                             "mean_l2"]}, indent=2))
    print("written:", out_path)


if __name__ == "__main__":
    main()
