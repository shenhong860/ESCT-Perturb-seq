"""Official CAPRA on the clean 12-split Norman protocol.

Uses the released CAPRA source, GenePT embedding table and scPerturBench DEG
reference. The train/val/test condition sets are the same development-only
80/10/10 gene splits used by the clean ESCT rerun. Test labels enter only at
metric evaluation.
"""

import gc
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from paths import (capra_data_dir, capra_root, data_dir, deg_dir,
                   env_path, experiments_dir)

CAPRA_ROOT = capra_root()
sys.path.insert(0, str(CAPRA_ROOT))

from frame import CAPRA, CAPRAData, load_gene_embedding_table  # noqa: E402
from utils import canonical_condition_name  # noqa: E402

from run_esct_devgamma_clean import (  # noqa: E402
    DEV_FRAC, TRAIN_FRAC, dir_acc, pcc, split_terms,
)


RAW = env_path("ESCT_CAPRA_RAW", data_dir() / "Norman.h5ad")
HVG = env_path("ESCT_CAPRA_HVG", data_dir() / "norman" / "norman_hvg1000.h5ad")
EMB = env_path("ESCT_CAPRA_EMB",
               capra_data_dir() / "norman_example_genept_embeddings.pkl")
DEG = env_path("ESCT_CAPRA_DEG", deg_dir() / "DEG_hvg5000_Norman.pkl")
OUT = env_path("ESCT_CAPRA_OUT", experiments_dir() / "capra_clean_protocol.json")
CAPRA_RUN_DIR = env_path("ESCT_CAPRA_RUN_DIR",
                         experiments_dir() / "capra_runs_80")
N_SPLITS = int(os.environ.get("CAPRA_SPLITS", "12"))
N_EPOCHS = int(os.environ.get("CAPRA_EPOCHS", "80"))
N_PRED = int(os.environ.get("CAPRA_NPRED", "200"))
SEED = 24

if len(sys.argv) > 1:
    N_SPLITS = int(sys.argv[1])
if len(sys.argv) > 2:
    N_EPOCHS = int(sys.argv[2])


def metric_row(pred_delta, true_delta):
    return [pcc(pred_delta, true_delta), dir_acc(pred_delta, true_delta),
            float(np.sqrt(np.sum((pred_delta - true_delta) ** 2)))]


def main():
    raw = ad.read_h5ad(RAW)
    hvg = ad.read_h5ad(HVG)
    emb_table = load_gene_embedding_table(EMB, add_control=True,
                                          control_label="ctrl")
    hvg_to_raw = {g: i for i, g in enumerate(raw.var_names)}
    raw_to_hvg = np.array([i for i, g in enumerate(raw.var_names)
                           if g in set(hvg.var_names)], dtype=int)
    hvg_genes = list(hvg.var_names)

    raw_pert = raw.obs["perturbation"].astype(str).map(
        canonical_condition_name).to_numpy()
    ctrl_mask = raw_pert == "control"
    raw_ctrl_mean = np.asarray(raw.X[ctrl_mask].mean(0)).ravel()
    pert_idx = {}
    for label in np.unique(raw_pert):
        if label != "control":
            pert_idx[label] = np.where(raw_pert == label)[0]

    # Same perturbation-gene universe as the clean ESCT rerun.
    hvg_guides = hvg.obs["guide_identity"].astype(str).to_numpy()

    def genes_from_guide(g):
        parts = g.replace("__", "_").split("_")
        return sorted({x for x in parts if not x.startswith("NegCtrl")})

    hvg_pert = np.array(["+".join(genes_from_guide(g)) if genes_from_guide(g)
                         else "control" for g in hvg_guides])
    labels = sorted(set(hvg_pert) - {"control"})
    singles = [x for x in labels if len(split_terms(x)) == 1]
    combos = [x for x in labels if len(split_terms(x)) == 2]
    all_genes = sorted({g for x in singles + combos for g in split_terms(x)
                        if g in emb_table.index})
    print("clean genes:", len(all_genes), flush=True)

    all_rows = []
    for split_seed in range(1, N_SPLITS + 1):
        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(all_genes).tolist()
        n_tr = int(len(all_genes) * TRAIN_FRAC)
        n_dev = int(len(all_genes) * DEV_FRAC)
        train_set = set(perm[:n_tr])
        dev_set = set(perm[n_tr:n_tr + n_dev])
        test_set = set(perm[n_tr + n_dev:])

        train_conds = [c for c in labels
                       if all(g in train_set for g in split_terms(c))
                       and c in pert_idx]
        dev_conds = [c for c in labels
                     if any(g in dev_set for g in split_terms(c))
                     and not any(g in test_set for g in split_terms(c))
                     and c in pert_idx]
        test_conds = [c for c in combos
                      if any(g in test_set for g in split_terms(c))
                      and all(g not in train_set for g in split_terms(c))
                      and c in pert_idx]
        if not train_conds or not dev_conds or not test_conds:
            print(f"seed {split_seed}: skip "
                  f"train={len(train_conds)} dev={len(dev_conds)} "
                  f"test={len(test_conds)}", flush=True)
            continue

        capra_data = CAPRAData(
            adata=raw,
            study_name=f"esct_capra_seed{split_seed}",
            condition_key="perturbation",
            perturbation_key="perturbation",
            control_label="control",
        )
        capra_data.harmonize_perturbation_metadata(copy=False)
        capra_data.embedding_table = emb_table
        capra_data.register_evaluation_partitions(
            train_conds=train_conds,
            val_conds=dev_conds,
            test_conds=test_conds,
        )
        capra_data.build_control_relative_training_state(
            topk_deg=100,
            knn_topk=5,
            knn_temperature=12.0,
            deg_pickle_path=DEG,
        )

        model = CAPRA(capra_data)
        print(f"CAPRA fit split={split_seed} epochs={N_EPOCHS}", flush=True)
        model.fit_capra_response_operator(
            output_dir=CAPRA_RUN_DIR / f"seed{split_seed}",
            run_name=f"capra_seed{split_seed}",
            n_epochs=N_EPOCHS,
            min_epochs=min(20, N_EPOCHS),
            patience=min(10, N_EPOCHS),
            batch_size=192,
            learning_rate=6e-4,
            seed=SEED,
            num_workers=0,
        )
        predictions = model.generate_counterfactual_profiles(
            pert_list=test_conds, n_pred=N_PRED)

        for c in test_conds:
            pred = predictions[c].mean(0) - raw_ctrl_mean
            pred_hvg = pred[raw_to_hvg]
            true_hvg = (np.asarray(raw.X[pert_idx[c]].mean(0)).ravel() -
                        raw_ctrl_mean)[raw_to_hvg]
            if np.std(true_hvg) < 1e-12:
                continue
            all_rows.append({"seed": split_seed, "combo": c,
                             "metrics": metric_row(pred_hvg, true_hvg),
                             "n_train": len(train_conds),
                             "n_dev": len(dev_conds)})
        print(f"seed {split_seed}: n={len(test_conds)} done", flush=True)
        del capra_data, model, predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arr = np.array([r["metrics"] for r in all_rows])
    result = {"n": len(all_rows), "mean": {
        "pcc": float(arr[:, 0].mean()), "dir": float(arr[:, 1].mean()),
        "l2": float(arr[:, 2].mean())}, "rows": all_rows,
        "epochs": N_EPOCHS, "n_pred": N_PRED, "seed": SEED}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["mean"], indent=2))
    print("written:", OUT)


if __name__ == "__main__":
    main()
