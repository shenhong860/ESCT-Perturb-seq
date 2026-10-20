"""Joint mean-and-transport baseline on the leakage-clean 12-split protocol.

This baseline removes ESCT's closed-form ridge stage. A condition network and
the cell transport head are optimized together from the same training and
development perturbations. Test labels are never used for fitting or model
selection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from esct_model import ESCTModel
from run_esct_devgamma_clean import (
    DEV_FRAC,
    KNN_K,
    KNN_TAU,
    N_SPLITS as DEFAULT_N_SPLITS,
    TRAIN_FRAC,
    load_dataset,
    metrics,
    norm,
    split_terms,
)
from run_esct_w1_devgamma_clean import dense_sample, json_default, sparse_mean


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 100
N_SPLITS = int(os.environ.get("ESCT_SPLITS", str(DEFAULT_N_SPLITS)))
MODEL_SEEDS = [int(x) for x in os.environ.get("ESCT_MODEL_SEEDS", "0,1").split(",")]
ITERS = int(os.environ.get("ESCT_ITERS", "500"))
BATCH = int(os.environ.get("ESCT_BATCH", "128"))
N_CELLS = int(os.environ.get("ESCT_N_CELLS", "200"))
MAX_TRAIN_CELLS = int(os.environ.get("ESCT_MAX_TRAIN_CELLS", "300"))
MEAN_WEIGHT = float(os.environ.get("ESCT_E2E_MEAN_WEIGHT", "1.0"))
DIAGNOSTICS_DIR = os.environ.get("ESCT_DIAGNOSTICS_DIR", "")


class MeanResidualHead(nn.Module):
    """Feature-conditioned residual used in place of the ridge estimator."""

    def __init__(self, feature_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def rbf_mmd(x: np.ndarray, y: np.ndarray, sigma: float = 2.0) -> float:
    xt = torch.as_tensor(x, device=DEVICE, dtype=torch.float32)
    yt = torch.as_tensor(y, device=DEVICE, dtype=torch.float32)
    xx = torch.exp(-torch.cdist(xt, xt) / (2 * sigma * sigma))
    yy = torch.exp(-torch.cdist(yt, yt) / (2 * sigma * sigma))
    xy = torch.exp(-torch.cdist(xt, yt) / (2 * sigma * sigma))
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def condition_feature(label: str, emb: dict[str, np.ndarray], dim: int) -> np.ndarray:
    terms = split_terms(label)
    if len(terms) == 1:
        h = norm(emb[terms[0]])
        return np.concatenate([np.zeros(dim), h, np.zeros(dim)]).astype(np.float32)
    ha, hb = [norm(emb[g]) for g in terms]
    return np.concatenate([ha * hb, ha, hb]).astype(np.float32)


def condition_embedding(label: str, emb: dict[str, np.ndarray]) -> np.ndarray:
    return np.sum([emb[g] for g in split_terms(label)], axis=0).astype(np.float32)


def joint_train_step(
    model: ESCTModel,
    mean_head: MeanResidualHead,
    optimizer: torch.optim.Optimizer,
    x0: torch.Tensor,
    x1: torch.Tensor,
    condition: torch.Tensor,
    mean_feature: torch.Tensor,
    anchor: torch.Tensor,
    target_delta: torch.Tensor,
) -> tuple[float, float]:
    optimizer.zero_grad()
    g = model.g(condition)
    delta = x1 - x0
    mean_delta = anchor + mean_head(mean_feature)
    mean_direction = mean_delta / (mean_delta.norm(dim=1, keepdim=True) + 1e-6)

    t = torch.rand(len(x0), 1, device=model.device)
    s = torch.clamp(t + 0.05, max=1.0)
    xt = x0 + t * delta
    xs = x0 + s * delta
    h0 = model.cell_enc(xt)
    hs = model.cell_enc(xs)
    vt = model.velocity(h0, t, g, mean_direction)
    with torch.no_grad():
        vs = model.velocity(hs, s, g, mean_direction)
        fs = xs + (1.0 - s) * vs
    ft = xt + (1.0 - t) * vt

    loss = ((vt - delta) ** 2).mean()
    loss = loss + ((ft - fs) ** 2).mean() + ((vt - vs) ** 2).mean()
    if model.mmd_weight > 0:
        from esct_model import rbf_mmd as differentiable_rbf_mmd

        loss = loss + model.mmd_weight * differentiable_rbf_mmd(x0 + vt, x1)

    target_norm = delta.norm(dim=1, keepdim=True) + 1e-6
    direction_target = delta / target_norm
    w = target_norm.squeeze(1) / target_norm.sum()
    loss = loss - model.dir_weight * (
        (vt / (vt.norm(dim=1, keepdim=True) + 1e-6) * direction_target).sum(1) * w
    ).mean()

    if model.orth_weight > 0:
        residual = model.vel(h0, t, g)
        residual = residual / (residual.norm(dim=1, keepdim=True) + 1e-6)
        loss = loss + model.orth_weight * (
            (residual * mean_direction).sum(1) ** 2
        ).mean()
    if model.res_l2_weight > 0:
        loss = loss + model.res_l2_weight * model.vel(h0, t, g).pow(2).mean()
    loss = loss + model.aux_weight * ((model.aux_dec(model.cell_enc(x0)) - x0) ** 2).mean()
    mean_loss = ((mean_delta - target_delta) ** 2).mean()
    loss = loss + MEAN_WEIGHT * mean_loss

    pot_value = None
    if model.w1:
        direction = model.w1_dir(x0, condition)
        loss = loss + 0.1 * ((direction_target - direction) ** 2).mean()
        h0_pot = model.pot(x0.detach(), condition.detach()).mean()
        h1_pot = model.pot(x1.detach(), condition.detach()).mean()
        pot = -(h0_pot - h1_pot)
        model.opt_pot.zero_grad()
        pot.backward()
        model.opt_pot.step()
        pot_value = float(pot.item())

    loss.backward()
    optimizer.step()
    model.ema_update()
    return float(loss.item()), float(mean_loss.item())


def run_split(name: str, X, pert, emb, dim: int, seed: int):
    labels = sorted(set(pert) - {"control"})
    singles = [x for x in labels if len(split_terms(x)) == 1]
    combos = [
        x for x in labels
        if len(split_terms(x)) == 2 and all(g in emb for g in split_terms(x))
    ]
    genes = sorted(
        {g for x in singles + combos for g in split_terms(x) if g in emb}
    )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(genes).tolist()
    n_train = int(len(genes) * TRAIN_FRAC)
    n_dev = int(len(genes) * DEV_FRAC)
    train_set = set(perm[:n_train])
    dev_set = set(perm[n_train:n_train + n_dev])
    test_set = set(perm[n_train + n_dev:])

    pert_idx = {x: np.where(pert == x)[0] for x in labels}
    ctrl_idx = np.where(pert == "control")[0]
    ctrl_mean = sparse_mean(X, ctrl_idx)
    ctrl_cells = dense_sample(X, ctrl_idx, N_CELLS, 42)

    single_map = {}
    for label in singles:
        terms = split_terms(label)
        if len(terms) == 1 and terms[0] in train_set and terms[0] in emb:
            single_map[terms[0]] = sparse_mean(X, pert_idx[label]) - ctrl_mean
    global_delta = (
        np.mean(list(single_map.values()), axis=0)
        if single_map else np.zeros(X.shape[1])
    )
    train_genes = sorted(single_map)
    bank_emb = np.stack([norm(emb[g]) for g in train_genes])
    bank_delta = np.stack([single_map[g] for g in train_genes])
    dev_sims = [
        float(np.max(bank_emb @ norm(emb[g])))
        for g in sorted(dev_set) if g in emb
    ]
    gamma = float(np.median(dev_sims)) if dev_sims else 1.0

    def retrieve(gene: str) -> np.ndarray:
        if gene in single_map:
            return single_map[gene]
        sims = bank_emb @ norm(emb[gene])
        if float(sims.max()) < gamma:
            return global_delta
        neighbors = np.argsort(-sims)[:KNN_K]
        weights = np.exp(KNN_TAU * sims[neighbors])
        weights /= weights.sum() + 1e-12
        return (weights[:, None] * bank_delta[neighbors]).sum(0)

    def anchor(label: str) -> np.ndarray:
        return np.sum([retrieve(g) for g in split_terms(label)], axis=0)

    train_combos = [
        c for c in combos if all(g in train_set for g in split_terms(c))
    ]
    dev_combos = [
        c for c in combos
        if any(g in dev_set for g in split_terms(c))
        and not any(g in test_set for g in split_terms(c))
    ]
    true_delta = {
        label: sparse_mean(X, pert_idx[label]) - ctrl_mean for label in labels
    }

    train_singles = [
        label for label in singles
        if split_terms(label)[0] in train_set and split_terms(label)[0] in single_map
    ]
    supervised = train_singles + train_combos + dev_combos
    cells = {
        label: dense_sample(
            X,
            pert_idx[label],
            MAX_TRAIN_CELLS,
            500 + seed + index,
        )
        for index, label in enumerate(supervised)
    }
    features = {
        label: condition_feature(label, emb, dim) for label in supervised
    }
    embeddings = {
        label: condition_embedding(label, emb) for label in supervised
    }
    anchors = {label: anchor(label) for label in supervised}

    seed_means = {}
    seed_cells = {}
    for model_seed in MODEL_SEEDS:
        torch.manual_seed(seed * 1000 + model_seed)
        np.random.seed(seed * 1000 + model_seed)
        model = ESCTModel(
            X.shape[1],
            dim,
            DEVICE,
            seed=model_seed,
            w1=True,
            dir_weight=2.0,
            mmd_weight=10.0,
            use_res_gate=False,
            dist_loss="mmd",
            res_eps=0.01,
        )
        mean_head = MeanResidualHead(3 * dim, X.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(mean_head.parameters()), lr=3e-4
        )
        for iteration in range(ITERS):
            label = supervised[np.random.randint(len(supervised))]
            x1_all = cells[label]
            batch_size = min(len(x1_all), BATCH)
            x0_index = np.random.choice(len(ctrl_cells), batch_size, replace=True)
            x1_index = np.random.choice(len(x1_all), batch_size, replace=True)
            x0_np = ctrl_cells[x0_index]
            x1_np = x1_all[x1_index]
            a = torch.as_tensor(x0_np, device=DEVICE, dtype=torch.float32)
            b = torch.as_tensor(x1_np, device=DEVICE, dtype=torch.float32)
            rows, cols = linear_sum_assignment(torch.cdist(a, b).cpu().numpy())
            condition = torch.as_tensor(
                np.tile(embeddings[label], (len(rows), 1)),
                device=DEVICE,
                dtype=torch.float32,
            )
            mean_feature = torch.as_tensor(
                np.tile(features[label], (len(rows), 1)),
                device=DEVICE,
                dtype=torch.float32,
            )
            anchor_t = torch.as_tensor(
                np.tile(anchors[label], (len(rows), 1)),
                device=DEVICE,
                dtype=torch.float32,
            )
            target_t = torch.as_tensor(
                np.tile(true_delta[label], (len(rows), 1)),
                device=DEVICE,
                dtype=torch.float32,
            )
            joint_train_step(
                model,
                mean_head,
                optimizer,
                a[rows],
                b[cols],
                condition,
                mean_feature,
                anchor_t,
                target_t,
            )
            if (iteration + 1) % 100 == 0:
                print(
                    f"{name} split {seed} seed {model_seed} "
                    f"iter {iteration + 1}/{ITERS}",
                    flush=True,
                )

        with torch.no_grad():
            ctrl_t = torch.as_tensor(ctrl_cells, device=DEVICE, dtype=torch.float32)
            predictions = {}
            predicted_means = {}
            for label in combos:
                feature = torch.as_tensor(
                    features.get(label, condition_feature(label, emb, dim)),
                    device=DEVICE,
                    dtype=torch.float32,
                ).unsqueeze(0).expand(len(ctrl_t), -1)
                anchor_t = torch.as_tensor(
                    anchors.get(label, anchor(label)),
                    device=DEVICE,
                    dtype=torch.float32,
                ).unsqueeze(0).expand(len(ctrl_t), -1)
                mean_delta = anchor_t + mean_head(feature)
                direction = mean_delta / (mean_delta.norm(dim=1, keepdim=True) + 1e-6)
                condition = torch.as_tensor(
                    embeddings.get(label, condition_embedding(label, emb)),
                    device=DEVICE,
                    dtype=torch.float32,
                ).unsqueeze(0).expand(len(ctrl_t), -1)
                g = model.g(condition)
                h = model.cell_enc(ctrl_t)
                zeros = torch.zeros(len(ctrl_t), 1, device=DEVICE)
                mapped = ctrl_t + model.velocity(h, zeros, g, direction)
                predicted_means[label] = mean_delta[0].cpu().numpy()
                predictions[label] = mapped.cpu().numpy()
        seed_means[model_seed] = predicted_means
        seed_cells[model_seed] = predictions

    rows = []
    vectors = {}
    diagnostics = []
    test_combos = [
        c for c in combos
        if any(g in test_set for g in split_terms(c))
        and all(g not in train_set for g in split_terms(c))
    ]
    for label in test_combos:
        mean_delta = np.mean(
            [seed_means[model_seed][label] for model_seed in MODEL_SEEDS], axis=0
        )
        raw_cells = np.mean(
            [seed_cells[model_seed][label] for model_seed in MODEL_SEEDS], axis=0
        )
        calibrated = raw_cells + (
            mean_delta - (raw_cells.mean(axis=0) - ctrl_mean)
        )
        truth = true_delta[label]
        true_cells = dense_sample(
            X,
            pert_idx[label],
            N_CELLS,
            900 + seed * 100 + len(test_combos),
        )
        response_genes = np.argsort(-np.abs(truth))[:TOP_K]
        additive = anchor(label)
        rows.append(
            {
                "combo": label,
                "add": metrics(additive, truth),
                "e2e": metrics(mean_delta, truth),
                "e2e_cell_deg100_mmd": rbf_mmd(
                    calibrated[:, response_genes], true_cells[:, response_genes]
                ),
                "add_cell_deg100_mmd": rbf_mmd(
                    (ctrl_cells + additive)[:, response_genes],
                    true_cells[:, response_genes],
                ),
            }
        )
        vectors[label] = {
            "true": truth.tolist(),
            "add": additive.tolist(),
            "e2e": mean_delta.tolist(),
        }
        diagnostics.append(
            {
                "combo": label,
                "ctrl": ctrl_cells,
                "raw": raw_cells,
                "calibrated": calibrated,
                "true": true_cells,
                "mean_delta": mean_delta,
            }
        )

    if DIAGNOSTICS_DIR and diagnostics:
        output_dir = Path(DIAGNOSTICS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {"ctrl_cells": ctrl_cells, "combo_names": np.asarray(test_combos)}
        diagnostic_rows = []
        for index, item in enumerate(diagnostics):
            prefix = f"combo_{index}"
            arrays[f"{prefix}_raw"] = item["raw"]
            arrays[f"{prefix}_calibrated"] = item["calibrated"]
            arrays[f"{prefix}_true"] = item["true"]
            arrays[f"{prefix}_mean_delta"] = item["mean_delta"]
            raw_mean = item["raw"].mean(axis=0) - ctrl_mean
            post_mean = item["calibrated"].mean(axis=0) - ctrl_mean
            diagnostic_rows.append(
                {
                    "combo": item["combo"],
                    "raw_mean_residual_l2": float(
                        np.sqrt(np.sum((raw_mean - item["mean_delta"]) ** 2))
                    ),
                    "calibrated_mean_residual_l2": float(
                        np.sqrt(np.sum((post_mean - item["mean_delta"]) ** 2))
                    ),
                    "map_displacement_mean_l2": float(
                        np.mean(
                            np.sqrt(
                                np.sum((item["calibrated"] - ctrl_cells) ** 2, axis=1)
                            )
                        )
                    ),
                }
            )
        np.savez_compressed(output_dir / f"{name}_split{seed}_e2e_map.npz", **arrays)
        (output_dir / f"{name}_split{seed}_e2e_map.json").write_text(
            json.dumps(
                {"dataset": name, "split": seed, "rows": diagnostic_rows},
                indent=2,
            )
        )

    calibration = {
        "gamma": gamma,
        "n_train_bank": len(train_genes),
        "n_dev_sims": len(dev_sims),
        "n_cat0": len(rows),
        "n_supervised_conditions": len(supervised),
    }
    return calibration, rows, vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["norman", "replogle", "wessels"])
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, pert, emb = load_dataset(args.dataset)
    dim = int(emb[next(iter(emb))].shape[0])
    rows = []
    vectors = {}
    calibrations = {}
    start = time.perf_counter()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for split in range(1, N_SPLITS + 1):
        calibration, split_rows, split_vectors = run_split(
            args.dataset, X, pert, emb, dim, split
        )
        rows.extend(split_rows)
        vectors[str(split)] = split_vectors
        calibrations[str(split)] = calibration
        print(
            f"{args.dataset} E2E split {split}: "
            f"n_cat0={calibration['n_cat0']}",
            flush=True,
        )

    result = {
        "dataset": args.dataset,
        "n": len(rows),
        "calibrations": calibrations,
        "vectors_cat0": vectors,
        "rows": rows,
        "method": "joint end-to-end neural mean and transport",
        "model_seeds": MODEL_SEEDS,
        "iterations": ITERS,
        "runtime_seconds": float(time.perf_counter() - start),
        "splits_run": N_SPLITS,
        "device": DEVICE,
        "peak_gpu_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if DEVICE == "cuda" else None
        ),
    }
    for index, metric in enumerate(["pcc", "dir", "mean_l2"]):
        additive = np.asarray([row["add"][index] for row in rows])
        end_to_end = np.asarray([row["e2e"][index] for row in rows])
        result[metric] = {
            "add": float(additive.mean()),
            "e2e": float(end_to_end.mean()),
            "diff": float(np.mean(end_to_end - additive)),
            "p": (
                float(st.wilcoxon(end_to_end, additive).pvalue)
                if np.any(end_to_end != additive) else 1.0
            ),
        }
    additive_mmd = np.asarray([row["add_cell_deg100_mmd"] for row in rows])
    e2e_mmd = np.asarray([row["e2e_cell_deg100_mmd"] for row in rows])
    result["cell_deg100_mmd"] = {
        "add": float(additive_mmd.mean()),
        "e2e": float(e2e_mmd.mean()),
        "diff": float(np.mean(e2e_mmd - additive_mmd)),
        "p": (
            float(st.wilcoxon(e2e_mmd, additive_mmd).pvalue)
            if np.any(e2e_mmd != additive_mmd) else 1.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=json_default))
    print(json.dumps(
        {key: result[key] for key in [
            "dataset", "n", "pcc", "dir", "mean_l2", "cell_deg100_mmd",
            "runtime_seconds", "peak_gpu_mb",
        ]},
        indent=2,
    ))


if __name__ == "__main__":
    main()
