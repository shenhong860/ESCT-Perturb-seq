"""Leakage-clean pre-experiment for response-module residual composition.

The model learns a low-rank response basis from training single-gene effects,
projects each observed combination residual onto that basis, and predicts only
the module coefficients for unseen combinations. Development combinations
select the rank, ridge penalty, and conservative blend with the ESCT anchor.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from run_esct_devgamma_clean import LAMBDAS, N_SPLITS, load_dataset, split_terms
from run_learned_composition_clean import build_split, fit_ridge, torch_ridge
from run_pair_retrieval_clean import (
    DEVICE,
    METRIC_NAMES,
    json_default,
    metrics,
    summarize,
)


RANKS = (8, 16, 32, 64, 128)
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def sparse_mean(matrix, indices):
    return np.asarray(matrix[indices].mean(0)).ravel()


def single_gene_deltas(matrix, perturbation):
    labels = sorted(set(perturbation) - {"control"})
    singles = [label for label in labels if len(split_terms(label)) == 1]
    control_indices = np.where(perturbation == "control")[0]
    control_mean = sparse_mean(matrix, control_indices)
    result = {}
    for label in singles:
        gene = split_terms(label)[0]
        indices = np.where(perturbation == label)[0]
        result[gene] = sparse_mean(matrix, indices) - control_mean
    return result


def response_basis(single_deltas, genes, rank):
    available = [gene for gene in sorted(genes) if gene in single_deltas]
    if not available:
        raise RuntimeError("no single-gene effects available for response basis")
    matrix = np.stack([single_deltas[gene] for gene in available])
    global_direction = matrix.mean(0)
    global_norm = float(np.sqrt(np.sum(global_direction ** 2)))
    if global_norm < 1e-12:
        global_unit = np.zeros(matrix.shape[1])
    else:
        global_unit = global_direction / global_norm

    centered = matrix - global_direction[None, :]
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    variable_rank = max(0, min(int(rank) - 1, len(singular_values)))
    columns = [global_unit] if global_norm >= 1e-12 else []
    if variable_rank:
        columns.extend(right[:variable_rank])
    if not columns:
        columns = [np.zeros(matrix.shape[1])]
    basis = np.stack(columns, axis=1)
    basis, _ = np.linalg.qr(basis)
    explained = float(
        np.sum(singular_values[:variable_rank] ** 2)
        / (np.sum(singular_values ** 2) + 1e-12)
    )
    return basis[:, : int(rank)], explained, len(available)


def select_module_model(context, single_deltas, dim, split_seed):
    train = context["train_combos"]
    dev = context["dev_combos"]
    phi_train = np.stack([context["phi"](label) for label in train])
    anchor_train = np.stack([context["anchor"](label) for label in train])
    residual_train = np.stack(
        [context["observed_delta"][label] - context["anchor"](label) for label in train]
    )
    phi_dev = (
        np.stack([context["phi"](label) for label in dev])
        if dev
        else np.zeros((0, 3 * dim))
    )
    anchor_dev = (
        np.stack([context["anchor"](label) for label in dev])
        if dev
        else np.zeros((0, residual_train.shape[1]))
    )
    observed_dev = (
        np.stack([context["observed_delta"][label] for label in dev])
        if dev
        else np.zeros((0, residual_train.shape[1]))
    )

    candidates = []
    basis_diagnostics = {}
    for rank in RANKS:
        basis, explained, n_basis_genes = response_basis(
            single_deltas, context["train_genes"], rank
        )
        basis_diagnostics[rank] = {
            "explained_variance": explained,
            "n_basis_genes": n_basis_genes,
            "actual_rank": int(basis.shape[1]),
        }
        module_train = residual_train @ basis
        for penalty in LAMBDAS:
            beta = torch_ridge(phi_train, module_train, penalty)
            module_dev = phi_dev @ beta
            reconstructed = module_dev @ basis.T
            for alpha in ALPHAS:
                prediction = anchor_dev + alpha * reconstructed
                if len(dev):
                    values = np.stack(
                        [metrics(prediction[i], observed_dev[i]) for i in range(len(dev))]
                    )
                    summary = {
                        "mean_pcc": float(values[:, 0].mean()),
                        "mean_dir": float(values[:, 1].mean()),
                        "mean_l2": float(values[:, 2].mean()),
                        "mean_recall_100": float(values[:, 3].mean()),
                        "mean_recall_200": float(values[:, 4].mean()),
                    }
                else:
                    error = float(np.mean((prediction - observed_dev) ** 2))
                    summary = {
                        "mean_pcc": -error,
                        "mean_dir": -error,
                        "mean_l2": error,
                        "mean_recall_100": -error,
                        "mean_recall_200": -error,
                    }
                candidates.append(
                    {
                        "rank": int(basis.shape[1]),
                        "penalty": float(penalty),
                        "alpha": float(alpha),
                        **summary,
                    }
                )

    objective_matrix = np.asarray(
        [
            [
                candidate["mean_pcc"],
                candidate["mean_dir"],
                -candidate["mean_l2"],
                0.5 * candidate["mean_recall_100"]
                + 0.5 * candidate["mean_recall_200"],
            ]
            for candidate in candidates
        ],
        dtype=np.float64,
    )
    scale = objective_matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (objective_matrix - objective_matrix.mean(axis=0)) / scale
    objective = standardized @ np.asarray([0.40, 0.30, 0.20, 0.10])
    for candidate, value in zip(candidates, objective):
        candidate["objective"] = float(value)

    best_objective = max(candidate["objective"] for candidate in candidates)
    near_best = [
        candidate
        for candidate in candidates
        if candidate["objective"] >= best_objective - 0.02
    ]
    selected = min(
        near_best,
        key=lambda candidate: (
            candidate["alpha"],
            -candidate["objective"],
            candidate["rank"],
            candidate["penalty"],
        ),
    )
    selected = dict(selected)
    selected["development_seed"] = split_seed
    return selected, basis_diagnostics


def evaluate_split(matrix, perturbation, embeddings, single_deltas, dim, split_seed):
    context = build_split(matrix, perturbation, embeddings, dim, split_seed)
    train = context["train_combos"]
    dev = context["dev_combos"]
    test = context["test_combos"]
    if not train or not test:
        return None

    selected, basis_diagnostics = select_module_model(
        context, single_deltas, dim, split_seed
    )

    phi_train = np.stack([context["phi"](label) for label in train])
    anchor_train = np.stack([context["anchor"](label) for label in train])
    residual_train = np.stack(
        [context["observed_delta"][label] - context["anchor"](label) for label in train]
    )
    phi_dev = (
        np.stack([context["phi"](label) for label in dev])
        if dev
        else np.zeros((0, 3 * dim))
    )
    anchor_dev = (
        np.stack([context["anchor"](label) for label in dev])
        if dev
        else np.zeros((0, residual_train.shape[1]))
    )
    residual_dev = (
        np.stack(
            [context["observed_delta"][label] - context["anchor"](label) for label in dev]
        )
        if dev
        else np.zeros((0, residual_train.shape[1]))
    )

    full_beta, full_penalty = fit_ridge(
        phi_train,
        residual_train,
        phi_dev,
        residual_dev,
    )

    final_genes = set(context["train_genes"]) | set(context["dev_genes"])
    final_basis, final_explained, final_basis_genes = response_basis(
        single_deltas, final_genes, selected["rank"]
    )
    final_phi = np.vstack([phi_train, phi_dev])
    final_residual = np.vstack([residual_train, residual_dev])
    module_beta = torch_ridge(
        final_phi,
        final_residual @ final_basis,
        selected["penalty"],
    )

    rows = []
    for label in test:
        observed = context["observed_delta"][label]
        anchor = context["anchor"](label)
        esct = anchor + context["phi"](label) @ full_beta
        module_prediction = anchor + selected["alpha"] * (
            (context["phi"](label) @ module_beta) @ final_basis.T
        )
        rows.append(
            {
                "combo": label,
                "strict_test_test": all(
                    gene in context["test_genes"] for gene in split_terms(label)
                ),
                "add": metrics(anchor, observed),
                "esct": metrics(esct, observed),
                "module": metrics(module_prediction, observed),
            }
        )

    calibration = {
        "selected": selected,
        "full_ridge_penalty": full_penalty,
        "final_response_basis": {
            "rank": int(final_basis.shape[1]),
            "explained_variance": final_explained,
            "n_basis_genes": final_basis_genes,
        },
        "basis_diagnostics": basis_diagnostics,
        "n_train_combos": len(train),
        "n_dev_combos": len(dev),
        "n_broad_test": len(test),
        "n_strict_test": sum(row["strict_test_test"] for row in rows),
    }
    return calibration, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["norman", "replogle", "wessels"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    matrix, perturbation, embeddings = load_dataset(args.dataset)
    single_deltas = single_gene_deltas(matrix, perturbation)
    dim = int(embeddings[next(iter(embeddings))].shape[0])
    rows = []
    calibrations = {}
    started = time.perf_counter()
    for split_seed in range(1, args.splits + 1):
        split_result = evaluate_split(
            matrix,
            perturbation,
            embeddings,
            single_deltas,
            dim,
            split_seed,
        )
        if split_result is None:
            print(f"{args.dataset} split {split_seed}: skipped", flush=True)
            continue
        calibration, split_rows = split_result
        calibrations[str(split_seed)] = calibration
        rows.extend(split_rows)
        selected = calibration["selected"]
        print(
            f"{args.dataset} split {split_seed}: "
            f"rank={selected['rank']} lambda={selected['penalty']} "
            f"alpha={selected['alpha']} n_broad={calibration['n_broad_test']} "
            f"n_strict={calibration['n_strict_test']}",
            flush=True,
        )

    strict_rows = [row for row in rows if row["strict_test_test"]]
    methods = ["add", "esct", "module"]
    result = {
        "dataset": args.dataset,
        "method": "response_module_composition",
        "device": DEVICE,
        "splits_requested": args.splits,
        "n_output_genes": int(matrix.shape[1]),
        "embedding_dim": dim,
        "broad": summarize(rows, methods),
        "strict": summarize(strict_rows, methods),
        "calibrations": calibrations,
        "rows": rows,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_gpu_mb": (
            float(torch.cuda.max_memory_allocated() / (1024 ** 2))
            if DEVICE == "cuda"
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, default=json_default)
    print(json.dumps({"broad": result["broad"], "strict": result["strict"]}, indent=2))
    print("written:", args.output)


if __name__ == "__main__":
    main()
