"""Leakage-clean pre-experiment for agreement-gated PRR.

PRR augments the global ESCT ridge residual with a local residual retrieved
from observed gene combinations in a symmetric gene-pair embedding space.
The local contribution is shrunk by the directional agreement of the
retrieved residuals, so inconsistent neighbours fall back toward ESCT.
Development combinations choose k, temperature, and the blend strength before
the final ridge and retrieval bank are refitted on training+development.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch

from run_esct_devgamma_clean import LAMBDAS, N_SPLITS, load_dataset, norm, split_terms
from run_learned_composition_clean import build_split, fit_ridge, torch_ridge


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 100
METRIC_NAMES = ["pcc", "dir", "mean_l2", "recall_100", "recall_200"]
K_GRID = (1, 3, 5, 8)
TEMPERATURE_GRID = (4.0, 8.0, 12.0, 20.0)
ALPHA_GRID = (0.1, 0.25, 0.5, 0.75, 1.0)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return str(value)


def pcc(predicted, observed):
    selected = np.argsort(-np.abs(observed))[:TOP_K]
    if np.std(predicted[selected]) < 1e-12 or np.std(observed[selected]) < 1e-12:
        return 0.0
    return float(st.pearsonr(predicted[selected], observed[selected])[0])


def direction_accuracy(predicted, observed):
    selected = np.argsort(-np.abs(observed))[:TOP_K]
    return float(np.mean(np.sign(predicted[selected]) == np.sign(observed[selected])))


def deg_recall(predicted, observed, top_k):
    observed_top = np.argsort(-np.abs(observed))[:top_k]
    predicted_top = np.argsort(-np.abs(predicted))[:top_k]
    return float(len(np.intersect1d(observed_top, predicted_top)) / top_k)


def metrics(predicted, observed):
    return [
        pcc(predicted, observed),
        direction_accuracy(predicted, observed),
        float(np.sqrt(np.sum((predicted - observed) ** 2))),
        deg_recall(predicted, observed, 100),
        deg_recall(predicted, observed, 200),
    ]


def pair_signature(label, embeddings):
    first, second = [norm(embeddings[gene]) for gene in split_terms(label)]
    return norm(np.concatenate([first * second, np.abs(first - second), first + second]))


def select_ridge_penalty(features, targets, seed):
    """Choose a ridge penalty using training combinations only."""
    best_penalty = LAMBDAS[0]
    if len(features) < 6:
        return best_penalty
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(features)), 3)
    best_error = np.inf
    for penalty in LAMBDAS:
        errors = []
        for fold in folds:
            keep = np.setdiff1d(np.arange(len(features)), fold)
            beta = torch_ridge(features[keep], targets[keep], penalty)
            residual = features[fold] @ beta - targets[fold]
            errors.append(float(np.mean(residual ** 2)))
        error = float(np.mean(errors))
        if error < best_error:
            best_penalty = penalty
            best_error = error
    return best_penalty


def fit_development_model(context, dim, split_seed):
    """Fit the provisional model using training-label information only."""
    train = context["train_combos"]
    features = np.stack([context["phi"](label) for label in train])
    targets = np.stack(
        [context["observed_delta"][label] - context["anchor"](label) for label in train]
    )
    penalty = select_ridge_penalty(features, targets, 5000 + split_seed)
    beta = torch_ridge(features, targets, penalty)
    return beta, penalty


def local_residual(query, bank_signatures, bank_residuals, k, temperature):
    similarities = bank_signatures @ query
    k = min(int(k), len(similarities))
    neighbours = np.argsort(-similarities)[:k]
    shifted = temperature * (similarities[neighbours] - similarities[neighbours[0]])
    weights = np.exp(shifted)
    weights /= weights.sum() + 1e-12
    local = (weights[:, None] * bank_residuals[neighbours]).sum(0)
    weighted_norm = float(
        np.sum(weights * np.sqrt(np.sum(bank_residuals[neighbours] ** 2, axis=1)))
    )
    agreement = float(np.sqrt(np.sum(local ** 2)) / (weighted_norm + 1e-12))
    return (
        local,
        float(similarities[neighbours[0]]),
        min(max(agreement, 0.0), 1.0),
    )


def tune_prr(context, beta, embeddings, split_seed):
    dev = context["dev_combos"]
    if not dev:
        return {"k": 1, "temperature": 4.0, "alpha": 0.0, "objective": 0.0}, []

    train = context["train_combos"]
    bank_signatures = np.stack(
        [pair_signature(label, embeddings) for label in train]
    )
    bank_residuals = np.stack(
        [context["observed_delta"][label] - context["anchor"](label) for label in train]
    )
    dev_signatures = np.stack([pair_signature(label, embeddings) for label in dev])
    dev_anchor = np.stack([context["anchor"](label) for label in dev])
    dev_observed = np.stack([context["observed_delta"][label] for label in dev])
    dev_ridge = dev_anchor + np.stack([context["phi"](label) for label in dev]) @ beta

    candidates = []
    for k in K_GRID:
        for temperature in TEMPERATURE_GRID:
            local = []
            top_similarities = []
            agreements = []
            for signature in dev_signatures:
                residual, top_similarity, agreement = local_residual(
                    signature, bank_signatures, bank_residuals, k, temperature
                )
                local.append(residual)
                top_similarities.append(top_similarity)
                agreements.append(agreement)
            local = np.stack(local)
            agreements = np.asarray(agreements)
            ridge_residual = dev_ridge - dev_anchor
            for alpha in ALPHA_GRID:
                effective_alpha = alpha * agreements[:, None]
                prediction = dev_anchor + ridge_residual + effective_alpha * (
                    local - ridge_residual
                )
                values = np.stack(
                    [metrics(prediction[i], dev_observed[i]) for i in range(len(dev))]
                )
                candidates.append(
                    {
                        "k": int(k),
                        "temperature": float(temperature),
                        "alpha": float(alpha),
                        "mean_pcc": float(values[:, 0].mean()),
                        "mean_dir": float(values[:, 1].mean()),
                        "mean_l2": float(values[:, 2].mean()),
                        "mean_recall_100": float(values[:, 3].mean()),
                        "mean_recall_200": float(values[:, 4].mean()),
                        "mean_top_similarity": float(np.mean(top_similarities)),
                        "mean_agreement": float(agreements.mean()),
                        "mean_effective_alpha": float(effective_alpha.mean()),
                    }
                )

    # The objective is fixed before seeing test labels.  Each metric is scaled
    # across the pre-specified grid so no metric dominates by raw magnitude.
    objective_matrix = []
    for candidate in candidates:
        objective_matrix.append(
            [
                candidate["mean_pcc"],
                candidate["mean_dir"],
                -candidate["mean_l2"],
                0.5 * candidate["mean_recall_100"] + 0.5 * candidate["mean_recall_200"],
            ]
        )
    objective_matrix = np.asarray(objective_matrix, dtype=np.float64)
    scale = objective_matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (objective_matrix - objective_matrix.mean(axis=0)) / scale
    weights = np.asarray([0.40, 0.30, 0.20, 0.10])
    objective = standardized @ weights
    for candidate, value in zip(candidates, objective):
        candidate["objective"] = float(value)

    # Prefer a smaller alpha when objectives are practically tied.  This is a
    # deterministic one-standard-error-style safeguard against dev overfitting.
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
            candidate["k"],
            candidate["temperature"],
        ),
    )
    selected = dict(selected)
    selected["development_seed"] = split_seed
    return selected, candidates


def predict_prr(context, beta, embeddings, final_bank_labels, hyperparameters, labels):
    bank_signatures = np.stack(
        [pair_signature(label, embeddings) for label in final_bank_labels]
    )
    bank_residuals = np.stack(
        [
            context["observed_delta"][label] - context["anchor"](label)
            for label in final_bank_labels
        ]
    )
    predictions = {}
    diagnostics = {}
    for label in labels:
        anchor = context["anchor"](label)
        ridge_residual = context["phi"](label) @ beta
        local, top_similarity, agreement = local_residual(
            pair_signature(label, embeddings),
            bank_signatures,
            bank_residuals,
            hyperparameters["k"],
            hyperparameters["temperature"],
        )
        alpha = hyperparameters["alpha"]
        effective_alpha = alpha * agreement
        predictions[label] = (
            anchor + ridge_residual + effective_alpha * (local - ridge_residual)
        )
        diagnostics[label] = {
            "top_similarity": top_similarity,
            "agreement": agreement,
            "effective_alpha": effective_alpha,
            "local_residual_norm": float(np.linalg.norm(local)),
            "ridge_residual_norm": float(np.linalg.norm(ridge_residual)),
        }
    return predictions, diagnostics


def evaluate_split(matrix, perturbation, embeddings, dim, split_seed):
    context = build_split(matrix, perturbation, embeddings, dim, split_seed)
    train = context["train_combos"]
    dev = context["dev_combos"]
    test = context["test_combos"]
    if not train or not test:
        return None

    provisional_beta, provisional_penalty = fit_development_model(
        context, dim, split_seed
    )
    hyperparameters, dev_candidates = tune_prr(
        context, provisional_beta, embeddings, split_seed
    )

    phi_train = np.stack([context["phi"](label) for label in train])
    y_train = np.stack(
        [context["observed_delta"][label] - context["anchor"](label) for label in train]
    )
    phi_dev = (
        np.stack([context["phi"](label) for label in dev])
        if dev
        else np.zeros((0, 3 * dim))
    )
    y_dev = (
        np.stack(
            [context["observed_delta"][label] - context["anchor"](label) for label in dev]
        )
        if dev
        else np.zeros((0, matrix.shape[1]))
    )
    final_beta, final_penalty = fit_ridge(phi_train, y_train, phi_dev, y_dev)
    final_bank = train + dev
    prr_predictions, diagnostics = predict_prr(
        context,
        final_beta,
        embeddings,
        final_bank,
        hyperparameters,
        test,
    )

    rows = []
    for label in test:
        observed = context["observed_delta"][label]
        additive = context["anchor"](label)
        esct = additive + context["phi"](label) @ final_beta
        rows.append(
            {
                "combo": label,
                "strict_test_test": all(
                    gene in context["test_genes"] for gene in split_terms(label)
                ),
                "add": metrics(additive, observed),
                "esct": metrics(esct, observed),
                "prr": metrics(prr_predictions[label], observed),
                "prr_diagnostics": diagnostics[label],
            }
        )

    calibration = {
        "gamma": context["gamma"],
        "provisional_ridge_penalty": provisional_penalty,
        "final_ridge_penalty": final_penalty,
        "selected": hyperparameters,
        "n_train_combos": len(train),
        "n_dev_combos": len(dev),
        "n_final_bank": len(final_bank),
        "n_broad_test": len(test),
        "n_strict_test": sum(row["strict_test_test"] for row in rows),
        "n_dev_grid_candidates": len(dev_candidates),
    }
    return calibration, rows


def summarize(rows, methods):
    if not rows:
        return {"n": 0}
    result = {"n": len(rows)}
    for index, name in enumerate(METRIC_NAMES):
        result[name] = {
            method: float(np.mean([row[method][index] for row in rows]))
            for method in methods
        }
        higher_is_better = name != "mean_l2"
        for method in methods[1:]:
            baseline = np.asarray([row["esct"][index] for row in rows])
            candidate = np.asarray([row[method][index] for row in rows])
            additive = np.asarray([row["add"][index] for row in rows])
            result[name][method + "_vs_esct"] = {
                "p": (
                    float(st.wilcoxon(candidate, baseline).pvalue)
                    if np.any(candidate != baseline)
                    else 1.0
                ),
                "win": int(
                    np.sum(
                        (candidate > baseline)
                        if higher_is_better
                        else (candidate < baseline)
                    )
                ),
            }
            result[name][method + "_vs_add"] = {
                "p": (
                    float(st.wilcoxon(candidate, additive).pvalue)
                    if np.any(candidate != additive)
                    else 1.0
                ),
                "win": int(
                    np.sum(
                        (candidate > additive)
                        if higher_is_better
                        else (candidate < additive)
                    )
                ),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["norman", "replogle", "wessels"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    matrix, perturbation, embeddings = load_dataset(args.dataset)
    dim = int(embeddings[next(iter(embeddings))].shape[0])
    all_rows = []
    calibrations = {}
    started = time.perf_counter()
    for split_seed in range(1, args.splits + 1):
        split_result = evaluate_split(
            matrix, perturbation, embeddings, dim, split_seed
        )
        if split_result is None:
            print(f"{args.dataset} split {split_seed}: skipped", flush=True)
            continue
        calibration, rows = split_result
        calibrations[str(split_seed)] = calibration
        all_rows.extend(rows)
        selected = calibration["selected"]
        print(
            f"{args.dataset} split {split_seed}: "
            f"k={selected['k']} tau={selected['temperature']} "
            f"alpha={selected['alpha']} lambda={calibration['final_ridge_penalty']} "
            f"n_broad={calibration['n_broad_test']} "
            f"n_strict={calibration['n_strict_test']}",
            flush=True,
        )

    strict_rows = [row for row in all_rows if row["strict_test_test"]]
    methods = ["add", "esct", "prr"]
    result = {
        "dataset": args.dataset,
        "method": "agreement_gated_pair_retrieval_residual",
        "agreement_gated": True,
        "device": DEVICE,
        "splits_requested": args.splits,
        "n_output_genes": int(matrix.shape[1]),
        "embedding_dim": dim,
        "broad": summarize(all_rows, methods),
        "strict": summarize(strict_rows, methods),
        "calibrations": calibrations,
        "rows": all_rows,
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
