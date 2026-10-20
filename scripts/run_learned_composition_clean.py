"""Evaluate a learned vector-composition extension of clean ESCT."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy.stats as st
import torch

from run_esct_devgamma_clean import (
    DEV_FRAC,
    KNN_K,
    KNN_TAU,
    LAMBDAS,
    N_SPLITS,
    TRAIN_FRAC,
    load_dataset,
    norm,
    split_terms,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 100
COMPOSITION_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return str(value)


def sparse_mean(matrix, indices):
    return np.asarray(matrix[indices].mean(0)).ravel()


def pcc(predicted, observed, top_k=TOP_K):
    selected = np.argsort(-np.abs(observed))[:top_k]
    if np.std(predicted[selected]) < 1e-12 or np.std(observed[selected]) < 1e-12:
        return 0.0
    return float(st.pearsonr(predicted[selected], observed[selected])[0])


def direction_accuracy(predicted, observed, top_k=TOP_K):
    selected = np.argsort(-np.abs(observed))[:top_k]
    return float(np.mean(np.sign(predicted[selected]) == np.sign(observed[selected])))


def metrics(predicted, observed):
    return [
        pcc(predicted, observed),
        direction_accuracy(predicted, observed),
        float(np.sqrt(np.sum((predicted - observed) ** 2))),
    ]


def torch_ridge(features, targets, penalty):
    x = torch.as_tensor(features, device=DEVICE, dtype=torch.float64)
    y = torch.as_tensor(targets, device=DEVICE, dtype=torch.float64)
    system = x.T @ x + penalty * torch.eye(x.shape[1], device=DEVICE, dtype=torch.float64)
    return torch.linalg.solve(system, x.T @ y).cpu().numpy()


def torch_dual_ridge(features, targets, penalty):
    x = torch.as_tensor(features, device=DEVICE, dtype=torch.float64)
    y = torch.as_tensor(targets, device=DEVICE, dtype=torch.float64)
    system = x @ x.T + penalty * torch.eye(x.shape[0], device=DEVICE, dtype=torch.float64)
    dual = torch.linalg.solve(system, y)
    return (x.T @ dual).cpu().numpy()


def fit_ridge(phi_train, y_train, phi_dev, y_dev):
    best_penalty = LAMBDAS[0]
    best_error = np.inf
    if len(y_dev) >= 3:
        folds = np.array_split(np.arange(len(y_dev)), 3)
        for penalty in LAMBDAS:
            errors = []
            for fold in folds:
                keep = np.setdiff1d(np.arange(len(y_dev)), fold)
                x_fit = np.vstack([phi_train, phi_dev[keep]])
                y_fit = np.vstack([y_train, y_dev[keep]])
                beta = torch_ridge(x_fit, y_fit, penalty)
                residual = phi_dev[fold] @ beta - y_dev[fold]
                errors.append(float(np.mean(residual ** 2)))
            error = float(np.mean(errors))
            if error < best_error:
                best_penalty = penalty
                best_error = error
    x_fit = np.vstack([phi_train, phi_dev]) if len(phi_dev) else phi_train
    y_fit = np.vstack([y_train, y_dev]) if len(y_dev) else y_train
    return torch_ridge(x_fit, y_fit, best_penalty), best_penalty


def fit_composition(feature_train, y_train, feature_dev, y_dev):
    best_penalty = COMPOSITION_LAMBDAS[0]
    best_error = np.inf
    if len(y_dev) >= 3:
        folds = np.array_split(np.arange(len(y_dev)), 3)
        for penalty in COMPOSITION_LAMBDAS:
            errors = []
            for fold in folds:
                keep = np.setdiff1d(np.arange(len(y_dev)), fold)
                x_fit = np.vstack([feature_train, feature_dev[keep]])
                y_fit = np.vstack([y_train, y_dev[keep]])
                beta = torch_dual_ridge(x_fit, y_fit, penalty)
                residual = feature_dev[fold] @ beta - y_dev[fold]
                errors.append(float(np.mean(residual ** 2)))
            error = float(np.mean(errors))
            if error < best_error:
                best_penalty = penalty
                best_error = error
    x_fit = np.vstack([feature_train, feature_dev]) if len(feature_dev) else feature_train
    y_fit = np.vstack([y_train, y_dev]) if len(y_dev) else y_train
    return torch_dual_ridge(x_fit, y_fit, best_penalty), best_penalty


def build_features(phi, first_delta, second_delta):
    first = norm(first_delta)
    second = norm(second_delta)
    signed_product = first * second
    magnitude_product = np.abs(first) * np.abs(second)
    return {
        "signed": np.concatenate([phi, signed_product]),
        "signed_abs": np.concatenate(
            [phi, signed_product, magnitude_product]
        ),
        "product": np.concatenate([signed_product, magnitude_product]),
    }


def build_split(matrix, perturbation, embeddings, dim, split_seed):
    labels = sorted(set(perturbation) - {"control"})
    singles = [label for label in labels if len(split_terms(label)) == 1]
    combos = [
        label
        for label in labels
        if len(split_terms(label)) == 2
        and all(gene in embeddings for gene in split_terms(label))
    ]
    all_genes = sorted(
        {
            gene
            for label in singles + combos
            for gene in split_terms(label)
            if gene in embeddings
        }
    )
    rng = np.random.RandomState(split_seed)
    permuted = rng.permutation(all_genes).tolist()
    n_train = int(len(all_genes) * TRAIN_FRAC)
    n_dev = int(len(all_genes) * DEV_FRAC)
    train_genes = set(permuted[:n_train])
    dev_genes = set(permuted[n_train:n_train + n_dev])
    test_genes = set(permuted[n_train + n_dev:])

    perturbation_indices = {
        label: np.where(perturbation == label)[0] for label in labels
    }
    control_indices = np.where(perturbation == "control")[0]
    control_mean = sparse_mean(matrix, control_indices)

    single_map = {}
    for label in singles:
        genes = split_terms(label)
        if len(genes) == 1 and genes[0] in train_genes and genes[0] in embeddings:
            single_map[genes[0]] = (
                sparse_mean(matrix, perturbation_indices[label]) - control_mean
            )

    global_delta = (
        np.mean(list(single_map.values()), axis=0)
        if single_map
        else np.zeros(matrix.shape[1])
    )
    bank_genes = sorted(single_map)
    bank_embeddings = (
        np.stack([norm(embeddings[gene]) for gene in bank_genes])
        if bank_genes
        else np.zeros((0, dim))
    )
    bank_deltas = (
        np.stack([single_map[gene] for gene in bank_genes])
        if bank_genes
        else np.zeros((0, matrix.shape[1]))
    )
    dev_similarities = [
        float(np.max(bank_embeddings @ norm(embeddings[gene])))
        for gene in sorted(dev_genes)
        if gene not in train_genes and gene in embeddings and len(bank_genes)
    ]
    gamma = float(np.median(dev_similarities)) if dev_similarities else 1.0

    def retrieve(gene):
        if gene in single_map:
            return single_map[gene], 1.0
        if not bank_genes:
            return global_delta, -1.0
        similarities = bank_embeddings @ norm(embeddings[gene])
        top1 = float(similarities.max())
        neighbours = np.argsort(-similarities)[:KNN_K]
        weights = np.exp(KNN_TAU * similarities[neighbours])
        weights /= weights.sum() + 1e-12
        anchor = (weights[:, None] * bank_deltas[neighbours]).sum(0)
        return (anchor if top1 >= gamma else global_delta), top1

    def pair_components(label):
        genes = split_terms(label)
        return retrieve(genes[0])[0], retrieve(genes[1])[0]

    def anchor(label):
        first, second = pair_components(label)
        return first + second

    def phi(label):
        first, second = [norm(embeddings[gene]) for gene in split_terms(label)]
        return np.concatenate([first * second, first, second])

    observed_delta = {
        label: sparse_mean(matrix, perturbation_indices[label]) - control_mean
        for label in combos
    }
    train_combos = [
        label
        for label in combos
        if all(gene in train_genes for gene in split_terms(label))
    ]
    dev_combos = [
        label
        for label in combos
        if any(gene in dev_genes for gene in split_terms(label))
        and not any(gene in test_genes for gene in split_terms(label))
    ]
    broad_test_combos = [
        label
        for label in combos
        if any(gene in test_genes for gene in split_terms(label))
        and all(gene not in train_genes for gene in split_terms(label))
    ]

    return {
        "train_genes": train_genes,
        "dev_genes": dev_genes,
        "test_genes": test_genes,
        "gamma": gamma,
        "train_combos": train_combos,
        "dev_combos": dev_combos,
        "test_combos": broad_test_combos,
        "anchor": anchor,
        "phi": phi,
        "pair_components": pair_components,
        "observed_delta": observed_delta,
    }


def evaluate_split(matrix, perturbation, embeddings, dim, split_seed):
    context = build_split(matrix, perturbation, embeddings, dim, split_seed)
    train = context["train_combos"]
    dev = context["dev_combos"]
    test = context["test_combos"]
    if not train:
        return None

    phi_train = np.stack([context["phi"](label) for label in train])
    phi_dev = (
        np.stack([context["phi"](label) for label in dev])
        if dev
        else np.zeros((0, 3 * dim))
    )
    y_train = np.stack(
        [context["observed_delta"][label] - context["anchor"](label) for label in train]
    )
    y_dev = (
        np.stack(
            [context["observed_delta"][label] - context["anchor"](label) for label in dev]
        )
        if dev
        else np.zeros((0, matrix.shape[1]))
    )
    ridge_beta, ridge_penalty = fit_ridge(phi_train, y_train, phi_dev, y_dev)

    feature_sets = {"signed": [], "signed_abs": [], "product": []}
    for label in train:
        first, second = context["pair_components"](label)
        features = build_features(context["phi"](label), first, second)
        for key in feature_sets:
            feature_sets[key].append(features[key])
    features_train = {key: np.stack(value) for key, value in feature_sets.items()}

    feature_sets = {"signed": [], "signed_abs": [], "product": []}
    for label in dev:
        first, second = context["pair_components"](label)
        features = build_features(context["phi"](label), first, second)
        for key in feature_sets:
            feature_sets[key].append(features[key])
    features_dev = {}
    widths = {
        "signed": (3 * dim) + matrix.shape[1],
        "signed_abs": (3 * dim) + 2 * matrix.shape[1],
        "product": 2 * matrix.shape[1],
    }
    for key, value in feature_sets.items():
        features_dev[key] = np.stack(value) if value else np.zeros((0, widths[key]))

    feature_models = {}
    for key in feature_sets:
        beta, penalty = fit_composition(
            features_train[key], y_train, features_dev[key], y_dev
        )
        feature_models[key] = (beta, penalty)

    rows = []
    for label in test:
        observed = context["observed_delta"][label]
        additive = context["anchor"](label)
        ridge = additive + context["phi"](label) @ ridge_beta
        first, second = context["pair_components"](label)
        test_features = build_features(context["phi"](label), first, second)
        compose_predictions = {
            key: additive + test_features[key] @ beta
            for key, (beta, _) in feature_models.items()
        }
        rows.append(
            {
                "combo": label,
                "strict_test_test": all(
                    gene in context["test_genes"] for gene in split_terms(label)
                ),
                "add": metrics(additive, observed),
                "esct": metrics(ridge, observed),
                **{
                    key: metrics(value, observed)
                    for key, value in compose_predictions.items()
                },
            }
        )
    calibration = {
        "gamma": context["gamma"],
        "ridge_penalty": ridge_penalty,
        "composition_penalties": {
            key: value[1] for key, value in feature_models.items()
        },
        "n_train_combos": len(train),
        "n_dev_combos": len(dev),
        "n_test_combos": len(test),
    }
    return calibration, rows


def summarize(rows, method_names):
    result = {"n": len(rows)}
    for metric_index, metric_name in enumerate(["pcc", "dir", "mean_l2"]):
        result[metric_name] = {
            method: float(np.mean([row[method][metric_index] for row in rows]))
            for method in method_names
        }
        higher_is_better = metric_name != "mean_l2"
        for method in method_names[1:]:
            first = np.asarray([row["add"][metric_index] for row in rows])
            second = np.asarray([row[method][metric_index] for row in rows])
            result[metric_name][method + "_vs_add"] = {
                "p": (
                    float(st.wilcoxon(second, first).pvalue)
                    if np.any(second != first)
                    else 1.0
                ),
                "win": int(
                    np.sum((second > first) if higher_is_better else (second < first))
                ),
            }
            if method != "esct":
                baseline = np.asarray([row["esct"][metric_index] for row in rows])
                result[metric_name][method + "_vs_esct"] = {
                    "p": (
                        float(st.wilcoxon(second, baseline).pvalue)
                        if np.any(second != baseline)
                        else 1.0
                    ),
                    "win": int(
                        np.sum(
                            (second > baseline)
                            if higher_is_better
                            else (second < baseline)
                        )
                    ),
                }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["norman", "replogle", "wessels"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--embedding-pkl", type=Path)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    matrix, perturbation, embeddings = load_dataset(args.dataset)
    if args.embedding_pkl is not None:
        with args.embedding_pkl.open("rb") as handle:
            payload = pickle.load(handle)
        embeddings = {
            str(gene): np.asarray(vector, dtype=np.float64)
            for gene, vector in zip(payload["genes"], payload["emb"])
        }
    dim = int(embeddings[next(iter(embeddings))].shape[0])

    rows = []
    calibrations = {}
    started = time.perf_counter()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for split_seed in range(1, args.splits + 1):
        split_result = evaluate_split(
            matrix, perturbation, embeddings, dim, split_seed
        )
        if split_result is None:
            continue
        calibration, split_rows = split_result
        calibrations[str(split_seed)] = calibration
        rows.extend(split_rows)
        print(
            f"{args.dataset} split {split_seed}: "
            f"ridge_lambda={calibration['ridge_penalty']} "
            f"composition={calibration['composition_penalties']} "
            f"n={len(split_rows)}",
            flush=True,
        )

    methods = ["add", "esct", "signed", "signed_abs", "product"]
    strict_rows = [row for row in rows if row["strict_test_test"]]
    result = {
        "dataset": args.dataset,
        "device": DEVICE,
        "splits_run": args.splits,
        "embedding_dim": dim,
        "n_output_genes": int(matrix.shape[1]),
        "n": len(rows),
        "n_strict": len(strict_rows),
        "broad": summarize(rows, methods),
        "strict": summarize(strict_rows, methods) if strict_rows else {"n": 0},
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
