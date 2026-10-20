"""Compute official scPerturBench DEG recall and DEG-restricted MMD.

The reference DEG tables are the released ``DEG_hvg5000_*.pkl`` files.
Recall is normalized by the number of official DEGs present in each retained
expression panel. Coverage is reported separately so panels containing only a
subset of the official DEGs cannot be mistaken for full DEG recovery.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

from paths import data_dir, dataset_paths, deg_dir, env_path, experiments_dir


NORMAN_REFERENCE = dataset_paths("norman")[0]
REPLOGLE_REFERENCE = dataset_paths("replogle")[0]
WESSELS_REFERENCE = dataset_paths("wessels")[0]
DATASETS = {
    "Norman_1k": {
        "matrix": NORMAN_REFERENCE,
        "predictions": experiments_dir() / "esct_w1_devgamma_clean_norman.json",
        "deg": deg_dir() / "DEG_hvg5000_Norman.pkl",
    },
    "Norman_5025": {
        "matrix": NORMAN_REFERENCE,
        "predictions": experiments_dir() / "norman_5025.json",
        "deg": deg_dir() / "DEG_hvg5000_Norman.pkl",
    },
    "Norman_20000": {
        "matrix": env_path("ESCT_NORMAN_20K_H5AD", data_dir() / "norman_20k.h5ad"),
        "predictions": experiments_dir() / "norman_20000.json",
        "deg": deg_dir() / "DEG_hvg5000_Norman.pkl",
    },
    "Norman_33694": {
        "matrix": env_path("ESCT_NORMAN_FULL_H5AD", data_dir() / "norman_full_panel.h5ad"),
        "predictions": experiments_dir() / "norman_33694.json",
        "deg": deg_dir() / "DEG_hvg5000_Norman.pkl",
    },
    "Replogle_exp6": {
        "matrix": REPLOGLE_REFERENCE,
        "predictions": experiments_dir() / "esct_w1_devgamma_clean_replogle.json",
        "deg": deg_dir() / "DEG_hvg5000_Replogle_exp6.pkl",
    },
    "Wessels": {
        "matrix": WESSELS_REFERENCE,
        "predictions": experiments_dir() / "esct_w1_devgamma_clean_wessels.json",
        "deg": deg_dir() / "DEG_hvg5000_Wessels.pkl",
    },
}
KS = (100, 200)
SIGMA = 2.0


def symbol_rank(values: np.ndarray, groups: np.ndarray, first_indices: np.ndarray,
                k: int) -> np.ndarray:
    """Rank unique gene symbols, collapsing duplicate feature names by max |value|."""
    scores = np.full(first_indices.size, -np.inf, dtype=np.float64)
    np.maximum.at(scores, groups, np.abs(values))
    k = min(k, scores.size)
    top_groups = np.argpartition(scores, -k)[-k:]
    return first_indices[top_groups]


def laplacian_mmd(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.shape[0] < 2 or true.shape[0] < 2:
        return float("nan")
    xx = np.exp(-cdist(pred, pred, metric="euclidean") / (2 * SIGMA**2))
    yy = np.exp(-cdist(true, true, metric="euclidean") / (2 * SIGMA**2))
    xy = np.exp(-cdist(pred, true, metric="euclidean") / (2 * SIGMA**2))
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def paired_p(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if not np.any(keep) or np.allclose(x[keep], y[keep]):
        return 1.0
    return float(wilcoxon(x[keep], y[keep]).pvalue)


def evaluate(name: str, config: dict[str, Path]) -> dict:
    matrix = ad.read_h5ad(config["matrix"], backed="r")
    genes = [str(gene) for gene in matrix.var_names]
    matrix.file.close()
    unique_genes = list(dict.fromkeys(genes))
    gene_to_group = {gene: index for index, gene in enumerate(unique_genes)}
    gene_to_first = {}
    for index, gene in enumerate(genes):
        gene_to_first.setdefault(gene, index)
    groups = np.asarray([gene_to_group[gene] for gene in genes], dtype=np.int64)
    first_indices = np.asarray([gene_to_first[gene] for gene in unique_genes],
                               dtype=np.int64)
    with config["deg"].open("rb") as handle:
        deg_tables = pickle.load(handle)
    payload = json.loads(config["predictions"].read_text())

    recalls = {
        f"{method}_recall_at_{k}": []
        for method in ("add", "esct")
        for k in KS
    }
    hits_of_k = {
        f"{method}_hits_of_{k}": []
        for method in ("add", "esct")
        for k in KS
    }
    coverage = {k: [] for k in KS}
    mmd = {k: {"add": [], "esct": []} for k in KS}
    n_combinations = 0

    for combinations in payload["vectors_cat0"].values():
        if not combinations:
            continue
        official_indices = {k: [] for k in KS}
        for combo, vectors in combinations.items():
            if combo not in deg_tables:
                continue
            n_combinations += 1
            truth = np.asarray(vectors["true"], dtype=np.float64)
            for k in KS:
                official = set(str(gene) for gene in deg_tables[combo].index[:k])
                indices = np.asarray(
                    [gene_to_first[gene] for gene in official if gene in gene_to_first],
                    dtype=np.int64,
                )
                official_indices[k].extend(indices.tolist())
                available = len(indices)
                coverage[k].append(available / min(k, len(genes)))
                truth_set = set(indices.tolist())
                for method in ("add", "esct"):
                    prediction = np.asarray(vectors[method], dtype=np.float64)
                    predicted = set(
                        symbol_rank(prediction, groups, first_indices, k).tolist()
                    )
                    hits = len(truth_set & predicted)
                    hits_of_k[f"{method}_hits_of_{k}"].append(
                        hits / min(k, len(genes))
                    )
                    recalls[f"{method}_recall_at_{k}"].append(
                        hits / available if available else float("nan")
                    )

        for k in KS:
            union = np.unique(np.asarray(official_indices[k], dtype=np.int64))
            if union.size < 2:
                continue
            true_rows = np.stack(
                [np.asarray(v["true"])[union] for v in combinations.values()]
            )
            add_rows = np.stack(
                [np.asarray(v["add"])[union] for v in combinations.values()]
            )
            esct_rows = np.stack(
                [np.asarray(v["esct"])[union] for v in combinations.values()]
            )
            mmd[k]["add"].append(laplacian_mmd(add_rows, true_rows))
            mmd[k]["esct"].append(laplacian_mmd(esct_rows, true_rows))

    result = {"dataset": name, "n_combinations": n_combinations}
    for k in KS:
        result[f"coverage_at_{k}"] = {
            "mean": float(np.nanmean(coverage[k])),
            "median_official_degs_available": float(
                np.nanmedian(np.asarray(coverage[k]) * min(k, len(genes)))
            ),
        }
        for method in ("add", "esct"):
            result[f"{method}_recall_at_{k}"] = {
                "mean": float(np.nanmean(recalls[f"{method}_recall_at_{k}"])),
            }
            result[f"{method}_hits_of_{k}"] = {
                "mean": float(np.nanmean(hits_of_k[f"{method}_hits_of_{k}"])),
            }
        result[f"recall_at_{k}_p"] = paired_p(
            recalls[f"esct_recall_at_{k}"], recalls[f"add_recall_at_{k}"]
        )
        result[f"mmd_at_{k}"] = {
            "add": float(np.nanmean(mmd[k]["add"])),
            "esct": float(np.nanmean(mmd[k]["esct"])),
            "p": paired_p(mmd[k]["esct"], mmd[k]["add"]),
            "n_splits": int(np.isfinite(mmd[k]["add"]).sum()),
        }
    return result


def write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "# Official scPerturBench DEG metrics",
        "",
        "Reference DEGs are read from the released `DEG_hvg5000_*.pkl` tables.",
        "Recall is normalized by the official DEGs available in the retained",
        "expression panel; coverage reports that available fraction separately.",
        "MMD is condition-level over the union of official DEGs within each split.",
        "",
        "| Dataset | n | DEG coverage@100 | R@100 add | R@100 ESCT | p | coverage@200 | R@200 add | R@200 ESCT | p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset']} | {row['n_combinations']} | "
            f"{row['coverage_at_100']['mean']:.3f} | "
            f"{row['add_recall_at_100']['mean']:.3f} | "
            f"{row['esct_recall_at_100']['mean']:.3f} | "
            f"{row['recall_at_100_p']:.3g} | "
            f"{row['coverage_at_200']['mean']:.3f} | "
            f"{row['add_recall_at_200']['mean']:.3f} | "
            f"{row['esct_recall_at_200']['mean']:.3f} | "
            f"{row['recall_at_200_p']:.3g} |"
        )
    lines += [
        "",
        "| Dataset | DEG-MMD@100 add | DEG-MMD@100 ESCT | p | DEG-MMD@200 add | DEG-MMD@200 ESCT | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['dataset']} | {row['mmd_at_100']['add']:.5f} | "
            f"{row['mmd_at_100']['esct']:.5f} | {row['mmd_at_100']['p']:.3g} | "
            f"{row['mmd_at_200']['add']:.5f} | {row['mmd_at_200']['esct']:.5f} | "
            f"{row['mmd_at_200']['p']:.3g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=experiments_dir() / "official_deg_metrics")
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.dataset or list(DATASETS)
    results = [evaluate(name, DATASETS[name]) for name in selected]
    (args.output_dir / "official_deg_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_markdown(results, args.output_dir / "official_deg_metrics.md")
    print((args.output_dir / "official_deg_metrics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
