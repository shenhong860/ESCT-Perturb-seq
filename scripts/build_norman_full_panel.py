"""Build a label-free Norman output panel for the dimension stress tests.

The released scPerturb Norman matrix contains the 5,025 HVGs used in the primary
analysis. This script starts from the full filtered GSE133344 count matrix and
retains the same 96,994 cells as the released matrix. Genes are ranked by total
counts in unperturbed cells only; perturbation labels are used solely to identify
controls and to ensure that every modelled perturbation gene is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

from paths import data_dir, dataset_paths, env_path, experiments_dir


DEFAULT_RAW = env_path(
    "ESCT_NORMAN_RAW_MTX", data_dir() / "norman/GSE133344_filtered_matrix.mtx.gz"
)
DEFAULT_GENES = env_path(
    "ESCT_NORMAN_RAW_GENES", data_dir() / "norman/GSE133344_filtered_genes.tsv.gz"
)
DEFAULT_BARCODES = env_path(
    "ESCT_NORMAN_RAW_BARCODES", data_dir() / "norman/GSE133344_filtered_barcodes.tsv.gz"
)
DEFAULT_REFERENCE = dataset_paths("norman")[0]
DEFAULT_OUTPUT = env_path(
    "ESCT_NORMAN_FULL_H5AD", experiments_dir() / "norman_full_panel.h5ad"
)
DEFAULT_METADATA = experiments_dir() / "norman_full_panel_build.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--genes", type=Path, default=DEFAULT_GENES)
    parser.add_argument("--barcodes", type=Path, default=DEFAULT_BARCODES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--n-genes", type=int, default=33_694)
    parser.add_argument("--target-sum", type=float, default=10_000.0)
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**3), b""):
            digest.update(block)
    return digest.hexdigest()


def perturbation_genes(perturbations: pd.Series) -> set[str]:
    genes: set[str] = set()
    for label in perturbations.astype(str):
        if label != "control":
            genes.update(g for g in label.split("+") if g)
    return genes


def select_genes(control_counts: np.ndarray, symbols: np.ndarray,
                 required: set[str], n_genes: int):
    if n_genes > len(symbols):
        raise ValueError(f"requested {n_genes} genes from {len(symbols)} rows")
    if n_genes == len(symbols):
        selected = np.arange(len(symbols), dtype=np.int64)
        metadata = {
            "required_perturbation_genes": len(required),
            "required_perturbation_genes_missing_from_matrix": sorted(required - set(symbols)),
            "required_perturbation_genes_inserted": 0,
            "selected_positive_control_genes": int(np.sum(control_counts[selected] > 0)),
        }
        return selected, metadata
    order = np.argsort(-control_counts, kind="stable")
    selected: list[int] = []
    seen: set[str] = set()
    for index in order:
        symbol = str(symbols[index])
        if symbol not in seen:
            selected.append(int(index))
            seen.add(symbol)
        if len(selected) == n_genes:
            break
    if len(selected) < n_genes:
        raise RuntimeError(f"only {len(selected)} unique genes available")

    symbol_to_index = {str(symbol): i for i, symbol in enumerate(symbols)}
    unavailable = sorted(g for g in required if g not in symbol_to_index)
    inserted = 0
    for gene in sorted(required - seen):
        if gene in unavailable:
            continue
        replacement = len(selected) - 1
        while replacement >= 0 and str(symbols[selected[replacement]]) in required:
            replacement -= 1
        if replacement < 0:
            raise RuntimeError("cannot retain all required perturbation genes")
        selected[replacement] = symbol_to_index[gene]
        inserted += 1
    selected = np.asarray(selected, dtype=np.int64)
    metadata = {
        "required_perturbation_genes": len(required),
        "required_perturbation_genes_missing_from_matrix": unavailable,
        "required_perturbation_genes_inserted": inserted,
        "selected_positive_control_genes": int(np.sum(control_counts[selected] > 0)),
    }
    return selected, metadata


def validate_against_reference(built: ad.AnnData, reference_path: Path,
                               seed: int = 0) -> dict[str, object]:
    reference = ad.read_h5ad(reference_path)
    common_cells = np.intersect1d(built.obs_names.astype(str),
                                  reference.obs_names.astype(str))
    common_genes = np.intersect1d(built.var_names.astype(str),
                                  reference.var_names.astype(str))
    if not len(common_cells) or not len(common_genes):
        raise RuntimeError("no overlap with the released 5,025-gene matrix")
    rng = np.random.RandomState(seed)
    cell_idx = rng.choice(len(common_cells), min(128, len(common_cells)), replace=False)
    gene_idx = rng.choice(len(common_genes), min(256, len(common_genes)), replace=False)
    built_gene_lookup = {}
    for index, gene in enumerate(built.var_names.astype(str)):
        built_gene_lookup.setdefault(gene, index)
    reference_gene_lookup = {}
    for index, gene in enumerate(reference.var_names.astype(str)):
        reference_gene_lookup.setdefault(gene, index)
    selected_genes = common_genes[gene_idx]
    built_gene_idx = np.asarray([built_gene_lookup[g] for g in selected_genes])
    reference_gene_idx = np.asarray([reference_gene_lookup[g] for g in selected_genes])
    built_cell_lookup = {cell: index for index, cell in enumerate(built.obs_names.astype(str))}
    reference_cell_lookup = {cell: index for index, cell in enumerate(reference.obs_names.astype(str))}
    selected_cells = common_cells[cell_idx]
    built_cell_idx = np.asarray([built_cell_lookup[c] for c in selected_cells])
    reference_cell_idx = np.asarray([reference_cell_lookup[c] for c in selected_cells])
    built_block = built.X[np.ix_(built_cell_idx, built_gene_idx)]
    reference_block = reference.X[np.ix_(reference_cell_idx, reference_gene_idx)]
    built_block = built_block.toarray() if sp.issparse(built_block) else np.asarray(built_block)
    reference_block = (reference_block.toarray()
                       if hasattr(reference_block, "toarray")
                       else np.asarray(reference_block))
    difference = np.abs(built_block.astype(np.float64) -
                        reference_block.astype(np.float64))
    correlation = np.corrcoef(built_block.ravel(), reference_block.ravel())[0, 1]
    return {
        "reference": str(reference_path),
        "common_cells": int(len(common_cells)),
        "common_genes": int(len(common_genes)),
        "validation_cells": int(len(cell_idx)),
        "validation_genes": int(len(gene_idx)),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "pearson": float(correlation),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.reuse:
        print(f"reusing {args.output}", flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    reference = ad.read_h5ad(args.reference)
    obs = reference.obs.copy()
    obs.index = reference.obs_names.astype(str)
    perturbations = obs["perturbation"].astype(str)

    barcodes = pd.read_csv(args.barcodes, sep="\t", header=None)[0].astype(str).values
    genes = pd.read_csv(args.genes, sep="\t", header=None)
    symbols = genes.iloc[:, 1].astype(str).values
    barcode_index = {barcode: i for i, barcode in enumerate(barcodes)}
    missing_barcodes = [barcode for barcode in obs.index if barcode not in barcode_index]
    if missing_barcodes:
        raise RuntimeError(f"{len(missing_barcodes)} reference barcodes absent from raw matrix")
    cell_index = np.asarray([barcode_index[barcode] for barcode in obs.index],
                            dtype=np.int64)

    print("reading full Norman Matrix Market", flush=True)
    matrix = mmread(args.raw).tocsr()
    if matrix.shape != (len(symbols), len(barcodes)):
        raise RuntimeError(
            f"matrix shape {matrix.shape} does not match genes x barcodes "
            f"{(len(symbols), len(barcodes))}"
        )
    raw_nnz = int(matrix.nnz)
    print(f"full matrix {matrix.shape}, nnz={raw_nnz:,}", flush=True)
    matrix = matrix[:, cell_index].tocsr()

    released_size_factors = obs["total_counts"].to_numpy(dtype=np.float64)
    if np.any(released_size_factors <= 0):
        raise RuntimeError("released Norman matrix contains non-positive size factors")
    control_columns = np.flatnonzero(perturbations.values == "control")
    if not len(control_columns):
        raise RuntimeError("reference matrix has no control cells")
    control_counts = np.asarray(matrix[:, control_columns].sum(axis=1),
                                dtype=np.float64).ravel()
    selected_genes, selection = select_genes(
        control_counts, symbols, perturbation_genes(perturbations), args.n_genes
    )
    selected_symbols = symbols[selected_genes]
    if len(selected_symbols) != args.n_genes:
        raise RuntimeError("selected output dimension does not match the request")

    # Use the released scPerturb cell-size factors so the 20k panel and the
    # primary 5,025-gene panel share exactly the same normalization scale.
    scale = (args.target_sum / released_size_factors).astype(np.float32)
    matrix = matrix[selected_genes, :].T.tocsr().astype(np.float32)
    matrix = matrix.multiply(scale[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
    built = ad.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(selected_symbols, name="gene_symbol")),
    )
    built.uns["output_panel"] = {
        "n_genes": args.n_genes,
        "n_unique_gene_symbols": int(len(set(selected_symbols))),
        "selection": ("all filtered-matrix genes" if args.n_genes == len(symbols)
                      else "top genes by raw counts in unperturbed cells"),
        "target_sum": args.target_sum,
        "normalization": (
            "released scPerturb per-cell total-count size factors, "
            "then log1p"
        ),
    }
    print("validating against released 5,025-gene matrix", flush=True)
    validation = validate_against_reference(built, args.reference)
    built.write_h5ad(args.output, compression="gzip", compression_opts=1)
    print(f"wrote {args.output}", flush=True)
    metadata = {
        "output": str(args.output),
        "shape": [int(built.n_obs), int(built.n_vars)],
        "cell_selection": "exact cell names from released scPerturb Norman matrix",
        "control_cells": int(len(control_columns)),
        "unique_gene_symbols": int(len(set(selected_symbols))),
        "gene_selection": selection,
        "validation": validation,
        "source_files": {
            "raw": str(args.raw),
            "genes": str(args.genes),
            "barcodes": str(args.barcodes),
            "reference": str(args.reference),
        },
        "raw_matrix_nnz": raw_nnz,
        "output_sha256": sha256(args.output),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
