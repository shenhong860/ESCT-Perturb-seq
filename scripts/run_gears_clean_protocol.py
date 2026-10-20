"""Retrain official GEARS on the 12 clean ESCT Norman splits.

This script deliberately does not use GEARS' simulation split or the earlier
protocol-incompatible snapshot. It reconstructs the exact 80/10/10 gene split
used by ESCT, trains GEARS separately for every split, and evaluates only on
combinations present in the matching ESCT result file.

The data cache is built once from ``norman_hvg1000.h5ad``. GEARS' standard
20-epoch training recipe is used, with validation top-20 DE MSE selecting the
best epoch. Test labels are touched only after training.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import shutil
import time
from copy import deepcopy
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.stats as st
import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR

from gears import GEARS, PertData
from gears.inference import compute_metrics, evaluate
from gears.utils import loss_fct

from paths import data_dir, dataset_paths, env_path, experiments_dir


TOP_K = 100
N_SPLITS = 12
TRAIN_FRAC = 0.8
DEV_FRAC = 0.1

DEFAULT_H5AD = dataset_paths("norman")[0]
DEFAULT_EMB = dataset_paths("norman")[1]
DEFAULT_ESCT = env_path(
    "ESCT_NORMAN_PREDICTIONS", experiments_dir() / "esct_w1_devgamma_clean_norman.json"
)
DEFAULT_CACHE = env_path(
    "ESCT_GEARS_CACHE", experiments_dir() / "gears_clean_protocol/cache"
)
DEFAULT_GENE2GO = env_path(
    "ESCT_GEARS_GENE2GO", data_dir() / "gears_data/gene2go.pkl"
)
DEFAULT_GO_CONFIG = env_path(
    "ESCT_GEARS_GO_CONFIG", data_dir() / "gears_data/config.pkl"
)
DEFAULT_OLD_GEARS_H5AD = env_path(
    "ESCT_GEARS_PERTURB_H5AD", data_dir() / "gears_data/perturb_processed.h5ad"
)
DEFAULT_OUT = env_path(
    "ESCT_GEARS_OUTPUT", experiments_dir() / "gears_same_12_splits.json"
)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def genes_from_guide(guide: str) -> list[str]:
    parts = str(guide).replace("__", "_").split("_")
    return sorted({x for x in parts if not x.startswith("NegCtrl")})


def plain_condition(guide: str) -> str:
    genes = genes_from_guide(guide)
    return "+".join(genes) if genes else "ctrl"


def gears_condition(plain: str) -> str:
    genes = [x for x in plain.split("+") if x and x != "ctrl"]
    if not genes:
        return "ctrl"
    if len(genes) == 1:
        return f"ctrl+{genes[0]}"
    return "+".join(genes)


def split_terms(label: str) -> list[str]:
    return sorted({x for x in str(label).split("+") if x and x != "ctrl"})


def pcc(pred: np.ndarray, true: np.ndarray) -> float:
    de = np.argsort(-np.abs(true))[:TOP_K]
    if np.std(pred[de]) < 1e-12 or np.std(true[de]) < 1e-12:
        return 0.0
    return float(pearsonr(pred[de], true[de])[0])


def dir_acc(pred: np.ndarray, true: np.ndarray) -> float:
    de = np.argsort(-np.abs(true))[:TOP_K]
    return float(np.mean(np.sign(pred[de]) == np.sign(true[de])))


def metric_row(pred: np.ndarray, true: np.ndarray) -> list[float]:
    return [
        pcc(pred, true),
        dir_acc(pred, true),
        float(np.sqrt(np.sum((pred - true) ** 2))),
    ]


def load_esct_rows(path: Path) -> dict[int, dict[str, dict[str, list[float]]]]:
    """Recover split id for row-only ESCT output using calibration counts."""
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload["rows"]
    calibrations = payload.get("calibrations", {})
    seed_for_row = []
    for seed in range(1, N_SPLITS + 1):
        count = int(calibrations[str(seed)]["n_cat0"])
        seed_for_row.extend([seed] * count)
    if len(seed_for_row) != len(rows):
        raise RuntimeError(
            f"ESCT calibration counts cover {len(seed_for_row)} rows, "
            f"but result contains {len(rows)}"
        )

    out: dict[int, dict[str, dict[str, list[float]]]] = {}
    for seed, row in zip(seed_for_row, rows):
        out.setdefault(seed, {})[row["combo"]] = {
            "esct": [float(x) for x in row["esct"]],
            "add": [float(x) for x in row["add"]],
        }
    return out


def make_gears_adata(path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    guides = adata.obs["guide_identity"].astype(str).to_numpy()
    conditions = [plain_condition(x) for x in guides]
    term_count = np.array([len(split_terms(x)) for x in conditions])

    # The GSE guide strings occasionally produce three non-control tokens.
    # ESCT's two-gene universe excludes them, so GEARS should not see them as
    # a hidden perturbation class.
    keep = term_count <= 2
    adata = adata[keep].copy()
    conditions = [x for x, flag in zip(conditions, keep) if flag]

    adata.obs["condition"] = [gears_condition(x) for x in conditions]
    adata.obs["cell_type"] = "K562"
    adata.var["gene_name"] = [str(x) for x in adata.var_names]
    adata.obs["condition"] = adata.obs["condition"].astype("category")
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    return adata


def load_reusable_go_graph(
    config_path: Path,
    old_h5ad_path: Path,
    new_gene_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Map the released GEARS GO graph onto the retained hvg1000 genes.

    The dataverse tarball download is extremely slow from this host. The
    previously trained official GEARS run already contains the exact released
    GO edge tensor. Re-indexing those edges by stable gene names is equivalent
    to constructing the graph on the hvg1000 subspace and avoids a network
    dependency during training.
    """
    with config_path.open("rb") as handle:
        config = pickle.load(handle)
    old = ad.read_h5ad(old_h5ad_path, backed="r")
    old_genes = [str(x) for x in old.var["gene_name"]]
    new_index = {gene: i for i, gene in enumerate(new_gene_names)}

    edge_index = config["G_go"]
    edge_weight = config["G_go_weight"]
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu().numpy()
    else:
        edge_index = np.asarray(edge_index)
    if isinstance(edge_weight, torch.Tensor):
        edge_weight = edge_weight.detach().cpu().numpy()
    else:
        edge_weight = np.asarray(edge_weight)

    src_old = edge_index[0].astype(np.int64)
    dst_old = edge_index[1].astype(np.int64)
    if src_old.size and max(int(src_old.max()), int(dst_old.max())) >= len(old_genes):
        raise RuntimeError("GO edge index exceeds the old GEARS gene list")
    src_name = [old_genes[i] for i in src_old]
    dst_name = [old_genes[i] for i in dst_old]
    keep = np.array(
        [src in new_index and dst in new_index for src, dst in zip(src_name, dst_name)],
        dtype=bool,
    )
    mapped_src = np.array(
        [new_index[x] for x, keep_edge in zip(src_name, keep) if keep_edge],
        dtype=np.int64,
    )
    mapped_dst = np.array(
        [new_index[x] for x, keep_edge in zip(dst_name, keep) if keep_edge],
        dtype=np.int64,
    )
    mapped_index = torch.as_tensor(
        np.stack([mapped_src, mapped_dst]), dtype=torch.long
    )
    mapped_weight = torch.as_tensor(edge_weight[keep], dtype=torch.float32)
    meta = {
        "source_config": str(config_path),
        "source_h5ad": str(old_h5ad_path),
        "old_genes": len(old_genes),
        "new_genes": len(new_gene_names),
        "edges_before_hvg1000_mapping": int(edge_weight.shape[0]),
        "edges_after_hvg1000_mapping": int(mapped_weight.numel()),
    }
    return mapped_index, mapped_weight, meta


def prepare_or_load_gears(
    h5ad_path: Path,
    cache_root: Path,
    gene2go_path: Path,
    rebuild: bool,
    preprocess_seed: int,
) -> tuple[PertData, dict]:
    cache_root.mkdir(parents=True, exist_ok=True)
    gene2go_all = cache_root / "gene2go_all.pkl"
    if not gene2go_all.exists():
        shutil.copyfile(gene2go_path, gene2go_all)

    dataset_name = "norman_esct_hvg1000"
    processed_dir = cache_root / dataset_name
    processed_h5ad = processed_dir / "perturb_processed.h5ad"
    graph_file = processed_dir / "data_pyg" / "cell_graphs.pkl"
    meta_file = cache_root / "preprocess_meta.json"

    np.random.seed(preprocess_seed)
    torch.manual_seed(preprocess_seed)
    pd = PertData(str(cache_root))

    if rebuild and processed_dir.exists():
        shutil.rmtree(processed_dir)

    if processed_h5ad.exists() and graph_file.exists():
        pd.load(data_name=None, data_path=str(processed_dir))
        status = "loaded"
    else:
        raw = make_gears_adata(h5ad_path)
        print(
            f"preprocess GEARS AnnData: {raw.shape[0]} cells x "
            f"{raw.shape[1]} genes, "
            f"{raw.obs['condition'].nunique()} conditions",
            flush=True,
        )
        pd.new_data_process(dataset_name, adata=raw, skip_calc_de=False)
        status = "built"

    pd.set_pert_genes()
    pd.ctrl_adata = pd.adata[pd.adata.obs["condition"] == "ctrl"]
    pd.gene_names = pd.adata.var.gene_name

    meta = {
        "status": status,
        "dataset_name": dataset_name,
        "processed_h5ad": str(processed_h5ad),
        "n_cells": int(pd.adata.n_obs),
        "n_genes": int(pd.adata.n_vars),
        "n_conditions": int(pd.adata.obs["condition"].nunique()),
        "n_control_cells": int(len(pd.ctrl_adata)),
        "n_perturbations_in_go": int(len(pd.pert_names)),
        "preprocess_seed": preprocess_seed,
    }
    meta_file.write_text(json.dumps(meta, indent=2, default=json_default))
    return pd, meta


def build_split(
    plain_conditions: list[str],
    pd: PertData,
    emb_genes: set[str],
    split_seed: int,
) -> dict:
    labels = sorted(set(plain_conditions) - {"ctrl"})
    singles = [x for x in labels if len(split_terms(x)) == 1]
    combos = [
        x
        for x in labels
        if len(split_terms(x)) == 2 and all(g in emb_genes for g in split_terms(x))
    ]
    all_genes = sorted(
        {
            g
            for condition in singles + combos
            for g in split_terms(condition)
            if g in emb_genes
        }
    )

    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(all_genes).tolist()
    n_train = int(len(all_genes) * TRAIN_FRAC)
    n_dev = int(len(all_genes) * DEV_FRAC)
    train_set = set(perm[:n_train])
    dev_set = set(perm[n_train : n_train + n_dev])
    test_set = set(perm[n_train + n_dev :])

    available = set(pd.adata.obs["condition"].astype(str).unique())
    perturb_genes_in_go = set(str(x) for x in pd.pert_names)

    def available_gear(condition: str) -> bool:
        return (
            gears_condition(condition) in available
            and all(g in perturb_genes_in_go for g in split_terms(condition))
        )

    train_plain = [
        c
        for c in labels
        if all(g in train_set for g in split_terms(c)) and available_gear(c)
    ]
    dev_plain = [
        c
        for c in labels
        if any(g in dev_set for g in split_terms(c))
        and not any(g in test_set for g in split_terms(c))
        and available_gear(c)
    ]
    test_plain = [
        c
        for c in combos
        if any(g in test_set for g in split_terms(c))
        and all(g not in train_set for g in split_terms(c))
    ]
    test_gear = [gears_condition(c) for c in test_plain]
    covered_test = [
        c for c, gear in zip(test_plain, test_gear) if available_gear(c)
    ]
    missing_test = [
        c for c, gear in zip(test_plain, test_gear) if not available_gear(c)
    ]

    set2conditions = {
        "train": ["ctrl"] + [gears_condition(c) for c in train_plain],
        "val": [gears_condition(c) for c in dev_plain],
        "test": [gears_condition(c) for c in covered_test],
    }
    return {
        "seed": split_seed,
        "all_genes": all_genes,
        "train_genes": sorted(train_set),
        "dev_genes": sorted(dev_set),
        "test_genes": sorted(test_set),
        "train_plain": train_plain,
        "dev_plain": dev_plain,
        "test_plain": test_plain,
        "covered_test_plain": covered_test,
        "missing_test_plain": missing_test,
        "set2conditions": set2conditions,
    }


def train_one_split(
    pd: PertData,
    split: dict,
    split_dir: Path,
    epochs: int,
    batch_size: int,
    train_seed: int,
    resume: bool,
    go_edge_index: torch.Tensor,
    go_edge_weight: torch.Tensor,
) -> tuple[GEARS, dict]:
    split_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = split_dir / "checkpoint.pt"
    done_file = split_dir / "done.json"

    # Avoid GEARS.prepare_split's broken custom branch while preserving the
    # official dataloader, graph construction and training paths.
    pd.split = "esct80"
    pd.seed = int(split["seed"])
    pd.subgroup = None
    pd.train_gene_set_size = TRAIN_FRAC
    pd.set2conditions = split["set2conditions"]
    pd.get_dataloader(batch_size=batch_size, test_batch_size=128)

    # Keep the test loader out of training. Predictions are evaluated only
    # after the best validation epoch has been frozen.
    pd.dataloader = {
        "train_loader": pd.dataloader["train_loader"],
        "val_loader": pd.dataloader["val_loader"],
    }

    torch.manual_seed(train_seed + int(split["seed"]))
    np.random.seed(train_seed + int(split["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gm = GEARS(pd, device=device)
    gm.model_initialize(
        hidden_size=64,
        num_go_gnn_layers=1,
        num_gene_gnn_layers=1,
        decoder_hidden_size=16,
        num_similar_genes_go_graph=20,
        num_similar_genes_co_express_graph=20,
        coexpress_threshold=0.4,
        G_go=go_edge_index,
        G_go_weight=go_edge_weight,
    )

    model = gm.model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    best_state = deepcopy(model.state_dict())
    best_val_mse_de = float("inf")
    start_epoch = 0
    history: list[dict] = []
    elapsed_before_resume = 0.0

    if resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_state = state["best_state"]
        best_val_mse_de = float(state["best_val_mse_de"])
        start_epoch = int(state["epoch"])
        history = list(state.get("history", []))
        elapsed_before_resume = float(state.get("elapsed", 0.0))
        torch.set_rng_state(state["torch_rng"].cpu())
        if device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        print(f"resume split {split['seed']} at epoch {start_epoch}", flush=True)

    train_loader = pd.dataloader["train_loader"]
    val_loader = pd.dataloader["val_loader"]
    started = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_started = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = loss_fct(
                pred,
                batch.y,
                batch.pert,
                ctrl=gm.ctrl_expression,
                dict_filter=gm.dict_filter,
                direction_lambda=gm.config["direction_lambda"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        scheduler.step()
        val_res = evaluate(val_loader, model, False, device)
        val_metrics, _ = compute_metrics(val_res)
        val_mse_de = float(val_metrics["mse_de"])
        improved = val_mse_de < best_val_mse_de
        if improved:
            best_val_mse_de = val_mse_de
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_mse": float(val_metrics["mse"]),
            "val_mse_de": val_mse_de,
            "best": improved,
            "seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(
            f"split {split['seed']} epoch {epoch + 1}/{epochs}: "
            f"loss={record['train_loss']:.4f} "
            f"val_de={val_mse_de:.4f} best={improved} "
            f"{record['seconds']:.0f}s",
            flush=True,
        )

        elapsed = elapsed_before_resume + time.time() - started
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_state": best_state,
                "best_val_mse_de": best_val_mse_de,
                "history": history,
                "elapsed": elapsed,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
            },
            checkpoint,
        )

    gm.best_model = deepcopy(model)
    gm.best_model.load_state_dict(best_state)
    gm.best_model = gm.best_model.to(device)
    gm.best_model.eval()
    torch.save(best_state, split_dir / "best_model.pt")

    summary = {
        "seed": int(split["seed"]),
        "epochs": epochs,
        "best_val_mse_de": best_val_mse_de,
        "history": history,
        "n_train_conditions": len(split["set2conditions"]["train"]),
        "n_dev_conditions": len(split["set2conditions"]["val"]),
        "n_test_conditions": len(split["set2conditions"]["test"]),
    }
    done_file.write_text(json.dumps(summary, indent=2, default=json_default))
    return gm, summary


def paired_summary(rows: list[dict]) -> dict:
    metrics = ["pcc", "dir", "mean_l2"]
    methods = ["gears", "esct", "add"]
    out = {
        "n": len(rows),
        "methods": {
            method: {
                metric: float(np.mean([r[method][metric] for r in rows]))
                for metric in metrics
            }
            for method in methods
        },
        "paired": {},
    }
    for left, right in [("esct", "gears"), ("esct", "add"), ("gears", "add")]:
        key = f"{left}_vs_{right}"
        out["paired"][key] = {}
        for metric in metrics:
            a = np.array([r[left][metric] for r in rows], dtype=float)
            b = np.array([r[right][metric] for r in rows], dtype=float)
            diff = a - b
            nonzero = np.abs(diff) > 1e-12
            if nonzero.sum() >= 2:
                p_value = float(st.wilcoxon(diff, zero_method="wilcox").pvalue)
            else:
                p_value = 1.0
            out["paired"][key][metric] = {
                "mean_diff": float(diff.mean()),
                "p_wilcoxon": p_value,
                "left_wins": int((diff > 0).sum()),
                "right_wins": int((diff < 0).sum()),
                "ties": int((np.abs(diff) <= 1e-12).sum()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--emb", type=Path, default=DEFAULT_EMB)
    parser.add_argument("--esct-json", type=Path, default=DEFAULT_ESCT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--gene2go", type=Path, default=DEFAULT_GENE2GO)
    parser.add_argument("--go-config", type=Path, default=DEFAULT_GO_CONFIG)
    parser.add_argument("--old-gears-h5ad", type=Path, default=DEFAULT_OLD_GEARS_H5AD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--start-split", type=int, default=1)
    parser.add_argument("--run-prefix", default="runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=24)
    parser.add_argument("--preprocess-seed", type=int, default=24)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.splits = min(args.splits, 1)
        args.epochs = min(args.epochs, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    esct_rows = load_esct_rows(args.esct_json)
    pd, preprocess_meta = prepare_or_load_gears(
        args.h5ad, args.cache, args.gene2go, args.rebuild, args.preprocess_seed
    )
    go_edge_index, go_edge_weight, go_meta = load_reusable_go_graph(
        args.go_config,
        args.old_gears_h5ad,
        [str(x) for x in pd.adata.var["gene_name"]],
    )
    preprocess_meta["go_graph"] = go_meta

    raw = ad.read_h5ad(args.h5ad, backed="r")
    plain_conditions = [plain_condition(x) for x in raw.obs["guide_identity"].astype(str)]
    with args.emb.open("rb") as handle:
        emb_payload = pickle.load(handle)
    emb_genes = set(emb_payload["genes"])
    control_mean = np.asarray(raw.X[np.array(plain_conditions) == "ctrl"].mean(0)).ravel()
    condition_index = {
        condition: np.where(np.array(plain_conditions) == condition)[0]
        for condition in sorted(set(plain_conditions) - {"ctrl"})
    }

    all_rows: list[dict] = []
    split_summaries: list[dict] = []
    started = time.time()

    for split_seed in range(args.start_split, args.splits + 1):
        split = build_split(plain_conditions, pd, emb_genes, split_seed)
        split_dir = args.out.parent / args.run_prefix / f"seed{split_seed}"
        print(
            f"split {split_seed}: genes={len(split['all_genes'])} "
            f"train={len(split['set2conditions']['train'])} "
            f"dev={len(split['set2conditions']['val'])} "
            f"test_esct={len(split['test_plain'])} "
            f"test_gears_covered={len(split['covered_test_plain'])} "
            f"test_gears_missing={len(split['missing_test_plain'])}",
            flush=True,
        )
        if not split["set2conditions"]["train"] or not split["set2conditions"]["val"]:
            raise RuntimeError(f"split {split_seed} has empty training/validation set")

        gm, train_summary = train_one_split(
            pd,
            split,
            split_dir,
            args.epochs,
            args.batch_size,
            args.train_seed,
            args.resume,
            go_edge_index,
            go_edge_weight,
        )

        if split["covered_test_plain"]:
            pert_lists = [c.split("+") for c in split["covered_test_plain"]]
            predictions = gm.predict(pert_lists)
        else:
            predictions = {}

        split_rows = []
        for combo in split["covered_test_plain"]:
            key = "_".join(combo.split("+"))
            pred_delta = np.asarray(predictions[key], dtype=float) - control_mean
            true_idx = condition_index[combo]
            true_delta = np.asarray(raw.X[true_idx].mean(0)).ravel() - control_mean
            gear_metrics = metric_row(pred_delta, true_delta)
            row = {
                "seed": split_seed,
                "combo": combo,
                "gears": {
                    "pcc": gear_metrics[0],
                    "dir": gear_metrics[1],
                    "mean_l2": gear_metrics[2],
                },
            }
            if combo in esct_rows.get(split_seed, {}):
                reference = esct_rows[split_seed][combo]
                row["esct"] = {
                    "pcc": reference["esct"][0],
                    "dir": reference["esct"][1],
                    "mean_l2": reference["esct"][2],
                }
                row["add"] = {
                    "pcc": reference["add"][0],
                    "dir": reference["add"][1],
                    "mean_l2": reference["add"][2],
                }
                all_rows.append(row)
                split_rows.append(row)

        split_summary = {
            **train_summary,
            "train_genes": split["train_genes"],
            "dev_genes": split["dev_genes"],
            "test_genes": split["test_genes"],
            "test_esct_combinations": split["test_plain"],
            "test_gears_covered": split["covered_test_plain"],
            "test_gears_missing": split["missing_test_plain"],
            "paired_rows": len(split_rows),
        }
        if split_rows:
            split_summary["metrics"] = paired_summary(split_rows)["methods"]
        split_summaries.append(split_summary)

        partial = {
            "protocol": {
                "dataset": "Norman",
                "n_splits": args.splits,
                "start_split": args.start_split,
                "train_frac": TRAIN_FRAC,
                "dev_frac": DEV_FRAC,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "train_seed": args.train_seed,
                "preprocess_seed": args.preprocess_seed,
                "smoke": bool(args.smoke),
            },
            "preprocess": preprocess_meta,
            "splits": split_summaries,
            "n_rows": len(all_rows),
            "rows": all_rows,
            "aggregate": paired_summary(all_rows) if all_rows else {"n": 0},
            "elapsed_seconds": time.time() - started,
        }
        args.out.write_text(json.dumps(partial, indent=2, default=json_default))

        del gm, predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(paired_summary(all_rows), indent=2), flush=True)
    print(f"written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
