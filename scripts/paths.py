"""Repository-relative path resolution shared by all ESCT scripts.

The public scripts avoid machine-specific absolute paths.  Set the
environment variables below to point at your local data and output
locations; the defaults are relative to this repository.

- ESCT_DATA_DIR: directory containing processed h5ad matrices
- ESCT_EMB_DIR: directory containing scGPT gene embedding pickles
- ESCT_DEG_DIR: directory containing DEG_hvg5000_*.pkl files
- ESCT_EXPERIMENTS_DIR: directory for run outputs (default outputs/)
- ESCT_SCGPT_WEIGHTS: scGPT checkpoint directory
- ESCT_CAPRA_ROOT: CAPRA source directory (baseline only)
- ESCT_CAPRA_DATA: CAPRA demo data directory (GenePT embeddings)

Per-dataset overrides (ESCT_NORMAN_H5AD, ESCT_REPLOGLE_H5AD,
ESCT_WESSELS_H5AD and their *_EMB counterparts) take precedence over the
directory defaults.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def env_path(name, default):
    """Return Path from environment variable ``name`` or a default Path."""
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


def data_dir():
    return env_path("ESCT_DATA_DIR", REPO_ROOT / "data")


def emb_dir():
    return env_path("ESCT_EMB_DIR", REPO_ROOT / "data" / "embeddings")


def deg_dir():
    return env_path("ESCT_DEG_DIR", REPO_ROOT / "data" / "deg")


def experiments_dir():
    return env_path("ESCT_EXPERIMENTS_DIR", REPO_ROOT / "outputs")


def scgpt_weights_dir():
    return env_path("ESCT_SCGPT_WEIGHTS", REPO_ROOT / "data" / "scgpt_weights")


def capra_root():
    return env_path("ESCT_CAPRA_ROOT", REPO_ROOT / "external" / "CAPRA" / "capra")


def capra_data_dir():
    return env_path("ESCT_CAPRA_DATA", REPO_ROOT / "data" / "capra")


_DATASET_FILES = {
    "norman": {
        "h5ad": "norman/norman_hvg1000.h5ad",
        "emb": "norman_scgpt_emb.pkl",
    },
    "replogle": {
        "h5ad": "Replogle_exp6/Replogle_exp6.h5ad",
        "emb": "replogle_exp6_scgpt_emb.pkl",
    },
    "wessels": {
        "h5ad": "Wessels/Wessels.h5ad",
        "emb": "wessels_scgpt_emb.pkl",
    },
}


def dataset_paths(name):
    """Return (h5ad, embedding pickle) paths for one dataset."""
    key = name.lower()
    if key not in _DATASET_FILES:
        raise ValueError(f"unknown dataset: {name}")
    cfg = _DATASET_FILES[key]
    tag = key.upper()
    h5ad = env_path(f"ESCT_{tag}_H5AD", data_dir() / cfg["h5ad"])
    emb = env_path(f"ESCT_{tag}_EMB", emb_dir() / cfg["emb"])
    return h5ad, emb
