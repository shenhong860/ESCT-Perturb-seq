#!/bin/bash
set -e

# Update these paths before running (see scripts/paths.py).
export ESCT_DATA_DIR=/path/to/h5ad
export ESCT_EMB_DIR=/path/to/scgpt_embeddings
export ESCT_DEG_DIR=/path/to/deg
export ESCT_EXPERIMENTS_DIR=/path/to/outputs
# Optional per-dataset overrides:
# export ESCT_NORMAN_H5AD=/path/to/norman/norman_hvg1000.h5ad
# export ESCT_REPLOGLE_H5AD=/path/to/Replogle_exp6.h5ad
# export ESCT_WESSELS_H5AD=/path/to/Wessels.h5ad

python scripts/run_esct_devgamma_clean.py norman outputs/norman.json
python scripts/run_esct_devgamma_clean.py replogle outputs/replogle.json
python scripts/run_esct_devgamma_clean.py wessels outputs/wessels.json

python scripts/run_esct_w1_devgamma_clean.py norman outputs/norman_w1.json
python scripts/eval_official_pd_devgamma.py norman outputs/norman.json outputs/norman_pd.json
