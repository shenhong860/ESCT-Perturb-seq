#!/bin/bash
set -e

# Update these paths before running.
NORMAN_H5AD=/path/to/norman.h5ad
REPLOGLE_H5AD=/path/to/Replogle_exp6.h5ad
WESSELS_H5AD=/path/to/Wessels.h5ad
NORMAN_EMB=/path/to/norman_scgpt_emb.pkl

python scripts/run_esct_devgamma_clean.py norman outputs/norman.json
python scripts/run_esct_devgamma_clean.py replogle outputs/replogle.json
python scripts/run_esct_devgamma_clean.py wessels outputs/wessels.json

python scripts/run_esct_w1_devgamma_clean.py norman outputs/norman_w1.json
python scripts/eval_official_pd_devgamma.py norman outputs/norman.json outputs/norman_pd.json
