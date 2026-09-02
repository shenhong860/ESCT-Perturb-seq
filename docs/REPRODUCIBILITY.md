# Reproducibility

## Protocol

- 12 independent 80/10/10 gene splits per dataset
- combo_seen0: both genes absent from training split, at least one in test split
- gate threshold recalibrated per split from development genes
- lambda selected by 3-fold CV on development combinations
- test expression and labels never used for fitting or selection

## Core commands

Set the data/output directories first (see `scripts/paths.py` for the full
list of environment variables):

```bash
export ESCT_DATA_DIR=/path/to/h5ad
export ESCT_EMB_DIR=/path/to/scgpt_embeddings
export ESCT_DEG_DIR=/path/to/deg
export ESCT_EXPERIMENTS_DIR=/path/to/outputs
```

```bash
python scripts/run_esct_devgamma_clean.py norman outputs/norman.json
python scripts/run_esct_devgamma_clean.py replogle outputs/replogle.json
python scripts/run_esct_devgamma_clean.py wessels outputs/wessels.json

python scripts/run_esct_w1_devgamma_clean.py norman outputs/norman_w1.json
python scripts/eval_official_pd_devgamma.py norman outputs/norman.json outputs/norman_pd.json
```

## Baselines

```bash
python scripts/run_linearModel_clean_protocol.py
python scripts/run_scgpt_mlp_clean_protocol.py
python scripts/run_scgpt_transformer_finetune_clean.py
python scripts/run_capra_clean_protocol.py 12 80
```

## Expected key results

Norman cat0 n=50: PCC 0.667, dir 0.805, mean L2 2.487, MMD 0.0503.
CAPRA 80-epoch: PCC 0.601, dir 0.775, mean L2 3.390.

Full tables and p-values are in the manuscript.

## Attribution

The evaluation protocol, pretrained embeddings, and baseline code used here
come from external sources; see `docs/ATTRIBUTION.md` before redistribution.
