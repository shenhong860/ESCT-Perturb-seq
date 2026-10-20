# Reproducibility

## Protocol

- 12 independent 80/10/10 gene splits per dataset
- broad no-train-gene: neither component gene is in the training single-gene bank, with at most one in development
- strict test+test: both component genes are absent from training and development
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
python scripts/build_norman_full_panel.py --n-genes 33694
python scripts/eval_official_deg_metrics.py --dataset Norman_33694
```

## Baselines

```bash
python scripts/run_linearModel_clean_protocol.py
python scripts/run_scgpt_mlp_clean_protocol.py
python scripts/run_scgpt_transformer_finetune_clean.py
python scripts/run_capra_clean_protocol.py 12 80
```

## Expected key results

Broad no-train-gene, additive -> ESCT:

- Norman (n=50): PCC 0.531 -> 0.667, direction 0.754 -> 0.805,
  mean L2 2.990 -> 2.487, MMD 0.0619 -> 0.0503
- Norman full filtered panel (33,694 measured features; n=50):
  PCC 0.704 -> 0.801, direction 0.887 -> 0.926,
  mean L2 6.707 -> 5.818
- CAPRA 80-epoch on Norman: PCC 0.601, direction 0.775, mean L2 3.390

Full tables and p-values are in the manuscript.

## Attribution

The evaluation protocol, pretrained embeddings, and baseline code used here
come from external sources; see `docs/ATTRIBUTION.md` before redistribution.
