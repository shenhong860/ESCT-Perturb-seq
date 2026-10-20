# ESCT-Perturb-seq

Embedding-Supervised Compositional Transport (ESCT) for unseen-gene
two-gene perturbation response prediction.

ESCT decomposes combination responses into a frozen additive backbone and a
regularized embedding-conditioned mean correction, then generates cell-level
distributions with a direction-regularized one-step transport head carrying a
W1-inspired Lipschitz-gradient penalty. For genes without direct single-gene
observations, a gated scGPT retrieval anchor replaces the global-mean fallback
when embedding similarity is sufficiently high.

## Highlights

- Strict gene-split protocol: 12 x 80/10/10 splits over perturbation genes
- Development-only gate calibration and hyperparameter selection
- Official scPerturBench pearson_distance evaluation
- Same-split baselines: additive, GEARS, linearModel, CAPRA, scGPT MLP, scGPT full transformer
- Joint end-to-end mean-and-transport baseline under the same 12 splits
- Component ablations and unfiltered Pearson Delta
- Full-pipeline embedding sensitivity: scGPT, Geneformer, GenePT (Norman), co-expression PCA, random gene ID
- Cell-level top-response-gene MMD, model-induced one-step maps, and corrected-mean-shift controls
- Learned vector-composition ablation and a complete 33,694-feature Norman output stress test
- Strict test+test audit reported separately from the broad no-train-gene protocol

## Results (broad no-train-gene protocol, additive -> ESCT)

| Dataset | n | PCC | dir | mean L2 | MMD |
|---|---:|---:|---:|---:|---:|
| Norman | 50 | 0.531 -> 0.667 | 0.754 -> 0.805 | 2.990 -> 2.487 | 0.0619 -> 0.0503 |
| Replogle_exp6 | 12 | 0.513 -> 0.568 | 0.769 -> 0.810 | 3.489 -> 2.884 | 0.0130 -> 0.0120 |
| Wessels | 59 | 0.730 -> 0.809 | 0.884 -> 0.928 | 5.535 -> 4.560 | 0.0175 -> 0.0163 |

The complete Norman filtered panel contains 33,694 measured features (33,660
unique symbols). Under the same 12 splits, the additive -> ESCT comparison was
PCC 0.704 -> 0.801, direction accuracy 0.887 -> 0.926, and mean L2
6.707 -> 5.818. This is a full-filtered-panel scaling check, not a
whole-genome DEG-recovery claim.

## Installation

```bash
conda create -n esct python=3.10 -y
conda activate esct
pip install -r requirements.txt
```

Verify the environment first:

```bash
python scripts/import_check.py
```

## Data

Download processed Perturb-seq datasets from scPerturb Zenodo record 14638780.
See `data/README.md` for expected files.

All scripts resolve data paths through `scripts/paths.py`. Set these
environment variables before running (per-dataset `ESCT_*_H5AD`/`ESCT_*_EMB`
overrides are also supported):

```bash
export ESCT_DATA_DIR=/path/to/h5ad
export ESCT_EMB_DIR=/path/to/scgpt_embeddings
export ESCT_DEG_DIR=/path/to/deg
export ESCT_EXPERIMENTS_DIR=/path/to/outputs
```

For the CAPRA baseline, additionally set:

```bash
export ESCT_CAPRA_ROOT=/path/to/CAPRA/capra
export ESCT_CAPRA_DATA=/path/to/CAPRA/demo/data
```

## Quickstart

```bash
python scripts/run_esct_devgamma_clean.py norman results/norman.json
python scripts/run_esct_w1_devgamma_clean.py norman results/norman_w1.json
python scripts/run_end_to_end_clean_protocol.py norman results/norman_e2e.json
python scripts/run_gears_clean_protocol.py --splits 12 --epochs 20 --batch-size 64 --resume
python scripts/eval_official_pd_devgamma.py norman results/norman.json results/norman_pd.json
python scripts/build_norman_full_panel.py --n-genes 33694
python scripts/eval_official_deg_metrics.py --dataset Norman_33694
```

See `examples/run_all.sh` for the full three-dataset pipeline and baseline
wrappers. See `docs/FULL_PANEL_AND_DEG.md` for the complete 33,694-feature
stress test, duplicate-symbol handling, and official DEG metric definitions.

## Repository structure

```text
scripts/       model, training, baseline, evaluation, and config scripts
docs/          attribution, reproducibility, and full-panel result notes
figures/       figure generation notes
examples/      example commands
data/          data download instructions
outputs/       ignored output directory
```

## Attribution

This repository uses scPerturb data, scGPT embeddings, the scPerturBench
protocol, and the CAPRA/linearModel baselines. See
[docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) for the exact sources and
modifications. These sources must remain cited when this repository or
derivatives are redistributed.

## Citation

Please cite this repository when using ESCT:

```text
Fan, X., Wang, J., Ma, Y., et al. ESCT-Perturb-seq: Embedding-Supervised
Compositional Transport for unseen-gene combination perturbation prediction.
```

## License

MIT
