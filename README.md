# ESCT-Perturb-seq

Epistasis-Weighted Compositional Transport (ESCT) for zero-shot
two-gene perturbation response prediction.

ESCT decomposes combination responses into a frozen additive backbone and a
regularized interaction residual, then generates cell-level distributions
with a W1 direction-aware one-step transport. For genes without direct
single-gene observations, a gated scGPT retrieval anchor replaces the
global-mean fallback when embedding similarity is sufficiently high.

## Highlights

- Strict gene-split protocol: 12 x 80/10/10 splits over perturbation genes
- Development-only gate calibration and hyperparameter selection
- Official scPerturBench pearson_distance evaluation
- Same-split baselines: additive, linearModel, CAPRA, scGPT MLP, scGPT full transformer
- Component ablations and unfiltered Pearson Delta

## Results (combo_seen0, additive -> ESCT+W1)

| Dataset | n | PCC | dir | mean L2 | MMD |
|---|---:|---:|---:|---:|---:|
| Norman | 50 | 0.531 -> 0.667 | 0.754 -> 0.805 | 2.990 -> 2.487 | 0.0619 -> 0.0503 |
| Replogle_exp6 | 12 | 0.513 -> 0.568 | 0.769 -> 0.810 | 3.489 -> 2.884 | 0.0130 -> 0.0120 |
| Wessels | 59 | 0.730 -> 0.809 | 0.884 -> 0.928 | 5.535 -> 4.560 | 0.0175 -> 0.0163 |

## Installation

```bash
conda create -n esct python=3.10 -y
conda activate esct
pip install -r requirements.txt
```

## Data

Download processed Perturb-seq datasets from scPerturb Zenodo
(DOI 14638780). See `data/README.md` for expected files.

## Quickstart

Paths in the scripts are local examples; update the constants at the top of
each script before running.

```bash
python scripts/run_esct_devgamma_clean.py norman results/norman.json
python scripts/run_esct_w1_devgamma_clean.py norman results/norman_w1.json
python scripts/eval_official_pd_devgamma.py norman results/norman.json results/norman_pd.json
```

## Repository structure

```text
scripts/       main model, training, baseline, and evaluation scripts
docs/          reproducibility notes
examples/      example commands
data/          data download instructions
outputs/       ignored output directory
```

## Citation

Please cite this repository when using ESCT:

```text
Guo, L., Wang, J., Ma, Y., et al. ESCT-Perturb-seq: Epistasis-Weighted
Compositional Transport for unseen-gene combination perturbation prediction.
```

## License

MIT
