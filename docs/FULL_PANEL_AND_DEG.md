# Full Norman panel and official DEG metrics

## Scope

The complete released Norman filtered matrix contains 96,994 cells and 33,694
measured feature rows. It contains 33,660 unique gene symbols because 34 symbols
occur more than once. All rows are retained as model outputs. Duplicate symbols
are collapsed only when ranking official DEGs, using the row with the largest
absolute predicted response.

This experiment is a full-filtered-panel output stress test. It is not a
whole-genome DEG-recovery experiment, and the manuscript does not make that
claim.

## Build the panel

Download the GSE133344 filtered matrix triplet described in `data/README.md`,
then run:

```bash
export ESCT_DATA_DIR=/path/to/data
export ESCT_OUT=$PWD/outputs
export ESCT_EXPERIMENTS_DIR=$ESCT_OUT

python scripts/build_norman_full_panel.py \
  --n-genes 33694 \
  --output "$ESCT_OUT/norman_full_panel.h5ad" \
  --metadata "$ESCT_OUT/norman_full_panel_build.json"
```

The builder uses only control-cell counts for feature selection when a subset is
requested. With `--n-genes 33694`, every row is retained. Normalization reuses
the released scPerturb per-cell size factors before `log1p`.

## Run the model and metrics

```bash
export ESCT_NORMAN_FULL_H5AD=$ESCT_OUT/norman_full_panel.h5ad
export ESCT_NORMAN_H5AD=$ESCT_NORMAN_FULL_H5AD

python scripts/run_esct_w1_devgamma_clean.py norman "$ESCT_OUT/norman_33694.json"
python scripts/eval_official_deg_metrics.py \
  --dataset Norman_33694 \
  --output-dir "$ESCT_OUT/official_deg_metrics_33694"
```

Official recall uses the released `DEG_hvg5000_Norman.pkl` reference table.
Coverage is reported separately so a restricted expression panel cannot be
mistaken for complete DEG recovery. DEG-MMD is computed at condition level over
the union of available official DEGs within each split.

## Archived result

The 12-split, two-model-seed run used 500 iterations per split.

| Metric | Additive | ESCT |
|---|---:|---:|
| PCC | 0.704 | 0.801 |
| Direction accuracy | 0.887 | 0.926 |
| Mean L2 | 6.707 | 5.818 |
| Official DEG recall@100 | 0.254 | 0.299 |
| Official DEG recall@200 | 0.264 | 0.306 |
| Official DEG-MMD@100 | 0.287 | 0.204 |
| Official DEG-MMD@200 | 0.310 | 0.221 |

Paired Wilcoxon p-values were `3.49e-7`, `2.52e-5`, and `1.91e-8` for PCC,
direction accuracy, and mean L2, respectively. Runtime was 1,023.2 seconds with
1,317.3 MB peak allocated GPU memory.

## Reproducibility checks

- The full matrix has shape `96,994 x 33,694`.
- The intersection with the released 5,025-gene matrix has Pearson correlation
  `1.0` and maximum absolute reconstruction error `4.77e-7`.
- Only six of the 13,694 features added beyond the 20,000-feature check had
  positive counts in control cells.
- Large prediction files and h5ad matrices are intentionally excluded from the
  repository.
