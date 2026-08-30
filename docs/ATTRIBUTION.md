# Attribution

ESCT-Perturb-seq builds on public benchmark datasets, pretrained models, and
baseline code.  The corresponding original sources must remain cited when this
repository or its derivatives are redistributed.

## Datasets and evaluation protocol

- Processed single-cell perturbation matrices are provided by scPerturb:
  Peidli, S. et al., *scPerturb: harmonized single-cell perturbation data*,
  Nature Methods 21, 531-540 (2024), doi:10.1038/s41592-023-02144-y.
  Original studies: Norman et al., Science 365, 786-793 (2019);
  Replogle et al., Cell 185, 2559-2575 (2022); Wessels et al.,
  Nature Methods 20, 86-94 (2023).
- The perturbation-generalization protocol and official evaluation metric are
  taken from scPerturBench:
  `https://github.com/bm2-lab/scPerturBench` and Wei, Z. et al.,
  *Benchmarking algorithms for generalizable single-cell perturbation response
  prediction*, Nature Methods (2026), doi:10.1038/s41592-025-02980-0.
  Local modifications in this release: repository-relative path handling,
  development-only gate calibration, and the 12x80/10/10 gene-split runner.
- DEG reference files (`DEG_hvg5000_*.pkl`) are from the scPerturBench
  perturbation-generalization dataset record (Zenodo DOI 14638780).

## Pretrained embeddings and models

- scGPT gene embeddings and the full transformer fine-tuning baseline use the
  scGPT release: Cui, H. et al., Nature Methods 21, 1470-1480 (2024),
  doi:10.1038/s41592-024-02201-0; repository:
  `https://github.com/bowang-lab/scGPT`.
- The CAPRA baseline uses the released CAPRA source and GenePT embedding
  table: `https://github.com/zc-fang/CAPRA` and Wei, D.-Q. et al., CAPRA
  preprint (Research Square, 2026), doi:10.21203/rs.3.rs-10231666/v1.
- The linearModel baseline follows the scPerturBench linearModel.r wrapper:
  Ahlmann-Eltze, C. et al., bioRxiv (2024),
  doi:10.1101/2024.09.16.613342.

The `scripts/stubs/torchtext/` directory is a minimal import compatibility
shim for environments where the scGPT package's `torchtext` import is not
available; it is not part of the scGPT release and is not a modification of
the scGPT source.

## Local modifications

- Path resolution moved to environment variables via `scripts/paths.py`.
- All model scripts use the development-only 12x80/10/10 split protocol.
- ESCT-specific model, W1 transport, retrieval gate, and evaluation code is
  original to this repository.
