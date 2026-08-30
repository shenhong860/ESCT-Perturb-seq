# Data

Download the processed scPerturb datasets from Zenodo DOI 14638780.

Expected inputs:

- Norman: `norman/norman_hvg1000.h5ad` + `embeddings/norman_scgpt_emb.pkl`
- Replogle_exp6: `Replogle_exp6/Replogle_exp6.h5ad` + `embeddings/replogle_exp6_scgpt_emb.pkl`
- Wessels: `Wessels/Wessels.h5ad` + `embeddings/wessels_scgpt_emb.pkl`
- DEG references: `deg/DEG_hvg5000_Norman.pkl`, `deg/DEG_hvg5000_Replogle_exp6.pkl`, `deg/DEG_hvg5000_Wessels.pkl`

For CAPRA, use the official GenePT embedding table from the CAPRA repository.

Large data files are not stored in this repository.

The defaults above can be overridden per dataset with `ESCT_NORMAN_H5AD`,
`ESCT_REPLOGLE_H5AD`, `ESCT_WESSELS_H5AD` and the corresponding `ESCT_*_EMB`
variables; see `scripts/paths.py`.
