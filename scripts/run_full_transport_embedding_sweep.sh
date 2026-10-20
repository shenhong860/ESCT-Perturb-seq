#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="${1:-${repo_root}/outputs/full_transport_embedding_20261020}"
python_bin="${PYTHON_BIN:-python}"
mkdir -p \
  "${base}/results/norman" \
  "${base}/results/replogle_exp6" \
  "${base}/results/wessels" \
  "${base}/logs"

run_one() {
  local model_name="$1"
  local folder="$2"
  local representation="$3"
  local embedding="${base}/embeddings/${folder}/${representation}.pkl"
  local output="${base}/results/${folder}/${representation}.json"
  local log="${base}/logs/${folder}_${representation}.log"

  if [[ ! -f "${embedding}" ]]; then
    echo "missing embedding: ${embedding}" >&2
    return 1
  fi
  echo "START ${folder} ${representation} $(date -Is)"
  ESCT_SPLITS="${ESCT_SPLITS:-12}" \
  ESCT_MODEL_SEEDS="${ESCT_MODEL_SEEDS:-0,1}" \
  ESCT_ITERS="${ESCT_ITERS:-500}" \
  ESCT_BATCH="${ESCT_BATCH:-128}" \
  ESCT_N_CELLS="${ESCT_N_CELLS:-200}" \
  ESCT_MAX_TRAIN_CELLS="${ESCT_MAX_TRAIN_CELLS:-300}" \
    "${python_bin}" "${repo_root}/scripts/run_esct_w1_devgamma_clean.py" \
      "${model_name}" "${output}" \
      --embedding-pkl "${embedding}" \
      --representation "${representation}" >"${log}" 2>&1
  echo "DONE ${folder} ${representation} $(date -Is)"
}

for representation in scGPT Geneformer coexpression_PCA random_gene_ID GenePT; do
  run_one norman norman "${representation}"
done
for representation in scGPT Geneformer coexpression_PCA random_gene_ID; do
  run_one replogle replogle_exp6 "${representation}"
done
for representation in scGPT Geneformer coexpression_PCA random_gene_ID; do
  run_one wessels wessels "${representation}"
done

echo "ALL_DONE"
