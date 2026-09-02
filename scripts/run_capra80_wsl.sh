#!/bin/bash
set -e

# WSL-specific helper for the full 80-epoch CAPRA baseline.
# Set before running:
#   ESCT_REPO_DIR=/path/to/repository/root
#   ESCT_WSL_CONDA=/opt/miniconda3/etc/profile.d/conda.sh (optional)
#   ESCT_WSL_ENV=esct (optional)
#   ESCT_LOG_DIR=/path/to/logs (optional, default $ESCT_REPO_DIR/outputs)
CONDA_SH="${ESCT_WSL_CONDA:-/opt/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ESCT_WSL_ENV:-esct}"
REPO_DIR="${ESCT_REPO_DIR:?set ESCT_REPO_DIR to the repository root}"
LOG_DIR="${ESCT_LOG_DIR:-$REPO_DIR/outputs}"
mkdir -p "$LOG_DIR"

source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO_DIR"
rm -f "$LOG_DIR/capra_full80.log" "$LOG_DIR/capra_full80.pid"
nohup env CAPRA_SPLITS=12 CAPRA_NPRED=200 PYTHONPATH="$PWD" \
  python -u scripts/run_capra_clean_protocol.py 12 80 \
  > "$LOG_DIR/capra_full80.log" 2>&1 &
echo "PID=$!" > "$LOG_DIR/capra_full80.pid"
sleep 3
cat "$LOG_DIR/capra_full80.pid"
