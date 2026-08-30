#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate cw1ot
cd /mnt/c/Users/Administrator/Documents/ChatGPT/虚拟细胞/dc_w1t
rm -f /mnt/e/guo/capra_full80.log /mnt/e/guo/capra_full80.pid
nohup env CAPRA_SPLITS=12 CAPRA_NPRED=200 PYTHONPATH="$PWD" \
  python -u run_capra_clean_protocol.py 12 80 \
  > /mnt/e/guo/capra_full80.log 2>&1 &
echo "PID=$!" > /mnt/e/guo/capra_full80.pid
sleep 3
cat /mnt/e/guo/capra_full80.pid
