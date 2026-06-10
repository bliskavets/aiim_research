#!/usr/bin/env bash
# download.sh — fetch wiki-18 corpus + E5 FAISS index for Search-R1 retrieval.
# Runs in unsloth docker so it gets a recent huggingface_hub.

set -e
set -o pipefail

EXP_DIR="/mnt/data/aiim_research/experiments/exp_056_searchr1_qwen3_grpo_vs_shaped"
DATA_DIR="${EXP_DIR}/retrieval/data"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"

mkdir -p "${DATA_DIR}"

echo "=== [$(date -Is)] Free space before download ==="
df -h /mnt/data | tail -1

echo ""
echo "=== [$(date -Is)] Download wiki-18 corpus ==="
docker run --rm --entrypoint /bin/bash \
  --user root \
  -v /mnt/data:/mnt/data \
  -e "HF_TOKEN=${HF_TOKEN}" \
  unsloth/unsloth -c "
    source /opt/venv/bin/activate
    python -c \"
from huggingface_hub import hf_hub_download
import os
local = '${DATA_DIR}'
os.makedirs(local, exist_ok=True)
# 1. corpus (~3GB compressed)
print('-> wiki-18.jsonl.gz ...')
hf_hub_download(repo_id='PeterJinGo/wiki-18-corpus', repo_type='dataset',
                filename='wiki-18.jsonl.gz', local_dir=local, token=os.environ['HF_TOKEN'])
# 2. E5 FAISS index parts
for part in ('part_aa', 'part_ab'):
    print(f'-> {part} ...')
    hf_hub_download(repo_id='PeterJinGo/wiki-18-e5-index', repo_type='dataset',
                    filename=part, local_dir=local, token=os.environ['HF_TOKEN'])
print('downloads done')
\"
  "

echo ""
echo "=== [$(date -Is)] Decompress + assemble ==="
cd "${DATA_DIR}"
ls -lh wiki-18.jsonl.gz part_a* 2>&1
if [ -f wiki-18.jsonl.gz ] && [ ! -f wiki-18.jsonl ]; then
  echo "decompressing wiki-18.jsonl.gz -> wiki-18.jsonl ..."
  gunzip -k wiki-18.jsonl.gz
fi
if [ -f part_aa ] && [ -f part_ab ] && [ ! -f e5_Flat.index ]; then
  echo "concat part_aa + part_ab -> e5_Flat.index ..."
  cat part_aa part_ab > e5_Flat.index
  echo "removing parts"
  rm part_aa part_ab
fi

echo ""
echo "=== [$(date -Is)] Final state ==="
ls -lh "${DATA_DIR}"
df -h /mnt/data | tail -1
echo "Done."
