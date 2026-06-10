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
echo "=== [$(date -Is)] Decompress + assemble (space-safe) ==="
cd "${DATA_DIR}"
ls -lh wiki-18.jsonl.gz part_a* 2>&1
# CAUTION: do NOT do `cat part_aa part_ab > e5_Flat.index`. That keeps all
# three files (40+21+61 = 122GB) on disk simultaneously and OOMs the FS on
# hosts with <130GB free at this point in the pipeline. Instead:
#  1. delete the .gz after a non-keep gunzip so we recover 4.8GB
#  2. rename part_aa to e5_Flat.index (instant, no copy)
#  3. append part_ab to e5_Flat.index, then rm part_ab
# Peak extra disk during step 3 is only ~21GB (the size of part_ab).
if [ -f wiki-18.jsonl.gz ] && [ ! -s wiki-18.jsonl ]; then
  rm -f wiki-18.jsonl    # might be a partial leftover from a prior attempt
  echo "decompressing wiki-18.jsonl.gz -> wiki-18.jsonl (deleting .gz)..."
  gunzip wiki-18.jsonl.gz
fi
if [ -f part_aa ] && [ -f part_ab ] && [ ! -f e5_Flat.index ]; then
  echo "mv part_aa -> e5_Flat.index ..."
  mv part_aa e5_Flat.index
  echo "append part_ab to e5_Flat.index then rm part_ab ..."
  cat part_ab >> e5_Flat.index && rm part_ab
fi

echo ""
echo "=== [$(date -Is)] Final state ==="
ls -lh "${DATA_DIR}"
df -h /mnt/data | tail -1
echo "Done."
