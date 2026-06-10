#!/usr/bin/env bash
# launch_server.sh — start the Search-R1 E5 retrieval server in a persistent
# docker container. FAISS runs on CPU (the 61GB Flat index loads into RAM;
# we have ~200GB), the E5 query encoder runs on GPU (~1GB, hardcoded .cuda()).
# Port 8123 (8000 is taken by another service on this host).
#
# Run this BEFORE training so the encoder's GPU allocation happens before
# vLLM grabs the rest of the card.

set -e

EXP_DIR="/mnt/data/aiim_research/experiments/exp_056_searchr1_qwen3_grpo_vs_shaped"
DATA_DIR="${EXP_DIR}/retrieval/data"
REPO_DIR="${EXP_DIR}/retrieval/search_r1_repo"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var not set}"
PORT="${RETRIEVAL_PORT:-8123}"
NAME="searchr1_retrieval"

# Stop any prior instance we started
docker rm -f "${NAME}" 2>/dev/null || true

echo "=== [$(date -Is)] Launching ${NAME} on port ${PORT} (FAISS=CPU, E5=GPU) ==="
docker run -d \
  --name "${NAME}" \
  --gpus all \
  --user root \
  --network=host \
  -v /mnt/data:/mnt/data \
  -e "HF_TOKEN=${HF_TOKEN}" \
  -e "RETRIEVAL_PORT=${PORT}" \
  --entrypoint /bin/bash \
  unsloth/unsloth -c "
    set -e
    source /opt/venv/bin/activate
    echo '[setup] installing faiss-cpu + fastapi + uvicorn...'
    uv pip install --quiet faiss-cpu fastapi uvicorn pydantic
    cd ${REPO_DIR}
    echo '[launch] retrieval_server.py (e5, faiss CPU)...'
    python search_r1/search/retrieval_server.py \
      --index_path ${DATA_DIR}/e5_Flat.index \
      --corpus_path ${DATA_DIR}/wiki-18.jsonl \
      --topk 3 \
      --retriever_name e5 \
      --retriever_model intfloat/e5-base-v2
  "

echo "container id:"
docker ps --filter "name=${NAME}" --format '{{.ID}}  {{.Status}}'
echo ""
echo "Tail startup with:  docker logs -f ${NAME}"
echo "It will take a few minutes to read the 61GB index + 14GB corpus into RAM."
echo "Verify when ready:"
echo "  curl -s -X POST http://127.0.0.1:${PORT}/retrieve -H 'Content-Type: application/json' -d '{\"queries\":[\"who wrote hamlet\"],\"topk\":3}'"
