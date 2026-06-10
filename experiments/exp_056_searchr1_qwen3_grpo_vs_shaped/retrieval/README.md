# Search-R1 retrieval server setup

This experiment needs an HTTP retrieval server running on `http://127.0.0.1:8000/retrieve` that the multi-turn rollout calls between turns.

We use the official Search-R1 setup (E5 dense retriever over wiki-18).

## Downloads (do once, ~33 GB on disk)

```bash
# 1. corpus (~3GB compressed, ~12GB uncompressed)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='PeterJinGo/wiki-18-corpus', repo_type='dataset',
                filename='wiki-18.jsonl.gz', local_dir='./retrieval/data')
"
gunzip retrieval/data/wiki-18.jsonl.gz

# 2. E5 FAISS index, served as two parts to dodge HF 50GB limit
python -c "
from huggingface_hub import hf_hub_download
for part in ('part_aa', 'part_ab'):
    hf_hub_download(repo_id='PeterJinGo/wiki-18-e5-index', repo_type='dataset',
                    filename=part, local_dir='./retrieval/data')
"
cat retrieval/data/part_aa retrieval/data/part_ab > retrieval/data/e5_Flat.index
rm retrieval/data/part_aa retrieval/data/part_ab

# 3. E5 retriever model (~440MB, downloaded on first use)
# nothing to do — the launch script pulls intfloat/e5-base-v2 lazily
```

## Launch the server

```bash
# Inside the unsloth container (or any env with faiss-gpu + sentence-transformers):
pip install fastapi uvicorn faiss-gpu sentence-transformers

git clone https://github.com/PeterGriffinJin/Search-R1.git /opt/search-r1
cd /opt/search-r1
python search_r1/search/retrieval_server.py \
  --index_path /mnt/data/aiim_research/experiments/exp_056_searchr1_qwen3_grpo_vs_shaped/retrieval/data/e5_Flat.index \
  --corpus_path /mnt/data/aiim_research/experiments/exp_056_searchr1_qwen3_grpo_vs_shaped/retrieval/data/wiki-18.jsonl \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu
```

Server listens on `0.0.0.0:8000`. Verify:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries": ["who is the president of the united states"], "topk": 3}'
```

## GPU notes

E5 retriever + FAISS-GPU index together use ~10-15 GB. On our single A100 80GB this competes with training. Options:

- Run retrieval on CPU (drop `--faiss_gpu`) — slower (~200ms/query vs ~30ms) but doesn't compete
- Run retrieval on a second GPU if available
- Time-share: pause training during the retrieval call (default — works because rollouts only need retrieval mid-completion, which is sequential anyway)

For initial smoke runs we'll start with CPU retrieval.

## Falling back to StubRetriever

Until the real server is up, `train.py` can use `StubRetriever` which returns canned doc strings per query. This is enough to validate the rollout / mask / reward pipeline but obviously won't train a useful policy.
