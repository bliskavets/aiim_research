#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /root/aiim/.venv/bin/activate; source /root/aiim/.env.session
export PYTHONUNBUFFERED=1
log(){ echo "[ae] $1 $(date '+%F %T')"; }
# uses the already-running 8B server (no restart)
log "gen baseline"
python experiments/thinking/run_alpaca_gen.py --method baseline --num-samples 200 --seed 42 --batch-size 24 --output logs/alpaca_baseline_outputs.json > logs/queue_alpaca_base.log 2>&1 && log "baseline gen DONE" || log "baseline gen FAIL"
log "gen sage"
python experiments/thinking/run_alpaca_gen.py --method sage --num-samples 200 --seed 42 --batch-size 12 --output logs/alpaca_sage_outputs.json > logs/queue_alpaca_sage.log 2>&1 && log "sage gen DONE" || log "sage gen FAIL"
log "ALL DONE"
