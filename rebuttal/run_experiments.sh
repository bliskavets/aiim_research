#!/usr/bin/env bash
# =============================================================================
# SAGE ICML 2026 Rebuttal — Experiment Runner
# =============================================================================
# Usage:
#   bash run_experiments.sh [PLAN]
#
# Plans:
#   A   — Minimal (~10 H100-hours, ~3 days): A1 + A2 + A3
#   B   — Average (~40 H100-hours, ~10 days): A + B2 + B3 + B4
#   C   — Maximal (~80 H100-hours, ~25 days): A + B + C1 + C2 + C3
#   all — Run every experiment
#
#   Individual experiment IDs: a1 a2 a3 b2 b3 b4 c1 c2 c3
#
# Examples:
#   bash run_experiments.sh A          # Plan A only
#   bash run_experiments.sh a2 a3      # Run A2 and A3 only
#   bash run_experiments.sh all        # Everything
#
# Environment variables (override defaults):
#   IP            vLLM server host           (default: localhost)
#   PORT          vLLM server port           (default: 9090)
#   MODEL         Qwen3-8B model identifier  (default: Qwen/Qwen3-8B-FP8)
#   MODEL_32B     Qwen3-32B model identifier (default: Qwen/Qwen3-32B-FP8)
#   MODEL_SMALL   Qwen3-1.7B model identifier(default: Qwen/Qwen3-1.7B-FP8)
#   SEED          Global random seed         (default: 42)
#   LOGS          Output root directory      (default: ./logs)
#   BATCH         Async batch size           (default: 4)
#   EPOCHS        SAGE optimization epochs   (default: 2)
#   N_GENS        Generations per epoch      (default: 7)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — override via environment variables
# ---------------------------------------------------------------------------
IP="${IP:-localhost}"
PORT="${PORT:-9090}"
MODEL="${MODEL:-Qwen/Qwen3-8B-FP8}"
MODEL_32B="${MODEL_32B:-Qwen/Qwen3-32B-FP8}"
MODEL_SMALL="${MODEL_SMALL:-Qwen/Qwen3-1.7B-FP8}"
SEED="${SEED:-42}"
LOGS="${LOGS:-./logs}"
BATCH="${BATCH:-4}"
EPOCHS="${EPOCHS:-2}"
N_GENS="${N_GENS:-7}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS="${SCRIPT_DIR}/configs"
EXP="${SCRIPT_DIR}/experiments"

MATH_PROMPT="${CONFIGS}/math500_judge_prompt.txt"
MATH_CONFIG="${CONFIGS}/math500_judge_config.json"
IFEVAL_PROMPT="${CONFIGS}/ifeval_judge_prompt.txt"
IFEVAL_CONFIG="${CONFIGS}/ifeval_judge_config.json"

# Common flags shared by most scripts
COMMON="--ip ${IP} --port ${PORT} --seed ${SEED} --batch-size ${BATCH}"
MODEL_FLAG="--model-name ${MODEL}"
JUDGE_FLAGS_MATH="--judge-prompt ${MATH_PROMPT} --judge-config ${MATH_CONFIG}"
JUDGE_FLAGS_IFEVAL="--judge-prompt ${IFEVAL_PROMPT} --judge-config ${IFEVAL_CONFIG}"
SAGE_FLAGS="--num-optimization-epochs ${EPOCHS} --number-of-gens-per-epoch ${N_GENS}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
sep()  { echo; echo "=== $* ==="; echo; }
skip() { log "SKIP: $1 (already completed — delete ${LOGS}/$2 to re-run)"; }

# Check if an output directory is already populated (resume protection)
already_done() {
    local dir="${LOGS}/$1"
    [[ -d "$dir" ]] && [[ -n "$(ls -A "$dir" 2>/dev/null)" ]]
}

mkdir -p "${LOGS}"

# ---------------------------------------------------------------------------
# Individual experiment functions
# ---------------------------------------------------------------------------

run_a1() {
    sep "A1 — Baseline Clarification (Qwen3-8B non-thinking on MATH-500)"
    local outdir="${LOGS}/a1_baseline"
    if already_done a1_baseline; then skip "A1" a1_baseline; return; fi
    python "${EXP}/a1_baseline_clarification/run_math500_baseline.py" \
        ${COMMON} ${MODEL_FLAG} \
        --num-samples 500 \
        --output-path "${outdir}"
    log "A1 done. Results in ${outdir}"
}

run_a2() {
    sep "A2 — Latency Table (SAGE vs BoN vs baseline on 100 MATH-500 problems)"
    local outdir="${LOGS}/a2_latency"
    if already_done a2_latency; then skip "A2" a2_latency; return; fi
    python "${EXP}/a2_latency_table/run_latency_comparison.py" \
        ${COMMON} ${MODEL_FLAG} \
        --num-samples 100 \
        --n-candidates 7 \
        --sage-epochs "${EPOCHS}" \
        ${JUDGE_FLAGS_MATH} \
        --output-path "${outdir}"
    log "A2 done. Results in ${outdir}"
}

run_a3() {
    sep "A3 — IFEval (SAGE + Baseline + BoN)"
    local outdir_sage="${LOGS}/a3_ifeval_sage"
    local outdir_base="${LOGS}/a3_ifeval_baseline"
    local outdir_bon="${LOGS}/a3_ifeval_bon"

    # Baseline
    if already_done a3_ifeval_baseline; then
        skip "A3 baseline" a3_ifeval_baseline
    else
        log "Running IFEval baseline..."
        python "${EXP}/a3_ifeval/run_ifeval_baseline.py" \
            ${COMMON} ${MODEL_FLAG} \
            --num-samples 541 \
            --output-path "${outdir_base}"
        log "A3 baseline done."
    fi

    # Best-of-N
    if already_done a3_ifeval_bon; then
        skip "A3 BoN" a3_ifeval_bon
    else
        log "Running IFEval BoN..."
        python "${EXP}/a3_ifeval/run_ifeval_bon.py" \
            ${COMMON} ${MODEL_FLAG} \
            --num-samples 541 \
            --n-candidates 7 \
            ${JUDGE_FLAGS_IFEVAL} \
            --output-path "${outdir_bon}"
        log "A3 BoN done."
    fi

    # SAGE
    if already_done a3_ifeval_sage; then
        skip "A3 SAGE" a3_ifeval_sage
    else
        log "Running IFEval SAGE..."
        python "${EXP}/a3_ifeval/run_ifeval_sage.py" \
            ${COMMON} ${MODEL_FLAG} \
            --num-samples 541 \
            ${JUDGE_FLAGS_IFEVAL} \
            ${SAGE_FLAGS} \
            --output-path "${outdir_sage}"
        log "A3 SAGE done."
    fi

    log "A3 done. Results in ${LOGS}/a3_ifeval_*"
}

run_b2() {
    sep "B2 — m_min Ablation (sweep {1,2,4,6,8} on MATH-500)"
    local outdir="${LOGS}/b2_mmin_ablation"
    if already_done b2_mmin_ablation; then skip "B2" b2_mmin_ablation; return; fi
    python "${EXP}/b2_mmin_ablation/run_mmin_sweep.py" \
        ${COMMON} ${MODEL_FLAG} \
        --num-samples 500 \
        --m-min-values "1,2,4,6,8" \
        ${JUDGE_FLAGS_MATH} \
        ${SAGE_FLAGS} \
        --output-path "${outdir}"
    log "B2 done. Results in ${outdir}"
}

run_b3() {
    sep "B3 — Aspect Sensitivity (3 configs × MATH-500 + IFEval)"
    for aspect in default generic task_specific; do
        for benchmark in math500 ifeval; do
            local outdir="${LOGS}/b3_aspect_${aspect}_${benchmark}"
            if already_done "b3_aspect_${aspect}_${benchmark}"; then
                skip "B3 ${aspect}/${benchmark}" "b3_aspect_${aspect}_${benchmark}"
                continue
            fi
            log "Running B3: aspect=${aspect} benchmark=${benchmark}..."
            python "${EXP}/b3_aspect_sensitivity/run_aspect_ablation.py" \
                ${COMMON} ${MODEL_FLAG} \
                --aspect-config "${aspect}" \
                --benchmark "${benchmark}" \
                --num-samples 500 \
                ${SAGE_FLAGS} \
                --output-path "${outdir}"
            log "B3 ${aspect}/${benchmark} done."
        done
    done
    log "B3 done."
}

run_b4() {
    sep "B4 — Small Model (Qwen3-1.7B on MATH-500 + IFEval)"
    local small_flag="--model-name ${MODEL_SMALL}"

    for benchmark in math500 ifeval; do
        local outdir="${LOGS}/b4_small_${benchmark}"
        if already_done "b4_small_${benchmark}"; then
            skip "B4 ${benchmark}" "b4_small_${benchmark}"
            continue
        fi
        log "Running B4: ${benchmark} with ${MODEL_SMALL}..."
        if [[ "$benchmark" == "math500" ]]; then
            python "${EXP}/b4_small_model/run_small_model_math500.py" \
                ${COMMON} ${small_flag} \
                --num-samples 500 \
                ${JUDGE_FLAGS_MATH} \
                ${SAGE_FLAGS} \
                --output-path "${outdir}"
        else
            python "${EXP}/b4_small_model/run_small_model_ifeval.py" \
                ${COMMON} ${small_flag} \
                --num-samples 541 \
                ${JUDGE_FLAGS_IFEVAL} \
                ${SAGE_FLAGS} \
                --output-path "${outdir}"
        fi
        log "B4 ${benchmark} done."
    done
    log "B4 done."
}

run_c1() {
    sep "C1 — Qwen3-32B (MATH-500 + IFEval)"
    local big_flag="--model-name ${MODEL_32B}"

    for benchmark in math500 ifeval; do
        local outdir="${LOGS}/c1_qwen32b_${benchmark}"
        if already_done "c1_qwen32b_${benchmark}"; then
            skip "C1 ${benchmark}" "c1_qwen32b_${benchmark}"
            continue
        fi
        log "Running C1: ${benchmark} with ${MODEL_32B}..."
        python "${EXP}/c1_qwen32b/run_qwen32b.py" \
            ${COMMON} ${big_flag} \
            --benchmark "${benchmark}" \
            --num-samples 500 \
            ${JUDGE_FLAGS_MATH} \
            ${SAGE_FLAGS} \
            --output-path "${outdir}"
        log "C1 ${benchmark} done."
    done
    log "C1 done."
}

run_c2() {
    sep "C2 — Updated RM Baseline (Skywork-Reward-Qwen-2.5-7B on MATH-500 + IFEval)"
    for benchmark in math500 ifeval; do
        local outdir="${LOGS}/c2_updated_rm_${benchmark}"
        if already_done "c2_updated_rm_${benchmark}"; then
            skip "C2 ${benchmark}" "c2_updated_rm_${benchmark}"
            continue
        fi
        log "Running C2: ${benchmark}..."
        python "${EXP}/c2_updated_rm/run_updated_rm.py" \
            ${COMMON} ${MODEL_FLAG} \
            --benchmark "${benchmark}" \
            --reward-model "Skywork/Skywork-Reward-Qwen-2.5-7B-v0.2" \
            --num-samples 500 \
            --n-candidates 7 \
            --output-path "${outdir}"
        log "C2 ${benchmark} done."
    done
    log "C2 done."
}

run_c3() {
    sep "C3 — MMLU-Pro STEM (SAGE + Baseline)"
    local outdir_base="${LOGS}/c3_mmlu_pro_baseline"
    local outdir_sage="${LOGS}/c3_mmlu_pro_sage"

    if already_done c3_mmlu_pro_baseline; then
        skip "C3 baseline" c3_mmlu_pro_baseline
    else
        log "Running C3 baseline..."
        python "${EXP}/c3_mmlu_pro/run_mmlu_pro_baseline.py" \
            ${COMMON} ${MODEL_FLAG} \
            --num-samples 2000 \
            --output-path "${outdir_base}"
        log "C3 baseline done."
    fi

    if already_done c3_mmlu_pro_sage; then
        skip "C3 SAGE" c3_mmlu_pro_sage
    else
        log "Running C3 SAGE..."
        python "${EXP}/c3_mmlu_pro/run_mmlu_pro_sage.py" \
            ${COMMON} ${MODEL_FLAG} \
            --num-samples 2000 \
            ${JUDGE_FLAGS_MATH} \
            ${SAGE_FLAGS} \
            --output-path "${outdir_sage}"
        log "C3 SAGE done."
    fi

    log "C3 done."
}

# ---------------------------------------------------------------------------
# Plan dispatcher
# ---------------------------------------------------------------------------

TARGETS="${*:-A}"  # Default to Plan A if nothing passed

run_plan_A() { run_a1; run_a2; run_a3; }
run_plan_B() { run_plan_A; run_b2; run_b3; run_b4; }
run_plan_C() { run_plan_B; run_c1; run_c2; run_c3; }

for target in $TARGETS; do
    case "${target,,}" in
        a)   run_plan_A ;;
        b)   run_plan_B ;;
        c)   run_plan_C ;;
        all) run_plan_C ;;
        a1)  run_a1 ;;
        a2)  run_a2 ;;
        a3)  run_a3 ;;
        b2)  run_b2 ;;
        b3)  run_b3 ;;
        b4)  run_b4 ;;
        c1)  run_c1 ;;
        c2)  run_c2 ;;
        c3)  run_c3 ;;
        *)   echo "Unknown target: ${target}"; echo "Valid: A B C all a1 a2 a3 b2 b3 b4 c1 c2 c3"; exit 1 ;;
    esac
done

log "All requested experiments finished. Logs under ${LOGS}/"
