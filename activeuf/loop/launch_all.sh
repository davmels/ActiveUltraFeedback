#!/bin/bash
# Launch loop jobs for all acquisition functions with per-function hyperparameters.
# Format: "acq_func replay_buffer exp_decay_base acq_beta [resume_checkpoint]"
# Leave resume_checkpoint as "-" or omit for fresh runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENN_CONFIGS=(
    # "drts        250  0.999   1.0 /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci/active_prompts/dpo/drts_enn_qwen_ultrafeedback_20260330-072904-200/checkpoint-900"
    "deltaucb    250  0.999   2.0"
    # "infomax     250  0.99   2.0 /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci/active_prompts/dpo/infomax_enn_qwen_ultrafeedback_20260330-072903-623/checkpoint-1000"
    # "dts         250  0.99   1.0 /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci/active_prompts/dpo/dts_enn_qwen_ultrafeedback_20260330-072904-690/checkpoint-900"
    # "maxminlcb   250  0.99   1.0 /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci/active_prompts/dpo/maxminlcb_enn_qwen_ultrafeedback_20260330-072902-926/checkpoint-900"
)

BLH_CONFIGS=(
    # "deltaucb    250  1.0" #try smaller beta from configs for blh.
)

# Minimum non-truncated completions required to keep a prompt (run.py filter).
MIN_NON_TRUNCATED=8

# Domain-quota prompt selection: cap each Olmo3 domain at floor(K * pool_fraction)
# on a ranked pass, then random-fill. Set "false" for the original (unconstrained) runs.
# Only affects the prompt-selection ablations (K != null); the L=256 baseline is skipped.
DOMAIN_QUOTA=true

# Format: "outer_batch_size prompt_K"
# Use "null" for prompt_K to disable prompt selection (baseline).
ABLATIONS=(
    "1024 256"
    "4096 256"
    "256  null"   # baseline: no prompt selection, L=256
)

# Static baseline: oracle_maxmin with LLM judge scores (no reward model)
# Format: "outer_batch_size prompt_K"
STATIC_CONFIGS=(
    # "1024 256"
    # "4096 256"
)

for config in "${ENN_CONFIGS[@]}"; do
    read -r acq replay decay beta resume <<< "$config"
    resume="${resume:--}"
    for ablation in "${ABLATIONS[@]}"; do
        read -r obs pk <<< "$ablation"
        quota_tag=""
        if [[ "$DOMAIN_QUOTA" == "true" ]]; then
            # Domain quota only changes behavior when prompts are actually subselected.
            if [[ "$pk" == "null" ]]; then
                echo "Skipping baseline (L=$obs, no selection) — domain_quota is a no-op without prompt_selection_K"
                continue
            fi
            quota_tag="_domquota"
        fi
        echo "Submitting: enn $acq (replay=$replay, decay=$decay, beta=$beta, L=$obs, K=$pk, domain_quota=$DOMAIN_QUOTA)"
        ACQ_FUNC=$acq \
        REWARD_MODEL=enn \
        REPLAY_BUFFER=$replay \
        EXP_DECAY_BASE=$decay \
        ACQ_BETA=$beta \
        RESUME_CHECKPOINT=$resume \
        OUTER_BATCH_SIZE=$obs \
        PROMPT_K=$pk \
        MIN_NON_TRUNCATED=$MIN_NON_TRUNCATED \
        DOMAIN_QUOTA=$DOMAIN_QUOTA \
        sbatch --job-name="online_po_enn_${acq}_L${obs}_K${pk}${quota_tag}" "$SCRIPT_DIR/run.sbatch"
    done
done

for config in "${BLH_CONFIGS[@]}"; do
    read -r acq replay beta resume <<< "$config"
    resume="${resume:--}"
    for ablation in "${ABLATIONS[@]}"; do
        read -r obs pk <<< "$ablation"
        echo "Submitting: blh $acq (replay=$replay, beta=$beta, L=$obs, K=$pk)"
        ACQ_FUNC=$acq \
        REWARD_MODEL=blh \
        REPLAY_BUFFER=$replay \
        ACQ_BETA=$beta \
        RESUME_CHECKPOINT=$resume \
        OUTER_BATCH_SIZE=$obs \
        PROMPT_K=$pk \
        sbatch --job-name="online_po_blh_${acq}_L${obs}_K${pk}" "$SCRIPT_DIR/run.sbatch"
    done
done

for config in "${STATIC_CONFIGS[@]}"; do
    read -r obs pk <<< "$config"
    echo "Submitting: static oracle_maxmin (L=$obs, K=$pk)"
    ACQ_FUNC=oracle_maxmin \
    REWARD_MODEL=static \
    REPLAY_BUFFER=1 \
    EXP_DECAY_BASE=1.0 \
    ACQ_BETA=1.0 \
    RESUME_CHECKPOINT=- \
    OUTER_BATCH_SIZE=$obs \
    PROMPT_K=$pk \
    sbatch --job-name="online_po_static_oracle_maxmin_L${obs}_K${pk}" "$SCRIPT_DIR/run.sbatch"
done

echo "All jobs submitted."
