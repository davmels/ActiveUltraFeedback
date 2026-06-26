#!/bin/bash

#SBATCH --job-name=online_po
#SBATCH -D .
#SBATCH -A infra01
#SBATCH --output=./logs/%j-%x.out
#SBATCH --error=./logs/%j-%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --reservation=SD-69241-apertus-1-5-0

# NOTE: the batch script runs on the HOST (no #SBATCH --environment), so that
# `srun` can load Slurm's host-side Slingshot plugin. The container is applied
# per-task via `srun --environment=...` below (the CSCS-recommended pattern).

export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Use local rewarduq source (prepend to PYTHONPATH so it overrides system install)
export PYTHONPATH="/iopsstor/scratch/cscs/dmelikidze/rewarduq/src:$PYTHONPATH"

# Defaults (override via environment variables)
ACQ_FUNC="${ACQ_FUNC:?ERROR: ACQ_FUNC not set}"
REWARD_MODEL="${REWARD_MODEL:-enn}"
REPLAY_BUFFER="${REPLAY_BUFFER:-100}"
EXP_DECAY_BASE="${EXP_DECAY_BASE:-0.9}"
ACQ_BETA="${ACQ_BETA:-1.0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:--}"
OUTER_BATCH_SIZE="${OUTER_BATCH_SIZE:-2048}"
PROMPT_K="${PROMPT_K:-256}"
MIN_NON_TRUNCATED="${MIN_NON_TRUNCATED:-2}"
DOMAIN_QUOTA="${DOMAIN_QUOTA:-false}"
DEBUG="${DEBUG:-false}"

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

RESUME_ARG=""
if [[ "$RESUME_CHECKPOINT" != "-" ]]; then
    RESUME_ARG="--resume_from_checkpoint=$RESUME_CHECKPOINT"
fi

DECAY_ARG=""
if [[ "$REWARD_MODEL" == "enn" ]]; then
    DECAY_ARG="--enn.regularization.exponential_decay_base=$EXP_DECAY_BASE"
fi

export CMD="accelerate launch \
    --config_file=configs/accelerate/multi_node.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    -m activeuf.loop.run \
    --config_path configs/loop.yaml \
    --acquisition_function_type=$ACQ_FUNC \
    --reward_model_type=$REWARD_MODEL \
    --replay_buffer_factor=$REPLAY_BUFFER \
    $DECAY_ARG \
    --acquisition_function.beta=$ACQ_BETA \
    --outer_loop_batch_size=$OUTER_BATCH_SIZE \
    --prompt_selection_K=$PROMPT_K \
    --min_non_truncated=$MIN_NON_TRUNCATED \
    --domain_quota_selection=$DOMAIN_QUOTA \
    --debug=$DEBUG \
    $RESUME_ARG"

srun --environment=activeuf_thesis bash -c "$CMD"
