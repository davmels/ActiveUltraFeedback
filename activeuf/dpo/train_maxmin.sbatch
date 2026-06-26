#!/bin/bash

#SBATCH --job-name=dpo_maxmin
#SBATCH -D .
#SBATCH -A infra01
#SBATCH --output=./logs/active_dpo/%j-%x.out
#SBATCH --error=./logs/active_dpo/%j-%x.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
##SBATCH --reservation=SD-69241-apertus-1-5-0

# Host-side batch script; container applied per-task via srun --environment.
export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

DATASET="${DATASET:?ERROR: DATASET not set (path to the MaxMin DPO dataset)}"
CONFIG="${CONFIG:-configs/dpo_maxmin_baseline.yaml}"

# Unique per-run output dir (jobid + timestamp). Computed once on the host so it is
# identical across all srun tasks/nodes. Overrides output_dir/run_name from the YAML.
RUN_TAG="${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S)"
OUTDIR="trainer_output/dpo_maxmin/${RUN_TAG}"

head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# One accelerate launcher per node; each sets its own machine_rank from SLURM_NODEID.
# \$SLURM_NODEID is escaped so it is evaluated per-task inside srun.
export CMD="accelerate launch \
    --config_file=configs/accelerate/deepspeed2.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    -m activeuf.dpo.training \
    --dataset_path $DATASET \
    --config_path $CONFIG \
    --output_dir=$OUTDIR \
    --run_name=dpo_maxmin_${RUN_TAG}"

srun --environment=activeuf_thesis bash -c "$CMD"
