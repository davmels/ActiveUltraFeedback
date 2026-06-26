#!/bin/bash

#SBATCH --job-name=likelihood_filter
#SBATCH -D /iopsstor/scratch/cscs/dmelikidze/work/ActiveUltraFeedback
#SBATCH -A a-infra01-1
#SBATCH --output=logs/likelihood-%x.%j.out
#SBATCH --error=logs/likelihood-%x.%j.err
#SBATCH --nodes=1
#SBATCH --time=12:00:00

mkdir -p logs

cd /iopsstor/scratch/cscs/dmelikidze/work/ActiveUltraFeedback

srun --environment=activeuf --chdir=/iopsstor/scratch/cscs/dmelikidze/work/ActiveUltraFeedback python -u /iopsstor/scratch/cscs/dmelikidze/work/ActiveUltraFeedback/activeuf/advanced/likelihood_filter_vllm.py \
    --model-path=/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364 \
    --dataset-path=/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/combined_annotated \
    --output-path=likelihood_stats_vllm.json \
    --concurrent=500 \
    --output-dataset=./datasets/combined_annotated_likelihoods \
    --dp-size=4 \
    --slurm-nodes=4 \
    --workers=4
