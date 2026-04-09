#!/bin/bash
#SBATCH --job-name=sweep_runner
#SBATCH -D .
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH --output=./logs/sweeps/%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=4

# In order to get a sweep ID, first create a sweep on wandb using:
#
#   wandb sweep --entity <YOUR_WANDB_ENTITY> --project <PROJECT> [--name <SWEEP_NAME>] sweep_config.yaml
#
# Note: For this to work you need to be in an environment where wandb is installed. E.g. in a debug job.
# Then replace <SWEEP_ID> below with the actual sweep ID. 
# To start all of the jobs in the sweep, use:
#
#   sbatch --array=1-<NUMBER_OF_JOBS_FOR_SWEEP>%20 ./scripts/sweep_runner.sbatch
#
# This will launch <NUMBER_OF_JOBS_FOR_SWEEP>, each retrieving the hyperparameters from wandb.

wandb agent --count=1 <SWEEP_ID>