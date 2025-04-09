#!/bin/bash

## Change this to a job name you want
#SBATCH --job-name=python-job

## Change based on length of job and sinfo partitions available
#SBATCH --partition=gpu

## Request for a specific type of node
## Commented out for now, change if you need one
#SBATCH --constraint xgph

## gpu:1 ==> any gpu. For e.g., gpu:a100-40:1 gets you one of the A100 GPU shared instances
#SBATCH --gres=gpu:a100-40:1

## Probably no need to change anything here
#SBATCH --ntasks=1

## May want to change this depending on how much host memory you need
#SBATCH --mem-per-cpu=40G

## Just useful logfile names
#SBATCH --output=cs4248_%j.slurmlog
#SBATCH --error=cs4248_%j.slurmlog

echo "Job is running on $(hostname), started at $(date)"

source venv/bin/activate
srun python3 rembert_finetune.py 

echo -e "\nJob completed at $(date)"
