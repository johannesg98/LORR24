#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -t 7-00:00
#SBATCH -p cpu
#SBATCH -o slurm/output_%j.txt
#SBATCH -e slurm/error_%j.txt
#SBATCH --array=1-8

conda run -n myenv python3 RL/train_multiple_slurm.py $SLURM_ARRAY_TASK_ID