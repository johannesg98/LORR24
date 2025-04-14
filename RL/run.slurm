#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -t 7-00:00
#SBATCH -p cpu
#SBATCH -o slurm_outputs/output_%j.txt
#SBATCH -e slurm_errors/error_%j.txt
#SBATCH --array=1-6

conda run -n myenv python3 train_multiple_slurm.py $SLURM_ARRAY_TASK_ID