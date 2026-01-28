#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH -t 7-00:00
#SBATCH -o slurm/output_%j.txt
#SBATCH -e slurm/error_%j.txt
#SBATCH --array=1-2
#SBATCH --partition=cpu-gpu-v100
#SBATCH --gpus=1

conda run -n myenv python3 RL/train_multiple_slurm.py $SLURM_ARRAY_TASK_ID