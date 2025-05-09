#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH -t 7-00:00
#SBATCH -p cpu-gpu-v100
#SBATCH --gpus=1
#SBATCH -o slurm/output_%j.txt
#SBATCH -e slurm/error_%j.txt
#SBATCH --array=1-2

conda run -n myenv python3 nn_optimization/nn_test_multiple_slurm_TransformerConvExtended.py $SLURM_ARRAY_TASK_ID