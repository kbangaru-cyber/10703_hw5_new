#!/bin/bash
#SBATCH --job-name=10703_hw5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=512G
#SBATCH --partition=general
#SBATCH --output=logs/rl_job%j.txt
#SBATCH --time=2-00:00:00

source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
cd /home/adityaku/HW/10703_HW5_new
python run.py
python train.py