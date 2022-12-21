#!/bin/bash
#SBATCH --job-name=01
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=5G

module load Python

source /data/$USER/.envs/four_room/bin/activate



echo Starting Python program
python3 run.py

deactivate