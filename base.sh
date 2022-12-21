#!/bin/bash
#SBATCH --job-name=para
#SBATCH --time=15:00:00
#SBATCH --mem=4G

module load Python

source /data/$USER/.envs/four_room/bin/activate

echo Starting Python program
python3 training_ddqn02.py $*

deactivate