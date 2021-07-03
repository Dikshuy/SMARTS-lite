#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=def-mtaylor3
module load singularity
singularity exec -B ../SMARTS-lite:/SMARTS-lite --env DISPLAY=$DISPLAY,PYTHONPATH=/SMARTS-lite/ultra:/src --home /SMARTS-lite/ultra ../smarts-0416_singularity.sif python ultra/hammer_train.py --task 0-3agents --level easy --episodes 10 --eval-episodes 2 --eval-rate 5 --policy ppo,ppo,ppo --headless
