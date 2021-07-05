#!/bin/bash
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --mem=32000M
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2

module load singularity
singularity exec -B ../SMARTS-lite:/SMARTS-lite --env DISPLAY=$DISPLAY,PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite:$PYTHONPATH --home /SMARTS-lite/ultra ../smarts-0416_singularity.sif python ultra/hammer_train.py --task 0-3agents --level easy --policy ppo,ppo,ppo --headless
