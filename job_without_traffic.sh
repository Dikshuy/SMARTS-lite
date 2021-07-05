#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

module load singularity
singularity exec -B ../SMARTS-lite:/SMARTS-lite --env DISPLAY=$DISPLAY,PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite:$PYTHONPATH --home /SMARTS-lite/ultra ../smarts-0416_singularity.sif python ultra/hammer_train.py --task 0-3agents --level no-traffic --policy ppo,ppo,ppo --headless
