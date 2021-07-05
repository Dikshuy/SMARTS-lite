#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=def-mtaylor3
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16

module load singularity
module load arch/avx512 StdEnv/2018.3
singularity exec -B ../SMARTS-lite:/SMARTS-lite --env DISPLAY=$DISPLAY,PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite:$PYTHONPATH --home /SMARTS-lite/ultra ../smarts-0416_singularity.sif python ultra/hammer_train.py --task 0-3agents --level easy --policy ppo,ppo,ppo --headless
