#!/bin/bash
#SBATCH --account=def-mtaylor3
#SBATCH --time=00:20:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
module load singularity
cd ~/projects/def-mtaylor3/dikshant/SMARTS-lite
singularity shell --bind SMARTS-lite/:/SMARTS-lite --env DISPLAY=$DISPLAY smarts-0416_singularity.sif
cd ~projects/def-mtaylor3/dikshant/SMARTS-lite/ultra
export PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite/:$PYTHONPATH
python ultra/hammer_train.py --task 0-3agents --level easy --episodes 10 --eval-episodes 2 --eval-rate 5 --policy ppo,ppo,ppo --headless

