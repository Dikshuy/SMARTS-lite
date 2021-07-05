#!/bin/bash
#SBATCH --account=def-mtaylor3
#SBATCH --time=00:20:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
# cd ~/projects/def-mtaylor3/dikshant/
# singularity shell --bind SMARTS-lite/:/SMARTS-lite --env DISPLAY=$DISPLAY smarts-0416_singularity.sif
# cd /SMARTS-lite/ultra/
# export PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite/:$PYTHONPATH
# python ultra/hammer_train.py --task 0-3agents --level easy --episodes 10 --eval-episodes 2 --eval-rate 5 --policy ppo,ppo,ppo --headless

cd ~/projects/def-mtaylor3/dikshant/
singularity exec -B /SMARTS-lite:/SMARTS-lite --env DISPLAY=$DISPLAY,PYTHONPATH=/SMARTS-lite/ultra:/SMARTS-lite/:/src --home /SMARTS-lite smarts-0416_singularity.sif python ultra/hammer_train.py --task 0-3agents --level easy --episodes 10 --eval-episodes 2 --eval-rate 5 --policy ppo,ppo,ppo --headless