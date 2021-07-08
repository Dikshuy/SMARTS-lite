#!/bin/bash
echo "evaluation starting............"
cd ~/github/SMARTS-lite/ultra/
source .VENV/bin/activate
echo $pwd
python ultra/evaluate.py --task 0-3agents --level easy --models logs/compute\ canada/experiment-2021.7.5-19\:22\:44-ppo-v0\:ppo-v0\:ppo-v0/models/002 --headless --experiment-dir logs/compute\ canada/experiment-2021.7.5-19\:22\:44-ppo-v0\:ppo-v0\:ppo-v0/
python ultra/evaluate.py --task 0-3agents --level no-traffic --models logs/compute\ canada/experiment-2021.7.5-19\:55\:42-ppo-v0\:ppo-v0\:ppo-v0/models/000 --headless --experiment-dir logs/compute\ canada/experiment-2021.7.5-19\:55\:42-ppo-v0\:ppo-v0\:ppo-v0/
python ultra/evaluate.py --task 0-3agents --level no-traffic --models logs/compute\ canada/experiment-2021.7.5-19\:55\:42-ppo-v0\:ppo-v0\:ppo-v0/models/001 --headless --experiment-dir logs/compute\ canada/experiment-2021.7.5-19\:55\:42-ppo-v0\:ppo-v0\:ppo-v0/
python ultra/evaluate.py --task 0-3agents --level easy --models logs/compute\ canada/experiment-2021.7.5-19\:22\:44-ppo-v0\:ppo-v0\:ppo-v0/models/003 --headless --experiment-dir logs/compute\ canada/experiment-2021.7.5-19\:22\:44-ppo-v0\:ppo-v0\:ppo-v0/
python ultra/evaluate.py --task 0-3agents --level no-traffic --models logs/compute\ canada/experiment-2021.7.5-19\:55\:42-ppo-v0\:ppo-v0\:ppo-v0/models/003 --headless --experiment-dir logs/compute\ canada/experiment-2021.7.5-19\:55\:42-ppo-v0\:ppo-v0\:ppo-v0/
echo "evaluation done................"
