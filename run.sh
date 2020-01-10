#!/bin/bash
#SBATCH --job-name=pong_train.job
#SBATCH --output=pong_train.out
python3 ~/DiscoveryWorldPongAIExhibit/main.py
