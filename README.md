# Atari Pong with Deep Reinforcement Learning
This repository contains all of the materials used to create [this video (and linked paper)](https://www.youtube.com/watch?v=tU5JZkZg4a0).

## Getting Started
1. (Optional) Enter a Python virtual environment if you intend to use one to run this application
2. Run `pip install -r requirements.txt`

## Repository Tour
A brief overview of the files in this repository and their purposes:
* `main.py`: Driver for training a new DRL Pong model from scratch
* `pong.py`: A custom pong implementation
* `player.py`: Various pong agents with a common interface
* `exhibit.py`, `output_visual.py`, `render_utils.py`: Scripts for visualizing a DRL model in real time as it plays a game of pong.
* `scripts/`: A directory of deployment scripts to run automatically on a remote cluster (currently configured for MSOE's ROSIE).
*  `initial_experiments/`: A directory of the first attempts at standing up a DRL pong solution, including some of the mistakes referenced in the paper.

## Training a Model
This repository provides the setup to train against either the OpenAI Gym Atari Emulator or a custom Pong implementation included in `pong.py`.

The driver for training a new model is `main.py`. Most hyperparameters and environment configuration can be specified via constants in this file. The defaults provided should provide decent results.

## Visualizing a Trained Model
The exhibit demo runs a Pong game using a pre-trained model, and shows its inputs and activations while playing against either a hard-coded or human opponent.

The driver for showing the exhibit demo is `exhibit.py`. To run against a custom trained model, you will need to change the path in `exhibit.py`. By default it will run against a hard-coded bot. Switch out the commented line initializing the opponent to run against a human-controlled paddle.

## Automated deployment on MSOE's ROSIE HPC Cluster
Scripts are provided to automate deploying via SLURM on ROSIE. To set this up:
1. Modify `scripts/spawn_job.bat` to use your valid MSOE email address
2. Set the `ROSIE_ACCESS` environment variable equal to your ROSIE password
3. Ensure you have [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) installed

Then, jobs can be deployed by running `spawn_job.bat {job name}`