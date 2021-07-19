# Atari Pong with Deep Reinforcement Learning
This repository contains all of the materials used to create [this video (and linked paper)](https://youtu.be/23g--Iu_cPE).

## To Play the Exhibit
* 'mosquitto -v -c ./mosquitto_br.conf': mosquitto MQTT bridge on linux computer 
* 'mosquitto -v -c ./mosquitto.conf': mosquitto MQTT server on windows computer
* 'python3 -m exhibit.ai.ai_driver': loads and gets actions from ai models on linux computer
* 'py -m exhibit.game.game_driver': the game module that runs the Pong environemnt on windows computer
* 'py -m exhibit.visualization.visualization_driver': visualization module that starts http server for the browser visualization
* 'Emulate 3D project fixed_pong': the emulate 3d project

## Getting Started
1. (Optional) Enter a Python virtual environment if you intend to use one to run this application
2. Run `pip install -r requirements_local.txt`

## Repository Tour
A brief overview of the folders in this repository and their purposes:
* `exhibit/ai`: AI model definition and MQTT interface
* `exhibit/game`: Custom pong implementation
* `exhibit/shared`: Configuration and utility methods
* `exhibit/train`: Scripts for simulated games and model training through deep reinforcement learning
* `exhibit/visualization`: Simple Python web server to serve visualizer (no data dependencies for the visualizer: just for ease of testing)
* `visualizer`: Javascript/HTML5 client code for the visualization module
* `scripts/`: A directory of deployment scripts to run automatically on a remote cluster (currently configured for MSOE's ROSIE).
* `initial_experiments/`: A directory of the first attempts at standing up a DRL pong solution, including some of the challenges referenced in the original paper.

## Training a Model
This repository provides the setup to train against the custom Pong implementation included in `pong.py`.

The driver for training a new model is `reinforcement.py`. Most hyperparameters and environment configuration can be specified via constants in this file and `simulator.py`.
The defaults provided should provide decent results.

## Visualizing a Trained Model (Exhibit Setup)
The following components must be running on the same network for a functioning deployment:
1. An MQTT server with websocket support (I used Eclipse Mosquitto during development. See `scripts/run_broker.bat`)
2. The AI module, `exhibit/ai/ai_driver.py`
3. The game module, `exhibit/game/game_driver.py`

To run the visualizer, also serve it locally using `visualization_driver.py` and access it from localhost:8000 within a web browser (or just load the HTML file directly from disk if your browser supports it).
There are no actual dependencies between the provided web server and the HTML5 visualization client: the client handles its own MQTT subscriptions and will run standalone.

The exhibit demo runs a Pong game using a pre-trained model, and shows its inputs and activations while playing against either a hard-coded or human opponent.
By default it will run against a hard-coded bot. Switch out the commented line initializing the opponent to run against a human-controlled paddle.

## Automated deployment on MSOE's ROSIE HPC Cluster
Scripts are provided to automate deploying via SLURM on ROSIE. To set this up:
1. Modify `scripts/spawn_job.bat` to use your valid MSOE email address
2. Set the `ROSIE_ACCESS` environment variable equal to your ROSIE password
3. Ensure you have [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) installed

Then, jobs can be deployed by running `spawn_job.bat {job name}`
