
# This script reads in a Keras model and returns a JSON representation. It is a quick hack, but is
# necessary since sending the full model weights over MQTT is too slow.

import struct
import time
import sys

sys.path.append("C:\dev\DiscoveryWorldPongAIExhibit")
print(sys.path)

import json
import numpy as np
from exhibit.shared.config import Config
from exhibit.ai.model import PGAgent

MODEL = './validation/level3_10000.h5'#'validation/canstop_randomstart_3k.h5'# 10k.h5'
DIFFICULTY = 2  # Easy: 0, Medium: 1, Hard: 2
DIFFICULTY_OUTPUTS = ['visualizer/models/easy.js', 'visualizer/models/medium.js', 'visualizer/models/hard.js']
DIFFICULTY_VARS = ['easy_model', 'medium_model', 'hard_model']
agent = PGAgent(Config.CUSTOM_STATE_SIZE, Config.CUSTOM_ACTION_SIZE)
agent.load(MODEL)
layers = []
i = 0
for w in agent.model.weights:
    l = None
    if i == 0:  # Rotate first weight matrix as temporary solution for rotated
        l = np.rot90(w.numpy().reshape(*Config.CUSTOM_STATE_SHAPE, -1), axes=(0, 1), k=1)
        l = l.reshape(Config.CUSTOM_STATE_SIZE, 200).tolist()
    else:
        l = w.numpy().tolist()

    layers.append(l)
    i += 1

print("Converting to json representation...")
jsonString = json.dumps(layers)
jsonFile = open(DIFFICULTY_OUTPUTS[DIFFICULTY], "w")

print("Writing output...")
jsonFile.write(f"// Generated from {MODEL}\n")
jsonFile.write(f"const {DIFFICULTY_VARS[DIFFICULTY]} = ")
jsonFile.write(jsonString)

print("Conversion complete.")