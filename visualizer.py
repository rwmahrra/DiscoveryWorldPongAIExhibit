from pong import Pong
from random import choice
import time
import cv2
import numpy as np
import player
from utils import Timer, encode_action
import multiprocessing
from ai import DQN
import subprocess



def view_train_progression(number, neuron):
    dqns = []
    for i in range(number + 1):
        dqn = DQN(resume=False)
        dqn.load_model(f"./models/{i}.h5")
        dqns.append(dqn)
    for dqn in dqns:
        dqn.show_weights(neuron)


def test_model(id):

    dqn = DQN(resume=False)
    dqn.load_model(f"./models/{id}.h5")
    #for i in range(200):
    #    dqn.show_weights(i)
    env = Pong()
    player.DeepQPlayer.EPSILON = 1
    right = player.DeepQPlayer(right=True)
    right.set_model(dqn)
    left = player.BotPlayer(env, left=True)
    last_screen = None
    state = env.reset()
    done = False
    while not done:
        left_action = left.move(state)
        right_action = right.move(state)
        print(right_action)
        state, reward, done = env.step(left_action, right_action)
        #dqn.show_attention_map(state)
        env.show(2)
        env.show_state(2, 0)
    l, r = env.get_score()
    print(f"Finished game with model {id}, {l} - {r}")


#view_train_progression(12, 1)
test_model(12)