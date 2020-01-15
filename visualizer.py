from pong import Pong
from random import choice
import time
import cv2
import numpy as np
import player
from utils import Timer, encode_action, discount_rewards
import multiprocessing
from ai import DQN
from main import run_simulations
import subprocess



def view_train_progression(number, neuron):
    dqns = []
    for i in range(number + 1):
        dqn = DQN(resume=False)
        dqn.load_model(f"./models/{i}.h5")
        dqns.append(dqn)
    for dqn in dqns:
        dqn.show_weights(neuron)


def visualize_game_memory():
    game = run_simulations(1, 0)
    game_states, game_actions, game_rewards = game
    rewards_l = discount_rewards(game_rewards[:, 0])
    rewards_r = discount_rewards(game_rewards[:, 1])
    actions_l = game_actions[:, 0]
    actions_r = game_actions[:, 1]
    for i in range(len(game_states)):
        state = game_states[i]
        reward = rewards_r[i]
        print(reward)
        if actions_r[i][0] == 1: print("UP " + str(actions_r[i]))
        else: print("DOWN " + str(actions_r[i]))
        cv2.imshow("state", state)
        cv2.waitKey(0)

def test_model(id):
    dqn = DQN(resume=False)
    dqn.load_model(f"./models/{id}.h5")
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
        right_action = right.move(state, debug=True)
        print(right_action)
        state, reward, done = env.step(left_action, right_action)
        #dqn.show_attention_map(state)
        env.show(2)
        env.show_state(2, 0)
    l, r = env.get_score()
    print(f"Finished game with model {id}, {l} - {r}")

def view_weights(id):
    dqn = DQN(resume=False)
    dqn.load_model(f"./models/{id}.h5")
    for i in range(200):
        dqn.show_weights(i)

def debug_step():
    env = Pong()
    right = player.HumanPlayer('up', 'down')
    left = player.BotPlayer(env, left=True)
    last_screen = None
    state = env.reset()
    done = False
    while not done:
        left_action = left.move(state)
        right_action = right.move(state)
        action = np.stack((encode_action(left_action), encode_action(right_action)))
        start_state = state
        state, reward, done = env.step(left_action, right_action)
        reward_l, reward_r = reward

        env.show(duration=0)
        env.show_state(duration=0)


test_model(4004)
#view_train_progression(100, 0)
#view_weights(4004)
#debug_step()