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


def simulate_pong(task_id):
    try:
        states = np.ndarray(shape=(0, Pong.HEIGHT//4, Pong.WIDTH//4), dtype=np.float32)
        actions = np.ndarray(shape=(0, 2, 3), dtype=np.float32)
        rewards = np.ndarray(shape=(0, 2), dtype=np.float32)

        env = Pong()
        right = player.DeepQPlayer(right=True)  #player.HumanPlayer('up', 'down')
        left = player.BotPlayer(env, left=True)
        last_screen = None
        screen = env.reset()
        done = False
        while not done:
            if last_screen is None:
                state = screen - np.zeros_like(screen, dtype=np.float32)
            else:
                state = screen - last_screen
            left_action = left.move(state)
            right_action = right.move(state)
            action = np.stack((encode_action(left_action), encode_action(right_action)))
            last_state = state
            state, reward, done = env.step(left_action, right_action)
            reward_l, reward_r = reward
            states = np.concatenate((states, np.expand_dims(last_state, axis=0)), axis=0)
            actions = np.concatenate((actions, np.expand_dims(action, axis=0)), axis=0)
            rewards = np.concatenate((rewards, np.asarray([[reward_l, reward_r]], dtype=np.float32)), axis=0)
        l, r = env.get_score()
        print(f"Finished game {task_id}, {l} - {r}")
        return states, actions, rewards
    except Exception as e:
        print(e)


def run_simulations(n):
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    games = []
    tasks = range(n)
    r = pool.map_async(simulate_pong, tasks, callback=games.append)
    Timer.start(f'{n} games')
    r.wait()  # Wait on the results
    Timer.stop(f'{n} games')
    states = np.ndarray(shape=(0, Pong.HEIGHT//4, Pong.WIDTH//4), dtype=np.float32)
    actions = np.ndarray(shape=(0, 2, 3), dtype=np.float32)
    rewards = np.ndarray(shape=(0, 2), dtype=np.float32)
    for game in games:
        game_states, game_actions, game_rewards = game[0]
        states = np.concatenate((game_states, states), axis=0)
        actions = np.concatenate((game_actions, actions), axis=0)
        rewards = np.concatenate((game_rewards, rewards), axis=0)
    return states, actions, rewards


if __name__ == '__main__':
    simulated_games = run_simulations(100000)
    s, a, r = simulated_games
    dqn = DQN()
    dqn.retrain(simulated_games)
