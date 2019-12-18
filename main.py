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
import os


def simulate_pong(task_id, show=False):
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
            if (show):
                print(right_action)
                env.show(2)
                env.show_state(2)
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
    games = games[0]
    states = np.ndarray(shape=(0, Pong.HEIGHT//4, Pong.WIDTH//4), dtype=np.float32)
    actions = np.ndarray(shape=(0, 2, 3), dtype=np.float32)
    rewards = np.ndarray(shape=(0, 2), dtype=np.float32)
    for game in games:
        game_states, game_actions, game_rewards = game
        states = np.concatenate((game_states, states), axis=0)
        actions = np.concatenate((game_actions, actions), axis=0)
        rewards = np.concatenate((game_rewards, rewards), axis=0)
    return states, actions, rewards

def test_nnet(model):
    s = np.load('./test_data/states.npy')
    for i in range(20):
        predictions = model.infer(s[i])
        for j in predictions:
            print(j)

dqn = DQN(resume=False)
#dqn.show_weights(0)
#dqn.load_model('models/0.h5')
#dqn.show_weights(0)
if __name__ == '__main__':
    player.DeepQPlayer.EPSILON = 0
    for i in range(100):
        test_nnet(dqn)
        simulated_games = run_simulations(1000)
        s, a, r = simulated_games
        r = (r + 1) / 2
        dqn = DQN()
        dqn.retrain((s, a, r))
        #dqn.show_weights(0)
        dqn.save(f'{i}.h5')
        player.DeepQPlayer.EPSILON = int(0.5 + (i / 200))
