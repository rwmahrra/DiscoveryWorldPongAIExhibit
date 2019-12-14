from pong import Pong
from random import choice
import time
import cv2
import numpy as np
import player
from utils import Timer
import multiprocessing
import subprocess

def simulate_pong(task_id):
    data = []
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
        last_state = state
        state, reward, done = env.step(left_action, right_action)
        reward_l, reward_r = reward
        data.append((state, (left_action, right_action), (reward_l, reward_r)))
        #env.show(4)
        #env.show_state(1)
    #left.retrain()
    #right.retrain()
    l, r = env.get_score()
    print(f"Finished game {task_id}, {l} - {r}")
    return data

if __name__ == '__main__':
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    results = []
    tasks = range(10000)
    r = pool.map_async(simulate_pong, tasks, callback=results.append)
    Timer.start('10000 games')
    r.wait()  # Wait on the results
    Timer.stop('10000 games')
    print(count)
