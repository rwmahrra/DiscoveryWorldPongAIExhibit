# DISABLE GPUs for nicer threading
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

def simulate_pong(task_id, show=False):
    try:
        states = np.ndarray(shape=(0, Pong.HEIGHT//4, Pong.WIDTH//4), dtype=np.float32)
        actions = np.ndarray(shape=(0, 2, 2), dtype=np.float32)
        rewards = np.ndarray(shape=(0, 2), dtype=np.float32)
        env = Pong()
        right = player.DeepQPlayer(right=True)  #player.HumanPlayer('up', 'down')
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
            states = np.concatenate((states, np.expand_dims(start_state, axis=0)), axis=0)
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


def run_simulations(n, threads):
    games = []
    tasks = range(n)
    if threads != 0:
        count = multiprocessing.cpu_count()
        to_use = count - 1
        if threads > 0:
            to_use = min(threads, count)
        pool = multiprocessing.Pool(processes=to_use)
        print(f"Running on {to_use} threads")
        r = pool.map_async(simulate_pong, tasks, callback=games.append)
        Timer.start(f'{n} games')
        r.wait()  # Wait on the results
        Timer.stop(f'{n} games')
        games = games[0]
    else:
        tasks = range(n)
        Timer.start(f'{n} games')
        for task in tasks:
            games.append(simulate_pong(task))
        Timer.stop(f'{n} games')

    states = np.ndarray(shape=(0, Pong.HEIGHT // 4, Pong.WIDTH // 4), dtype=np.float32)
    actions = np.ndarray(shape=(0, 2, 2), dtype=np.float32)
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

def test_model(id):
    dqn = DQN(resume=False)
    dqn.load_model(f"./models/{id}.h5")
    env = Pong()
    #player.DeepQPlayer.EPSILON = 1
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
        env.show(2)
        env.show_state(2)
    l, r = env.get_score()
    print(f"Finished game with model {id}, {l} - {r}")


#dqn = DQN(resume=False)
#dqn.show_weights(0)

#dqn.show_weights(0)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run pong simulations and periodically retrain a DQN on the generated data")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--batch", type=int, help="number of simulations", default=5)
    parser.add_argument("--repeats", type=int, help="number of times to simulate and retrain", default=1000)
    parser.add_argument("--threads", type=int, help="number of threads to run. 0 to run sequentially, -1 for max", default=0)

    batch_size = 500
    repeats = 1000
    threads = -1
    args = parser.parse_args()

    if args.verbose:
        pass
    if args.batch:
        batch_size = args.batch
    if args.repeats:
        repeats = args.repeats
    if args.threads or args.threads == 0:
        threads = args.threads
    if threads == 0:
        print(f'Beginning {repeats} training loops of {batch_size} simulations without parallel processing.')
    elif threads > 0:
        print(f'Beginning {repeats} training loops of {batch_size} simulations, requesting {threads} threads.')
    else:
        print(f'Beginning {repeats} training loops of {batch_size} simulations on all available threads.')

    player.DeepQPlayer.EPSILON = 0
    for i in range(repeats):
        simulated_games = run_simulations(batch_size, threads=threads)
        s, a, r = simulated_games
        r = (r + 1) / 2
        dqn = DQN()
        dqn.retrain((s, a, r))
        #dqn.show_weights(0)
        dqn.save(f'{i}.h5')
        dqn.load_model(f'models/{i}.h5')
        player.DeepQPlayer.EPSILON = int(0.5 + (i / 200))

