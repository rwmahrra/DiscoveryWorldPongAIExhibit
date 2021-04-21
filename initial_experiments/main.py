from exhibit.game.pong import Pong
import numpy as np
from exhibit.game import player
from exhibit.shared.utils import Timer, get_resume_index, encode_gradient, encode_action
import multiprocessing
from initial_experiments.ai import DQN
from keras import backend as K

# Optionally disable GPU training to work around parallelization crashes
DISABLE_GPU = False
if DISABLE_GPU:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def simulate_pong(task_id, show=True, show_debug=False):
    """
    Simulates a game of Pong
    NOTE: If used to train, this is coupled to the agent class (DeepQPlayer in the example below)
          locating and loading the latest available model state on class initialization.

    :param task_id:
    :param show: Render a human-suitable game view
    :param show_debug: Render a debug view, showing the preprocessed state to be fed to a DRL agent
    :return: tuple of three ndarrays representing the simulated game's (states, actions, rewards)
    """
    try:
        states = np.ndarray(shape=(0, Pong.HEIGHT//2, Pong.WIDTH//2), dtype=np.float32)
        actions = np.ndarray(shape=(0, 2, 2), dtype=np.float32)
        rewards = np.ndarray(shape=(0, 2), dtype=np.float32)
        env = Pong()
        right = player.DeepQPlayer(right=True)  #player.HumanPlayer('up', 'down') # <-- for manual control
        left = player.BotPlayer(env, left=True)
        state = env.reset()
        done = False
        while not done:
            left_action = left.move(state)
            right_action, prob = right.move(state)
            action = np.stack((encode_action(left_action), encode_gradient(right_action, prob)))
            start_state = state
            state, reward, done = env.step(left_action, right_action)
            reward_l, reward_r = reward
            states = np.concatenate((states, np.expand_dims(start_state, axis=0)), axis=0)
            actions = np.concatenate((actions, np.expand_dims(action, axis=0)), axis=0)
            rewards = np.concatenate((rewards, np.asarray([[reward_l, reward_r]], dtype=np.float32)), axis=0)
            if show:
                env.show(2)
            if show_debug:
                print(right_action)
                env.show_state(2)
        l, r = env.get_score()
        print(f"Finished game {task_id}, {l} - {r}")
        return states, actions, rewards
    except Exception as e:
        print(e)


def run_simulations(n, threads=0):
    """
    Simulates an arbitrary number of pong games to record training data.

    :param n: number of simulations to run
    :param threads: threads to use (runs synchronously if set to 0)
    :return: state, action, and reward ndarrays from all simulated games
    """
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

    states = np.ndarray(shape=(0, Pong.HEIGHT // 2, Pong.WIDTH // 2), dtype=np.float32)
    actions = np.ndarray(shape=(0, 2, 2), dtype=np.float32)
    rewards = np.ndarray(shape=(0, 2), dtype=np.float32)

    for game in games:
        game_states, game_actions, game_rewards = game
        states = np.concatenate((game_states, states), axis=0)
        actions = np.concatenate((game_actions, actions), axis=0)
        rewards = np.concatenate((game_rewards, rewards), axis=0)
    print(f"Lasted average {states.shape[0] / n} frames")
    return states, actions, rewards


def test_nnet(model):
    """
    Quick test script to sanity check that
    a model is making reasonable predictions
    :param model: A model to validate (currently only DQN in ai.py)
    :return:
    """
    s = np.load('./test_data/states.npy')
    for i in range(20):
        predictions = model.infer(s[i])
        for j in predictions:
            print(j)


"""
NOTE:
This was used in my first few approaches to this problem.
See main.py to see the code that drove later experiments.
"""
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run pong simulations and periodically retrain a DQN on the generated data")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--batch", type=int, help="number of simulations", default=1)
    parser.add_argument("--repeats", type=int, help="number of times to simulate and retrain", default=10000)
    parser.add_argument("--threads", type=int, help="number of threads to run. 0 to run sequentially, -1 for max", default=0)
    parser.add_argument("--epochs", type=int, help="number of epochs to train after simulation batches", default=1)
    parser.add_argument("--epsilon_min", type=int, help="minimum greed", default=0.1)
    parser.add_argument("--epsilon_decay", type=int, help="rate at which greed sustains (1 for no decay)", default=0.999)
    parser.add_argument("--save_interval", type=int, help="number of train cycles to run before saving", default=50)

    batch_size = 500
    repeats = 1000
    threads = -1
    epochs = 20
    save_interval = 50
    epsilon_decay = 0.999
    epsilon_min = 0.1
    args = parser.parse_args()

    if args.verbose:
        pass
    if args.batch:
        batch_size = args.batch
    if args.repeats:
        repeats = args.repeats
    if args.threads or args.threads == 0:
        threads = args.threads
    if args.epochs:
        epochs = args.epochs
    if args.save_interval:
        save_interval = args.save_interval
    if args.epsilon_decay:
        epsilon_decay = args.epsilon_decay
    if args.epsilon_min:
        epsilon_min = args.epsilon_min
    if threads == 0:
        print(f'Beginning {repeats} training loops of {batch_size} simulations without parallel processing.')
    elif threads > 0:
        print(f'Beginning {repeats} training loops of {batch_size} simulations, requesting {threads} threads.')
    else:
        print(f'Beginning {repeats} training loops of {batch_size} simulations on all available threads.')
    player.DeepQPlayer.EPSILON = 1
    begin_index = get_resume_index()
    if begin_index is None:
        begin_index = 0
    else:
        begin_index += 1

    for i in range(repeats):
        i += begin_index
        simulated_games = run_simulations(batch_size, threads=threads)
        s, a, r = simulated_games
        dqn = DQN()
        dqn.retrain((s, a, r), epochs=epochs)
        #dqn.show_weights(0)
        dqn.save('temp.h5')
        if i % save_interval == 0:
            dqn.save(f'{i}.h5')
        K.clear_session()
        dqn = DQN()
        player.DeepQPlayer.EPSILON *= epsilon_decay
        player.DeepQPlayer.EPSILON = max(player.DeepQPlayer.EPSILON, epsilon_min)

