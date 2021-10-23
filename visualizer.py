from exhibit.game.pong import Pong
import numpy as np
from exhibit.game import player
from exhibit.shared.config import Config
from exhibit.shared.utils import encode_action
from initial_experiments.ai import DQN
import matplotlib.pyplot as plt
import csv

"""
This file aggregates visualization helper methods I found useful for debugging and generating figures for the paper.
"""


def test_model(id):
    dqn = DQN(resume=False)
    dqn.load_model(f"./models/{id}.h5")
    env = Pong()
    player.DeepQPlayer.EPSILON = 0
    right = player.DeepQPlayer(right=True)
    right.set_model(dqn)
    left = player.BotPlayer(env, left=True)
    last_screen = None
    state = env.reset()
    done = False
    while not done:
        left_action = left.move(state)
        right_action, prob = right.move(state, debug=True)
        print(right_action)
        state, reward, done = env.step(left_action, right_action)
        #dqn.show_attention_map(state)
        env.show(2)
        env.show_state(2, 0)
    l, r = env.get_score()
    print(f"Finished game with model {id}, {l} - {r}")


def plot_scores(path='./tmp/', show=False, average_of=100):
    file_contents = {}
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        with open(path + file, 'r') as csvfile:
            plots = list(csv.reader(csvfile, delimiter=','))
            i = 0
            x = []
            yr = []
            print(path + file)
            for row in plots:
                if i > average_of:
                    x.append(i)
                    last_n = sum([float(plots[j][1]) for j in range(i-average_of+1, i+1)])
                    yr.append(last_n / average_of)
                i += 1
            print("finished rows")
            file_contents[str(file)] = (x, yr)
        print("last plot")
        plt.plot(x, yr, label=str(file).replace('.csv', '').replace('_', '-'))
        print("plotted")
    print("plot")
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Agent Score By Hidden Layer Structure')
    plt.legend(loc='lower right')
    if path:
        plt.savefig(f'{path}/out.png')
    plt.cla()


def get_weight_image(model, neuron=0, layer=0, size=(Config.instance().HEIGHT // 2, Config.instance().WIDTH // 2)):
    weights = model.get_weights()[layer][:, neuron]
    # Normalize and scale weights to pixel values
    weights /= np.max(weights)
    weights += 1
    weights *= 256
    image = weights.reshape(size).astype(np.uint8)
    return image


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

#
# The commented out lines below can be useful for running this
# file directly to get ad-hoc visualizations from stored data.
#
# plot_scores()
# visualize_conv_filters()
# visualize_conv_features()
# test_model(50)
# view_train_progression(4850, neuron=199, interval=50)
# view_weights(1300, 0)
# debug_step()
# plot_loss()
