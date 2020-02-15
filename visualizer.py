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
from keras.models import Model
from matplotlib import pyplot
import subprocess
import matplotlib.pyplot as plt
import csv


def view_train_progression(number, neuron=0, interval=50):
    dqns = []
    for i in range(0, number + 1, 50):
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


def view_weights(id, layer=0):
    dqn = DQN(resume=False)
    dqn.load_model(f"./models/{id}.h5")
    for i in range(200):
        dqn.show_weights(i, layer=layer)


def get_weight_image(model, neuron=0, layer=0, size=(Pong.HEIGHT // 2, Pong.WIDTH // 2)):
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


def visualize_conv_features():
    # load the model
    model = load_model()
    # redefine model to output right after the first hidden layer
    model = Model(inputs=model.inputs, outputs=model.layers[1].output)
    model.summary()
    # load the image with the required shape
    img = np.load("./example_down.npy")
    img = img.flatten()
    # expand dimensions so that it represents a single 'sample'
    img = np.expand_dims(img, axis=0)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot all 64 maps in an 8x8 squares
    square = 5
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            print(feature_maps[0, :, :, ix - 1].shape)
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1].reshape(9, 3), cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
    cv2.imshow("test image", img.reshape(Pong.HEIGHT // 2, Pong.WIDTH // 2))
    cv2.waitKey(0)


def visualize_conv_filters():
    model = load_model()
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[1].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[:, :, j], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()


#visualize_conv_filters()
#visualize_conv_features()
#test_model(50)
#view_train_progression(4850, neuron=199, interval=50)
#view_weights(1300, 0)
#debug_step()
#plot_loss()
