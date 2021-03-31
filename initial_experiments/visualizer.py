from src.game.pong import Pong
import cv2
import numpy as np
from src.shared.utils import discount_rewards
from initial_experiments.ai import DQN
from initial_experiments.main import run_simulations
from keras.models import Model
from matplotlib import pyplot

def visualize_conv_features():
    """
    This was used when testing convolutional layers.
    That idea was scrapped to keep a simple, easy-to-visualize exhibit
    (although visualizing convolutions could be really cool...)
    """
    # load the model
    model = None #load_model() # This call is broken. Commenting it out to prevent parsing errors
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


def view_train_progression(number, layer=0, interval=50):
    """
    This method views weight maps as a sequence from stored model snapshots captured during training.
    It provided a useful visualization of which pixels were receiving more (or less) focus
    :param number: Maximum training iteration to view
    :param interval: Step size between training snapshots (view every ith weight map)
    :return:
    """
    dqns = []
    for i in range(0, number + 1, interval):
        dqn = DQN(resume=False)
        dqn.load_model(f"./models/{i}.h5")
        dqns.append(dqn)
    for dqn in dqns:
        dqn.show_weights(layer)


def visualize_game_memory():
    """
    This method simulates a game and then reconstructs and
    visualizes it from the generated numpy arrays.
    This is a useful sanity check to manually validate the simulator and data piping.
    """
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


def view_weights(id, layer=0):
    """
    This function was used for manually inspecting neuron weights from an h5 file.
    This is helpful when debugging training.
    :param id: Model ID to load (file name in /models folder w/o .h5 extension)
    :param layer: Layer index to visualize (0 = weights connecting first hidden layer to input)
    """
    dqn = DQN(resume=False)
    dqn.load_model(f"./models/{id}.h5")
    for i in range(200):
        dqn.show_weights(i, layer=layer)


def visualize_conv_filters():
    """
    This was used when testing convolutional layers.
    That idea was scrapped to keep a simple, easy-to-visualize exhibit
    (although visualizing convolutions could be really cool...)
    :return:
    """
    model = None #load_model() # This call is broken. Commenting it out to prevent parsing errors
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
