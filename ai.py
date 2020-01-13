import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from pong import Pong
import cv2
import os
from utils import encode_action, discount_rewards

from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils as visutils
import utils
from keras import activations


class DQN:
    def __init__(self, gamma=0.99, epsilon=0.5, resume=True):
        self.gamma = gamma
        # creates a generic neural network architecture
        self.model = Sequential()
        self.epsilon = epsilon
        print("Constructing DQN")

        # hidden layer takes a pre-processed frame as input, and has 200 units
        self.model.add(Dense(200, input_dim=(Pong.HEIGHT//2 * Pong.WIDTH//2), activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        self.model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))

        # compile the model using traditional Machine Learning losses and optimizers
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if resume:
            file = utils.get_last_file()
            if file is not None:
                self.load_model(file)


    def load_model(self, path):
        try:
            print("Loading " + path)
            self.model.load_weights(path)
            print("Successfully loaded")
        except Exception as e:
            print("Error loading model")
            print(e)
            raise(e)

    def show_weights(self, neuron):
        weights = self.model.get_weights()[0][:, neuron]
        weights = weights.reshape(Pong.HEIGHT // 2, Pong.WIDTH // 2)

        weights = cv2.resize(weights, (Pong.WIDTH, Pong.HEIGHT)) + 0.5
        print(weights)
        cv2.imshow(f"DQN neuron weights {neuron}", weights)
        cv2.waitKey(0)

    def show_attention_map(self, frame):
        # Utility to search for layer index by name.
        # Alternatively we can specify this as -1 since it corresponds to the last layer.
        layer_idx = 1

        # Specific prediction
        class_idx = 2

        # Swap softmax with linear
        self.model.layers[layer_idx].activation = activations.linear
        self.model = visutils.apply_modifications(self.model)
        print(frame.shape)
        grads = visualize_saliency(self.model, layer_idx, filter_indices=[class_idx], seed_input=frame.flatten())
        print(grads)
        # Plot with 'jet' colormap to visualize as a heatmap.
        plt.imshow(grads, cmap='jet')

    def infer(self, state):
        state = state.flatten()
        state = np.expand_dims(state, axis=0)
        predictions = self.model.predict(state)[0]
        return predictions

    def retrain(self, games):
        states, actions, rewards = games
        rewards = discount_rewards(rewards[:, 1], gamma=self.gamma)
        states = np.stack([state.flatten().astype("float32") for state in states], axis=0)
        actions = actions[:, 1]
        print(np.unique(states[0]))
        self.model.fit(x=states, y=actions, sample_weight=rewards, epochs=20)

    def save(self, name):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        print('saving to ' + name)
        self.model.save('./models/' + name)
