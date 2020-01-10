import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from pong import Pong
import cv2
import os
from utils import encode_action


class DQN:
    def __init__(self, gamma=0.99, epsilon=0.5, resume=True):
        self.gamma = gamma
        # creates a generic neural network architecture
        self.model = Sequential()
        self.epsilon = epsilon
        self.get_last_file()
        print("Constructing DQN")

        # hidden layer takes a pre-processed frame as input, and has 200 units
        self.model.add(Dense(200, input_dim=(Pong.HEIGHT//4 * Pong.WIDTH//4), activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        self.model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform'))

        # compile the model using traditional Machine Learning losses and optimizers
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if resume:
            file = self.get_last_file()
            if file is not None:
                self.load_model(file)

    def get_last_file(self):
        files = [f for f in os.listdir("models") if os.path.isfile(os.path.join("models", f))]
        ids = [int(os.path.split(f)[1].split('.')[0]) for f in files]
        max = -1
        for id in ids:
            if id > max:
                max = id
        if max == -1:
            return None
        return os.path.join("models", f"{id}.h5")

    def load_model(self, path):
        print("Loading " + path)
        self.model.load_weights(path)

    def show_weights(self, neuron):
        weights = self.model.get_weights()[0][:, neuron]
        weights = weights.reshape(Pong.HEIGHT // 4, Pong.WIDTH // 4)

        weights = cv2.resize(weights, (Pong.WIDTH, Pong.HEIGHT)) + 0.5
        print(weights)
        cv2.imshow(f"DQN neuron weights {neuron}", weights)
        cv2.waitKey(0)

    def infer(self, state):
        state = state.flatten()
        state = np.expand_dims(state, axis=0)
        predictions = self.model.predict(state)[0]
        return predictions

    def retrain(self, games):
        states, actions, rewards = games
        rewards = self.discount_rewards(rewards[:, 1])
        states = np.stack([state.flatten().astype("float32") for state in states], axis=0)
        actions = actions[:, 1]

        self.model.fit(x=states, y=actions, sample_weight=rewards, epochs=20)

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def save(self, name):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        print('saving to ' + name)
        self.model.save('./models/' + name)
