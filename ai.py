import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from pong import Pong
from utils import encode_action


class DQN:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        # creates a generic neural network architecture
        self.model = Sequential()

        # hidden layer takes a pre-processed frame as input, and has 200 units
        self.model.add(Dense(200, input_dim=(Pong.HEIGHT//4 * Pong.WIDTH//4), activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        self.model.add(Dense(3, activation='linear', kernel_initializer='glorot_uniform'))

        # compile the model using traditional Machine Learning losses and optimizers
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def load_model(self):
        pass

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

        self.model.fit(x=states, y=actions, sample_weight=rewards, epochs=80)
        self.model.save("pong_dqn_100000.h5")

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
