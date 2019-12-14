from pong import Pong
from random import randint
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import cv2


class HumanPlayer:
    def __init__(self, up, down):
        self.up = up
        self.down = down

    def move(self, state):
        return Pong.read_key(self.up, self.down)


class RandomPlayer:
    def move(self, state):
        return Pong.random_action()


class BotPlayer:
    def __init__(self, game, left=False, right=False):
        self.left = left
        self.right = right
        self.paddle, self.ball = game.get_bot_data(left=left, right=right)

    def move(self, state):
        if self.left and not self.ball.right or self.right and self.ball.right:
            if self.ball.y > self.paddle.y:
                return "DOWN"
            elif self.ball.y < self.paddle.y:
                return "UP"
            else:
                return "UP" if randint(0, 1) == 1 else "DOWN"


class DeepQPlayer:

    def memorize(self, state, action, reward):
        self.new_memory.append((state, action, reward))

    def reverse_encode_action(self, action):
        if action == "UP":
            return np.asarray([1, 0, 0], dtype=np.float32)
        elif action == "DOWN":
            return np.asarray([0, 1, 0], dtype=np.float32)
        elif action == "NONE":
            return np.asarray([0, 0, 1], dtype=np.float32)
        else:
            print(action)

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r, dtype=np.float32)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def __init__(self, left=False, right=False, gamma = 0.99):
        # import necessary modules from keras
        self.left = left
        self.right = right
        self.new_memory = []
        self.game_memory = []
        self.gamma = gamma

        # creates a generic neural network architecture
        self.model = Sequential()

        # hidden layer takes a pre-processed frame as input, and has 200 units
        self.model.add(Dense(200, input_dim=(Pong.HEIGHT//4 * Pong.WIDTH//4), activation='relu', kernel_initializer='glorot_uniform'))

        # output layer
        self.model.add(Dense(3, activation='linear', kernel_initializer='glorot_uniform'))

        # compile the model using traditional Machine Learning losses and optimizers
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def move(self, state):
        state = state.flatten()
        state = np.expand_dims(state, axis=0)
        predictions = self.model.predict(state)[0]
        best = np.argmax(predictions)
        return Pong.ACTIONS[best]

    def retrain(self):
        self.memory = []
        states, actions, rewards = zip(*self.new_memory)
        rewards = self.discount_rewards(np.asarray(rewards, dtype=np.float32))
        actions = [self.reverse_encode_action(action) for action in actions]
        states = np.stack([state.flatten().astype("float32") for state in states], axis=0)
        actions = np.stack(actions)

        print(states.dtype, actions.dtype, rewards.dtype)
        self.model.fit(x=states, y=actions, sample_weight=rewards, epochs=10)

