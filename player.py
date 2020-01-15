from pong import Pong
from random import randint
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from ai import DQN
import cv2
from random import random, choice


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
        else:
            return "UP" if randint(0, 1) == 1 else "DOWN"


class DeepQPlayer:
    EPSILON = 0

    def __init__(self, left=False, right=False, brain=None):
        # import necessary modules from keras
        self.left = left
        self.right = right
        self.new_memory = []
        self.game_memory = []
        if brain is None:
            self.brain = DQN()
        else:
            self.brain = brain

    def set_model(self, model):
        self.brain = model

    def move(self, state, debug=False):
        predictions = self.brain.infer(state)
        best = np.argmax(predictions)
        prob = predictions[best]
        if debug:
            print(predictions)
        if random() < DeepQPlayer.EPSILON:
            return Pong.ACTIONS[best], prob
        else:
            return choice(Pong.ACTIONS), 0.5
