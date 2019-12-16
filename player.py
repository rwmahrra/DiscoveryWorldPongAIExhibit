from pong import Pong
from random import randint
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from ai import DQN
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
        else:
            return "NONE"


class DeepQPlayer:

    def __init__(self, brain=DQN(), left=False, right=False):
        # import necessary modules from keras
        self.left = left
        self.right = right
        self.new_memory = []
        self.game_memory = []
        self.brain = brain

    def move(self, state):
        predictions = self.brain.infer(state)
        best = np.argmax(predictions)
        return Pong.ACTIONS[best]
