from initial_experiments.ai import DQN
from random import random, choice
import numpy as np
from src.game.pong import Pong

class DeepQPlayer:
    """
    This class was an unnecessary layer of abstraction and never actually implemented a Deep Q model properly
    """
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
            return choice(Pong.ACTIONS), np.array([1/3, 1/3, 1/3])
        else:
            return Pong.ACTIONS[best], predictions