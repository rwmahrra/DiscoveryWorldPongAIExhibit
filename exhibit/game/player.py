from random import randint
from exhibit.game.pong import Pong
from exhibit.shared.config import Config

"""
NOTE: the classes defined in this file are intended to implement a common interface:

- Accepting an arbitrary ndarray state (with dimensions configured at initialization)
def act(self, state):
    - Should return (action id, confidence)
    - where action id = 0 (up), 1 (down), or 2 (do nothing)
    - and confidence is a double between 0 and 1 indicating % confidence in the predicted action
    - Agents that don't generate an actual confidence can leave this set to 1
    
This interface isn't actually used by the pong game itself, but when working with multiple agents and 
running experiments that involve swapping out agents, it's a lot nicer to follow this pattern to minimize
code changes.
"""


class HumanPlayer:
    """
    Listens to keyboard input to drive a paddle
    """

    def __init__(self, up, down):
        """
        Configure keybinds
        :param up: key code for up control
        :param down: key code for down control
        """
        self.up = up
        self.down = down


    def act(self):
        """
        Listen to key
        :param state: Unused, preserves interface
        :return: (action id, confidence)
        """
        return self.move(), 1

    def move(self):
        return Pong.read_key(self.up, self.down)


class RandomPlayer:
    """
    Mostly for testing. Makes purely random actions.
    """

    def move(self):
        return Pong.random_action()


class BotPlayer:
    """
    Opponent to train against. Hard-coded to calculate action
    based off of ball and paddle position.
    """

    def __init__(self, env=None, left=False, right=False, always_follow=False):
        """
        Set state
        :param env: Pong environment
        :param left: true if operating left paddle
        :param right: true of operating right paddle
        :param always_follow: if true, always tries to sit at ball position.
               if false, only follows ball when the ball is moving towards this paddle's side of the screen
        """
        self.left = left
        self.right = right
        self.always_follow = always_follow
        if env is not None:
            self.paddle, self.ball = env.get_bot_data(left=left, right=right)

    def attach_env(self, env):
        """
        Save objects from pong environment
        :param env: Pong environment
        :return:
        """
        self.paddle, self.ball = env.get_bot_data(left=self.left, right=self.right)

    def act(self, state=None):
        """
        Make decision based on state
        :param state: Unused, preserves interface for train script
        :return: (action id, confidence)
        """
        return self.move(), 1

    def move(self):
        if self.always_follow:
            if self.ball.y > self.paddle.y:
                return 1
            elif self.ball.y < self.paddle.y:
                return 0
            else:
                return 0 if randint(0, 1) == 1 else 1
        if self.left and not self.ball.right or self.right and self.ball.right:
            if self.ball.y > self.paddle.y:
                return 1
            elif self.ball.y < self.paddle.y:
                return 0
            else:
                return 0 if randint(0, 1) == 1 else 1
        else:
            return 0 if randint(0, 1) == 1 else 1


class AIPlayer:
    """
    Abstraction for MQTT networked AI Agent
    """
    def __init__(self, subscriber, left=False, right=False):
        self.left = left
        self.right = right
        if not self.right and not self.left:
            raise ValueError("AI paddle must be specified as left or right with the cooresponding keyword argument")
        self.subscriber = subscriber

    def act(self):
        """
        Send game details over network. Return response from agent.
        :param state: dictionary representing game state
        :return: (action id, confidence)
        """
        if self.left:
            return self.subscriber.paddle1_action, self.subscriber.paddle1_prob
        if self.right:
            return self.subscriber.paddle2_action, self.subscriber.paddle2_prob
