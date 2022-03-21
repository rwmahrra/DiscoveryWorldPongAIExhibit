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


    def act(self, state=None):
        """
        Listen to key
        :param state: Unused, preserves interface
        :return: (action id, confidence)
        """
        #print("HumanPlayer acting")
        return self.move(), None, 1

    def move(self):
        return Pong.read_key(self.up, self.down)

class CameraPlayer:

    def act(self, state=None):
        """
        Player controlled via the depth camera
        """
        print("CameraPlayer acting")
        return self.move(), Pong.get_human_x(), 1 # Pong.get_depth(), 1 #

    def move(self):
        return 3 # the code for moving based on depth value, which is different than fixed speed movement

class RandomPlayer:
    """
    Mostly for testing. Makes purely random actions.
    """

    def act(self):
        return randint(0, 2)
        #return Pong.random_action()


class BotPlayer:
    """
    Opponent to train against. Hard-coded to calculate action
    based off of ball and paddle position.
    """

    def __init__(self, env=None, bottom=False, top=False, always_follow=False):
        """
        Set state
        :param env: Pong environment
        :param bottom: true if operating bottom paddle
        :param top: true of operating top paddle
        :param always_follow: if true, always tries to sit at ball position.
               if false, only follows ball when the ball is moving towards this paddle's side of the screen
        """
        self.bottom = bottom
        self.top = top
        self.always_follow = always_follow
        if self.top:
            self.last_move = 1
        else:
            self.last_move = 0
        if env is not None:
            self.paddle, self.ball = env.get_bot_data(bottom=bottom, top=top)

    def attach_env(self, env):
        """
        Save objects from pong environment
        :param env: Pong environment
        :return:
        """
        self.paddle, self.ball = env.get_bot_data(bottom=self.bottom, top=self.top)

    def act(self, state=None):
        """
        Make decision based on state
        :param state: Unused, preserves interface for train script
        :return: (action id, confidence)
        """
        #print("BotPlayer acting")
        return self.move(), None, 1

    def move(self):
        if self.always_follow:
            if self.ball.x > self.paddle.x:
                return 1
            elif self.ball.x < self.paddle.x:
                return 0
            else:
                self.last_move = abs(self.last_move - 1)
                return self.last_move
        if self.bottom and not self.ball.up or self.top and self.ball.up:
            if self.ball.x > self.paddle.x:
                return 1
            elif self.ball.x < self.paddle.x:
                return 0
            else:
                self.last_move = abs(self.last_move - 1)
                return self.last_move
        else:
            self.last_move = abs(self.last_move - 1)
            return self.last_move


class AIPlayer:
    """
    Abstraction for MQTT networked AI Agent
    """
    def __init__(self, subscriber, top=False, bottom=False):
        self.top = top
        self.bottom = bottom
        if not self.top and not self.bottom:
            raise ValueError("AI paddle must be specified as left or right with the cooresponding keyword argument")
        self.subscriber = subscriber

    def act(self, current_frame=None):
        """
        Send game details over network. Return response from agent.
        :param state: dictionary representing game state
        :return: (action id, confidence)
        """
        if self.top:
            return self.subscriber.get_paddle1_action(current_frame), None, self.subscriber.paddle1_prob
        if self.bottom:
            return self.subscriber.paddle2_action, None, self.subscriber.paddle2_prob

    def get_frame(self):
        if self.top:
            return self.subscriber.paddle1_frame
        if self.bottom:
            return self.subscriber.paddle2_frame
