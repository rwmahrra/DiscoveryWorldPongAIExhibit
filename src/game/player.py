from random import randint
from src.game.pong import Pong
import paho.mqtt.client as mqtt

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


    async def act(self, state):
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

    async def move(self, state):
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

    async def act(self, state):
        """
        Make decision based on state
        :param state: Unused, preserves interface
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
    Networked AI Agent
    """

    def on_message(self, client, userdata, msg):

    def on_connect(self, client, userdata, flags, rc):
        print("AI player connected")
        client.subscribe("$SYS/#")

    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.connect("localhost", 1883, 60)
        self.client.on_message = self.on_message

    async def act(self, state):
        """
        Send game details over network. Return response from agent.
        :param state: dictionary representing game state
        :return: (action id, confidence)
        """
        self.client.publish("test", payload="awaiting action")
        self.client.loop_start()
        self.client.loop_stop()
        return (1, 1)
        #return action, prob
