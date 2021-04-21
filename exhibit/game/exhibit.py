from exhibit.game.player import HumanPlayer, AIPlayer, BotPlayer
from exhibit.game.game_subscriber import GameSubscriber
from exhibit.game import simulator
import time

"""
This file is the driver for the exhibit proof of concept.

It simply loads a pre-trained model pits it against a hardcoded bot, then hands the game off to the
visualization found in visualizer.py
"""

subscriber = GameSubscriber()
opponent = BotPlayer(left=True)
#opponent = HumanPlayer('w', 's')
agent = AIPlayer(subscriber, right=True)
time.sleep(5)
start = time.time()
simulator.simulate_game(simulator.CUSTOM, left=opponent, right=agent, subscriber=subscriber)
print(f"Completed simulation in {time.time() - start}s")