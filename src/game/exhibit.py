from src.visualization.output_visual import RealtimeVisualizer
from src.game.player import HumanPlayer, AIPlayer, BotPlayer
from src.game.game_subscriber import GameSubscriber
from src.game import simulator
import asyncio

"""
This file is the driver for the exhibit proof of concept.

It simply loads a pre-trained model pits it against a hardcoded bot, then hands the game off to the
visualization found in visualizer.py
"""

subscriber = GameSubscriber()
opponent = BotPlayer(left=True)
#opponent = HumanPlayer('w', 's')
agent = AIPlayer(subscriber, right=True)
simulator.simulate_game(simulator.CUSTOM, left=opponent, right=agent, subscriber=subscriber)
