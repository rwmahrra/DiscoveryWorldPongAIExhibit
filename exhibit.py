from output_visual import RealtimeVisualizer
from player import PGAgent, BotPlayer, HumanPlayer
import simulator

"""
This file is the driver for the exhibit proof of concept.

It simply loads a pre-trained model pits it against a hardcoded bot, then hands the game off to the
visualization found in visualizer.py
"""

agent = PGAgent(simulator.CUSTOM_STATE_SIZE, simulator.CUSTOM_ACTION_SIZE)
agent.load('./validation/6px_7k.h5')
opponent = BotPlayer(left=True)
#opponent = HumanPlayer('w', 's')
visualizer = RealtimeVisualizer(agent.model)
simulator.simulate_game(simulator.CUSTOM, left=opponent, right=agent, visualizer=visualizer)
