from output_visual import RealtimeVisualizer
from player import PGAgent, BotPlayer
import simulator

agent = PGAgent(simulator.CUSTOM_STATE_SIZE, simulator.CUSTOM_ACTION_SIZE)
agent.load('./validation/6px_7k.h5')
bot = BotPlayer(left=True)
visualizer = RealtimeVisualizer(agent.model)
simulator.simulate_game(simulator.CUSTOM, left=bot, right=agent, visualizer=visualizer)
