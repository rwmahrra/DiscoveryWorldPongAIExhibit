from output_visual import RealtimeVisualizer
from player import PGAgent, BotPlayer
import simulator

agent = PGAgent(simulator.ATARI_STATE_SIZE, simulator.ATARI_ACTION_SIZE)
#bot = BotPlayer()
visualizer = RealtimeVisualizer(agent.model)
simulator.simulate_game(simulator.ATARI, right=agent, visualizer=visualizer)
