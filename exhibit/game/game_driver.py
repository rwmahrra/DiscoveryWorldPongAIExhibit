from exhibit.game.game_subscriber import GameSubscriber
import numpy as np
from exhibit.game.player import BotPlayer, AIPlayer, HumanPlayer
import time

from exhibit.shared.config import Config
from exhibit.game.pong import Pong

"""
This file is the driver for the game component.

It polls the agents for actions and advances frames and game state at a steady rate.
"""


class GameDriver:
    def run(self):

        env = Pong()

        if type(self.left_agent) == BotPlayer: self.left_agent.attach_env(env)
        if type(self.right_agent) == BotPlayer: self.right_agent.attach_env(env)

        # Housekeeping
        score_l = 0
        score_r = 0

        # Emit state over MQTT and keep a running timer to track the interval
        self.subscriber.emit_state(env.get_packet_info(), request_action=True)
        last_frame_time = time.time()

        # Track skipped frame statistics
        frame_skips = []

        i = 0
        done = False
        last_frame_time = time.time()
        while not done:
            acted_frame = self.subscriber.paddle2_frame
            rendered_frame = env.frames

            if acted_frame is not None:
                frames_behind = rendered_frame - acted_frame
                frame_skips.append(frames_behind)
            action_l, prob_l = self.left_agent.act()

            for i in range(Config.AI_FRAME_INTERVAL):
                action_r, prob_r = self.right_agent.act()
                if type(self.left_agent) == HumanPlayer:
                    action_l, prob_l = self.left_agent.act()

                next_frame_time = last_frame_time + (1 / Config.GAME_FPS)
                state, reward, done = env.step(Config.ACTIONS[action_l], Config.ACTIONS[action_r], frames=1)
                reward_l, reward_r = reward
                if reward_r < 0: score_l -= reward_r
                if reward_r > 0: score_r += reward_r
                if i == Config.AI_FRAME_INTERVAL - 1:
                    self.subscriber.emit_state(env.get_packet_info(), request_action=True)
                to_sleep = next_frame_time - time.time()
                if to_sleep < 0:
                    print(f"Warning: render tick is lagging behind by {-int(to_sleep * 1000)} ms.")
                else:
                    time.sleep(to_sleep)

                last_frame_time = time.time()

            last_frame_time = time.time()

            i += 1

        print('Score: %f - %f.' % (score_l, score_r))
        if Config.DEBUG:
            print(f"Behind frames: {np.mean(frame_skips)} mean, {np.std(frame_skips)} stdev, "
                  f"{np.max(frame_skips)} max, {np.unique(frame_skips, return_counts=True)}")

    def __init__(self, subscriber, left_agent, right_agent):
        self.subscriber = subscriber
        self.left_agent = left_agent
        self.right_agent = right_agent


if __name__ == "__main__":
    subscriber = GameSubscriber()

    opponent = HumanPlayer('a', 'd')
    agent = AIPlayer(subscriber, right=True)

    # Uncomment the following line (and comment the above) to control the left paddle
    #opponent = HumanPlayer('w', 's')
    agent = AIPlayer(subscriber, left=True)

    # Wait for AI agent to spin up
    for level in range(1,4):
        subscriber.emit_level(3)
        time.sleep(5)
        start = time.time()
        instance = GameDriver(subscriber, opponent, agent)
        instance.run()
    print(f"Completed simulation in {time.time() - start}s")

