import sys
from exhibit.game.game_subscriber import GameSubscriber
import numpy as np
from exhibit.game.player import BotPlayer, AIPlayer, ControllerPlayer, MotionPlayer
import time

import pyrealsense2 as rs
import cv2

from exhibit.shared.config import Config
from exhibit.game.pong import Pong
import threading
import time
from queue import Queue
from exhibit.shared.utils import Timer

"""
This file is the driver for the game component.

It polls the agents for actions and advances frames and game state at a steady rate.
"""
class GameDriver:
    def run(self, level):
        print("running level: ", level)
        # The Pong environment
        pong_environment = Pong(config=self.config, level = level)
        currentFPS = self.config.GAME_FPS #(level * 40) + 40#Config.GAME_FPS 

        # if one of our players is Bot
        if type(self.bottom_agent) == BotPlayer: self.bottom_agent.attach_env(pong_environment)
        if type(self.top_agent) == BotPlayer: self.top_agent.attach_env(pong_environment)
        
        # score top and bottom
        score_l = 0
        score_r = 0

        # Emit state over MQTT and keep a running timer to track the interval
        self.subscriber.emit_state(pong_environment.get_packet_info(), request_action=True)
        last_frame_time = time.time()

        # Track skipped frame statistics
        frame_skips = []

        i = 0
        done = False
        last_frame_time = time.time()
        while not done:
            action_l, motion_l, prob_l = self.bottom_agent.act()
            for i in range(self.config.AI_FRAME_INTERVAL):
                rendered_frame = pong_environment.frames
                #Timer.start("act")
                action_r, depth_r, prob_r = self.top_agent.act()
                acted_frame = self.top_agent.get_frame()
                if self.config.MOVE_TIMESTAMPS:
                    print(f'{time.time_ns() // 1_000_000} F{pong_environment.frames} MOVE W/PRED {self.top_agent.get_frame()}')
                if acted_frame is not None:
                    frames_behind = rendered_frame - acted_frame
                    if frames_behind >= 0 and frames_behind:
                        # Throw out frame ids from previous games
                        if frames_behind <= self.config.AI_FRAME_INTERVAL:
                            frame_skips.append(0)  # Frame diffs of 0-5 frames are always intentional
                        else:
                            frame_skips.append(frames_behind - self.config.AI_FRAME_INTERVAL)

                if type(self.bottom_agent) == ControllerPlayer or type(self.bottom_agent) == MotionPlayer:
                    action_l, motion_l, prob_l = self.bottom_agent.act()

                #Timer.stop("act")

                next_frame_time = last_frame_time + (1 / currentFPS)

                #Timer.start("step")
                #pong.step(self, bottom_action, top_action, frames=3, motion_position=None):
                state, reward, done = pong_environment.step(self.config.ACTIONS[action_l], self.config.ACTIONS[action_r], frames=1, motion_position=motion_l)
                #Timer.stop("step")
                reward_l, reward_r = reward
                if reward_r < 0: score_l -= reward_r
                if reward_r > 0: score_r += reward_r
                #Timer.start("emit")
                if i == self.config.AI_FRAME_INTERVAL - 1:
                    self.subscriber.emit_state(pong_environment.get_packet_info(), request_action=True)
                else:
                    self.subscriber.emit_state(pong_environment.get_packet_info(), request_action=False)
                # self.subscriber.emit_depth_feed(pong_environment.depth_feed)
                #Timer.stop("emit")
                to_sleep = next_frame_time - time.time()
                if to_sleep < 0:
                    pass
                    #print(f"Warning: render tick is lagging behind by {-int(to_sleep * 1000)} ms.")
                else:
                    time.sleep(to_sleep)

                last_frame_time = time.time()

            last_frame_time = time.time()

            i += 1

        print('Score: %f - %f.' % (score_l, score_r))
        if self.config.BEHIND_FRAMES:
            print(frame_skips)
            try:
                print(f"Behind frames: {np.mean(frame_skips)} mean, {np.std(frame_skips)} stdev, "
                    f"{np.max(frame_skips)} max, {np.unique(frame_skips, return_counts=True)}")
            except Exception as excep:
                print(excep)

    def __init__(self, config, subscriber, bottom_agent, top_agent): # pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance):
        self.subscriber = subscriber
        self.bottom_agent = bottom_agent
        self.top_agent = top_agent
        # self.pipeline = pipeline
        # self.decimation_filter = decimation_filter
        # self.crop_percentage_w = crop_percentage_w
        # self.crop_percentage_h = crop_percentage_h
        # self.clipping_distance = clipping_distance
        self.config = config
        self.subscriber = subscriber



def main(in_q, config=Config.instance()):
    subscriber = GameSubscriber()
    print(f'The current MAX_SCORE is set to {config.MAX_SCORE}')

    agent = AIPlayer(subscriber, top=True)

    if config.USE_DEPTH_CAMERA:
        print("Configured to use depth Camera")
        opponent = MotionPlayer(subscriber)
    else:
        opponent = ControllerPlayer('a', 'd')
        print("setting opponent to be a ControllerPlayer")


    print("Level is set to 0 and game driver is starting fresh")
    level = 0
    # start at level 0, our start state when nobody is playing
    time.sleep(1) # wait a second for connection stuff to happen. It was previoiusly connecting only after it tried to emit
    subscriber.emit_level(level) 

    game_instance = GameDriver(
        config, subscriber, opponent, agent)

    while True: # play the exhibit on loop forever
        print("Starting game management loop")
        start = time.time()
        if not in_q.empty(): # if there's something in the queue to read
            dataQ = in_q.get() # remember that .get() removes the item from the queue
            if dataQ == "endThreads":
                print('thread quitting')
                subscriber.client.disconnect()
                print(f'in_q is {dataQ} and in_q.empty() is {in_q.empty()}')
                while not in_q.empty: # clear out the q
                    dataQ = in_q.get()
                cv2.destroyAllWindows() # close the Pong game window
                in_q.put('noneActive') # a message back to GUI/main thread to signal that game_driver is ended
                sys.exit() # kill the thread
                print('the sys exit didnt work')
                break
            else:
                print(f'in_q is {dataQ}')

        # set the game to a waiting state - enables the detection of a player entering motion area and any idle animations that we may want to implement
        # this should always be the case before starting any level - idle animations only when level == 0 and game state is waiting
        subscriber.emit_game_state(0) # 0 waiting, 1 ready, 2 running   
        # wait to verify someone is playing     
        while not subscriber.motion_presence:
            time.sleep(0.1)
        print("motion detected, beginning game")
        print(f'level {level}')

        if level < 3: # if the game hasn't started yet and the next level would be level 1
            print("proceed to next level")
            level = level + 1
            
            subscriber.emit_level(level) 
            subscriber.emit_game_state(1) # ready
            subscriber.reset_game_state()
            time.sleep(3) # delay for start here - gives a chance for components to give feedback
            subscriber.emit_game_state(2) # running
            time.sleep(1)
            game_instance.run(level) # RUN LEVEL (1)

        else: # level == 3
            level = 0 # the game is over so we need to reset to level 0 (the state before the game starts
            print(f'            Game reset to level {level} (zero).')
            subscriber.emit_level(level)
            subscriber.emit_game_state(3)
            time.sleep(6) # wait 1 second so person has time to leave and next person can come in

    sys.exit()


if __name__ == "__main__":
    main(Queue(), config=Config())



