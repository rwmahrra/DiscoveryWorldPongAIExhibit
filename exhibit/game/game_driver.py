from exhibit.game.game_subscriber import GameSubscriber
import numpy as np
from exhibit.game.player import BotPlayer, AIPlayer, HumanPlayer, CameraPlayer
import time

import pyrealsense2 as rs

from exhibit.shared.config import Config
from exhibit.game.pong import Pong

"""
This file is the driver for the game component.

It polls the agents for actions and advances frames and game state at a steady rate.
"""


class GameDriver:
    def run(self, level):

        env = Pong(level = level)
        currentFPS = (level * 40) + 40#Config.GAME_FPS 

        if type(self.left_agent) == BotPlayer: self.left_agent.attach_env(env)
        if type(self.right_agent) == BotPlayer: self.right_agent.attach_env(env)

        #wait until start
        if level == 1 :
            print("          Waiting for user interaction . . . ")
            while Pong.get_human_x() == 0.5 :
                time.sleep(0.01)
                #print(Pong.get_human_x())
            print(Pong.get_human_x())
            self.subscriber.emit_level(1)
            print("          Human detected, beginning game. ")
            
        
        
        # Housekeeping
        score_l = 0
        score_r = 0

        # Emit state over MQTT and keep a running timer to track the interval
        self.subscriber.emit_state(env.get_packet_info(), request_action=True)
        last_frame_time = time.time()

        # LW send depth camera feed over MQTT
        #self.subscriber.emit_depth_feed()

        # Track skipped frame statistics
        frame_skips = []

        i = 0
        done = False

        #env.change_speed(level)
        
        last_frame_time = time.time()
        while not done:
            acted_frame = self.subscriber.paddle2_frame
            rendered_frame = env.frames

            if acted_frame is not None:
                frames_behind = rendered_frame - acted_frame
                frame_skips.append(frames_behind)
            action_l, depth_l, prob_l = self.left_agent.act()

            for i in range(Config.AI_FRAME_INTERVAL):
                action_r, depth_r, prob_r = self.right_agent.act()
                if type(self.left_agent) == HumanPlayer or type(self.left_agent) == CameraPlayer:
                    action_l, depth_l, prob_l = self.left_agent.act()

                next_frame_time = last_frame_time + (1 / currentFPS)#Config.GAME_FPS)
                #print(f'Depth is {depth_l}')
                state, reward, done = env.step(Config.ACTIONS[action_l], Config.ACTIONS[action_r], frames=1, depth=depth_l)
                reward_l, reward_r = reward
                if reward_r < 0: score_l -= reward_r
                if reward_r > 0: score_r += reward_r
                if i == Config.AI_FRAME_INTERVAL - 1:
                    self.subscriber.emit_state(env.get_packet_info(), request_action=True)
                else:
                    self.subscriber.emit_state(env.get_packet_info(), request_action=False)
                #print(env.depth_feed)
                self.subscriber.emit_depth_feed(env.depth_feed)
                to_sleep = next_frame_time - time.time()
                if to_sleep < 0:
                    placeholder = 1
                    #print(f"Warning: render tick is lagging behind by {-int(to_sleep * 1000)} ms.")
                else:
                    time.sleep(to_sleep)

                last_frame_time = time.time()
            #self.subscriber.emit_depth_feed(env.depth_feed)

            last_frame_time = time.time()

            i += 1

        print('Score: %f - %f.' % (score_l, score_r))
        #if Config.DEBUG:
            #print(f"Behind frames: {np.mean(frame_skips)} mean, {np.std(frame_skips)} stdev, "
                  #f"{np.max(frame_skips)} max, {np.unique(frame_skips, return_counts=True)}")

    def __init__(self, subscriber, left_agent, right_agent, pipeline, decimation_filter, crop_percentage_w, crop_percentage_h):
        self.subscriber = subscriber
        self.left_agent = left_agent
        self.right_agent = right_agent
        self.pipeline = pipeline
        self.decimation_filter = decimation_filter
        self.crop_percentage_w = crop_percentage_w
        self.crop_percentage_h = crop_percentage_h


if __name__ == "__main__":

    print("from gameDriver, about to init GameSubscriber")
    subscriber = GameSubscriber()
    #opponent = BotPlayer(right=True)
    opponent = CameraPlayer()
    # Uncomment the following line (and comment the above) to control the left paddle
    #opponent = HumanPlayer('w', 's')
    agent = AIPlayer(subscriber, left=True)
    #agent = HumanPlayer('o', 'l')

    decimation_filter = rs.decimation_filter()
    decimation_filter.set_option(rs.option.filter_magnitude, 6)

    crop_percentage_w = 1.0
    crop_percentage_h = 1.0

    print("starting with crop w at {}".format(crop_percentage_w * 100))
    print("starting with crop h at {}".format(crop_percentage_h * 100))

    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    # filter out far away so we only have a person
    clipping_distance_in_meters = 1.8 #2 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    print(f'clipping distance is : {clipping_distance}');

    while True:
        level = 1
        
        # Wait for AI agent to spin up
        for level in range(1,4):
            if level == 1: 
                subscriber.emit_level(0)
            else:
                subscriber.emit_level(level) 
            time.sleep(5)
            start = time.time()
            instance = GameDriver(subscriber, opponent, agent, pipeline, decimation_filter, crop_percentage_w, crop_percentage_h)
            instance.run(level)

        print(f"Completed simulation in {time.time() - start}s")

