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

        env = Pong(level = level, pipeline = self.pipeline, decimation_filter = self.decimation_filter, crop_percentage_w = self.crop_percentage_w, crop_percentage_h = self.crop_percentage_h, clipping_distance = self.clipping_distance)
        currentFPS = (level * 40) + 40#Config.GAME_FPS 

        if type(self.left_agent) == BotPlayer: self.left_agent.attach_env(env)
        if type(self.right_agent) == BotPlayer: self.right_agent.attach_env(env)

            
        
        
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

    def __init__(self, subscriber, left_agent, right_agent, pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance):
        self.subscriber = subscriber
        self.left_agent = left_agent
        self.right_agent = right_agent
        self.pipeline = pipeline
        self.decimation_filter = decimation_filter
        self.crop_percentage_w = crop_percentage_w
        self.crop_percentage_h = crop_percentage_h
        self.clipping_distance = clipping_distance

# checks if theres a big enough player sized blob
def check_for_player(pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance):
        #try to get the frame 50 times
        for i in range(50): 
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            #color = frames.get_color_frame()
            if not depth: continue

            # filtering the image to make it less noisy and inconsistent
            depth_filtered = decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())
            
            # cropping the image based on a width and height percentage
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * crop_percentage_w)/2), int(w/2 + (w * crop_percentage_w)/2)
            hs, he = int(h/2 - (h * crop_percentage_h)/2), int(h/2 + (h * crop_percentage_h)/2)
            #print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
            depth_cropped = depth_image[ws:we, hs:he]
            #depth_cropped = depth_image

            cutoffImage = np.where((depth_cropped < clipping_distance) & (depth_cropped > 0.1), True, False)

            #print(f'cutoffImage shape is {cutoffImage.shape}, depth_cropped shape is {depth_cropped.shape}');
            avg_x = 0
            avg_x_array = np.array([])
            countB = 0
            for a in range(np.size(cutoffImage,0)):
                for b in range(np.size(cutoffImage,1)):
                    if cutoffImage[a,b] :
                        avg_x += b
                        #print(b)
                        avg_x_array = np.append(avg_x_array,b)
                        countB = countB+1
            # if we got no pixels in depth, return false
            if countB <= 40: 
                return False

            return True # successfully found a player, return true
        return False # failed to get camera image, return false

# checks if there is a player and returns their position from 0 to 1 so that we can tell if theyre walking through or still to play
def check_for_still_player(pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance):
        #try to get the frame 50 times
        for i in range(50): 
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            #color = frames.get_color_frame()
            if not depth: continue

            # filtering the image to make it less noisy and inconsistent
            depth_filtered = decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())
            
            # cropping the image based on a width and height percentage
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * crop_percentage_w)/2), int(w/2 + (w * crop_percentage_w)/2)
            hs, he = int(h/2 - (h * crop_percentage_h)/2), int(h/2 + (h * crop_percentage_h)/2)
            #print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
            depth_cropped = depth_image[ws:we, hs:he]
            #depth_cropped = depth_image

            cutoffImage = np.where((depth_cropped < clipping_distance) & (depth_cropped > 0.1), True, False)

            #print(f'cutoffImage shape is {cutoffImage.shape}, depth_cropped shape is {depth_cropped.shape}');
            avg_x = 0
            avg_x_array = np.array([])
            countB = 0
            for a in range(np.size(cutoffImage,0)):
                for b in range(np.size(cutoffImage,1)):
                    if cutoffImage[a,b] :
                        avg_x += b
                        #print(b)
                        avg_x_array = np.append(avg_x_array,b)
                        countB = countB+1
            # if we got no pixels in depth, return false
            if countB <= 40: 
                return -5.0
            
            avg_x_array.sort()
            islands = []
            i_min = 0
            i_max = 0
            p = avg_x_array[0]
            for index in range(np.size(avg_x_array,0)) :
                n = avg_x_array[index]
                if n > p+1 and not i_min == i_max : # if the island is done
                    islands.append(avg_x_array[i_min:i_max])
                    i_min = index
                i_max = index
                p = n
            if not i_min == i_max: islands.append(avg_x_array[i_min:i_max])
            

            #print(islands)
            bigIsland = np.array([])
            for array in islands:
                if np.size(array,0) > np.size(bigIsland,0): bigIsland = array
            
            #print(np.median(bigIsland))
            m = (np.median(bigIsland))

            return (m/(np.size(cutoffImage,1)) * 1) # -0.2 # return value
        return -5.0 # failed to get camera image, return bas value

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
    clipping_distance_in_meters = 1.0 #2 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    print(f'clipping distance is : {clipping_distance}')

    level = 0
    # start at level 0, our start state when nobody is playing
    subscriber.emit_level(level) 

    # was orginally in the loop
    instance = GameDriver(subscriber, opponent, agent, pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance)

    while True: # play the exhibit on loop forever

        #time.sleep(1)
        start = time.time()
        
        #wait until human detected, if no human after a few seconds, back to zero

        if level == 0: # if the game hasn't started yet and the next level would be level 1
            print("          Waiting for user interaction to begin game . . . ")
            # checking if theres a large enough human blob to track
            while not check_for_player(pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance):
                time.sleep(0.01) # will just loop and stay at level zero until it sees someone
            print("          Human detected, checking if still . . . ")

            arrayVals = np.array([]) # empty numpy array to store values to check if person is still
            has_bad_values = False 

            # take 40 measurements of the player to see if they actually are trying to
            for counter in range(0,40):
                c_value = check_for_still_player(pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance)
                if c_value == -0.5:
                    # c_value is our dummy value for not seeing a human blob
                    # continue loop back at waiting for interaction
                    has_bad_values = True
                arrayVals = np.append(arrayVals, c_value)
            
            # if standard deviation of values from check_for_still_player was too variable (person walking by) or there was no person:
            if np.std(arrayVals) > 0.07 or has_bad_values:
                # either there wasn't a person blob large enough, or the player wasn't still. Don't start game
                print("          No still player.        ")
                continue

            level = 1
            print(f'          Still human detected, beginning level {level}. ')
            subscriber.emit_level(level) 
            instance.run(level) # RUN LEVEL (1)
        elif level == 1 or level == 2: # if we just played level 1 or 2 and now have to play level 3
            print("          Waiting for user interaction to advance level . . . ")
            counter = 0
            while True:
                if check_for_player(pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance):
                    # there is still a large enough player blob present, move on to next level
                    level = level + 1
                    print(f'          Human detected, beginning level {level}. ')
                    subscriber.emit_level(level)
                    instance.run(level) # RUN LEVEL (2 or 3)
                    break
                elif counter > 300: # if we've been waiting for too long for a player to enter, reset game
                    level = 0
                    print(f'            No Player detected, resetting game')
                    print(f'            Game reset to level {level} (zero).')
                    subscriber.emit_level(level)
                    break
                counter = counter + 1 # incrementing a counter as a way to timeout of the game if nobody is playing
                time.sleep(0.01)

        else: # level == 3
            level = 0 # the game is over so we need to reset to level 0 (the state before the game starts)
            print(f'            Game reset to level {level} (zero).')
            subscriber.emit_level(level)
            time.sleep(1) # wait 1 second so person has time to leave and next person can come in





