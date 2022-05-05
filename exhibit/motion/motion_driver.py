import sys
from tabnanny import check
from turtle import pos
from exhibit.motion import motion_subscriber
from exhibit.shared.config import Config
from exhibit.motion.motion_subscriber import MotionSubscriber
import time
import numpy as np

import base64
import pyrealsense2 as rs
import cv2
import threading
from exhibit.shared.utils import Timer

import paho.mqtt.client as mqtt
import numpy as np
import json

from queue import Queue

"""
This is a class to provide motion input using the depth feature of a Realsense D435 depth camera.
The motion data is retrieved by finding the center of mass of the largest depth "blob" and then calculating it's position along the horizontal axis
and sending the value over mqtt
"""

class MotionDriver:

    def configure_pipeline(self):
        # decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 6)

        # crop_percentage_w = 1.0
        # crop_percentage_h = 1.0

        print("starting with crop w at {}".format(self.crop_percentage_w * 100))
        print("starting with crop h at {}".format(self.crop_percentage_h * 100))

        #pipeline = rs.pipeline()

        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        profile = self.pipeline.start(rs_config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        # filter out far away so we only have a person
        # clipping_distance_in_meters = 1.6 #2 meter
        self.clipping_distance = self.clipping_distance_in_meters / depth_scale
        print(f'the Clipping Distance is : {self.clipping_distance}')

    def get_position(self):
        # we should be running this when the game is running
        
        for i in range(20):
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception as ed:
                continue

            depth = frames.get_depth_frame()
            if not depth: continue

            # filtering the image to make it less noisy and inconsistent
            depth_filtered = self.decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())

            # cropping the image based on a width and height percentage
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * self.crop_percentage_w)/2), int(w/2 + (w * self.crop_percentage_w)/2)
            hs, he = int(h/2 - (h * self.crop_percentage_h)/2), int(h/2 + (h * self.crop_percentage_h)/2)
            #print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
            depth_cropped = depth_image[ws:we, hs:he]
            #depth_cropped = depth_image

            cutoffImage = np.where((depth_cropped < self.clipping_distance) & (depth_cropped > 0.1), True, False)

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

    def get_human_x(self):
        #try to get the frame 50 times
        #return 0.5
        for i in range(50):
            #print(f"trials: {i}")
            #Timer.start("wait_frame")
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception:
                continue
            #Timer.stop("wait_frame")
            #Timer.start("get_frame")
            depth = frames.get_depth_frame()
            #Timer.stop("get_frame")
            #color = frames.get_color_frame()
            if not depth: continue

            #Timer.start("filter_frame")
            # filtering the image to make it less noisy and inconsistent
            depth_filtered = self.decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())
            #Timer.stop("filter_frame")
            
            #Timer.start("crop_frame")
            # cropping the image based on a width and height percentage
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * self.crop_percentage_w)/2), int(w/2 + (w * self.crop_percentage_w)/2)
            hs, he = int(h/2 - (h * self.crop_percentage_h)/2), int(h/2 + (h * self.crop_percentage_h)/2)
            #print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
            depth_cropped = depth_image[ws:we, hs:he]
            #depth_cropped = depth_image

            cutoffImage = np.where((depth_cropped < self.clipping_distance) & (depth_cropped > 0.1), True, False)
            #Timer.stop("crop_frame")

            #Timer.start("get_islands")
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
            # if we got no pixels in depth, return dumb value
            if countB <= 40: 
                return 0.5
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
            #Timer.stop("get_islands")
            
            #Timer.start("compare_islands")
            #print(islands)
            bigIsland = np.array([])
            for array in islands:
                if np.size(array,0) > np.size(bigIsland,0): bigIsland = array
            
            #print(np.median(bigIsland))
            m = (np.median(bigIsland))
            #Timer.stop("compare_islands")

            # DISPLAYING ******************************************************************************
            #Timer.start("align")
            # print(frames.size())
            aligned_frames = self.align.process(frames)
            #Timer.stop("align")
            #Timer.start("extract")
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            #color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            #print(color_frame)
            #color_image = np.asanyarray(color_frame.get_data())
            #Timer.stop("extract")
            #Timer.start("stack")
            grey_color = 153
            grey_color2 = 40
            #depth_cropped_3d = np.dstack((depth_image,depth_image,depth_image))
            #Timer.stop("stack")
            #Timer.start("colorize")
            #bg_removed = np.where((depth_cropped_3d < self.clipping_distance) & (depth_cropped_3d > 0.1), color_image, grey_color )

            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW)

            depth_cropped_3d_actual = np.dstack((depth_cropped,depth_cropped,depth_cropped))
            depth_cropped_3d_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped_3d_actual, alpha=0.03), cv2.COLORMAP_RAINBOW)
            #Timer.stop("colorize")
            #Timer.start("drawline")
            depth_cropped_3d_colormap = np.where((depth_cropped_3d_actual < self.clipping_distance) & (depth_cropped_3d_actual > 0.1), depth_cropped_3d_colormap, grey_color2 )
            
            # Uncomment these lines to have a window showing the camera feeds
            # images = np.hstack((bg_removed, depth_colormap))
            
            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example',  images)
            # cv2.namedWindow('Filtered', cv2.WINDOW_NORMAL)
            depth_cropped_3d_colormap = cv2.line(depth_cropped_3d_colormap, (int(m),h), (int(m),0), (255,255,255), 1)
            #Timer.stop("drawline")
            
            # cv2.imshow('Filtered',  depth_cropped_3d_colormap)
            
            #Timer.start("imgencode")
            buffer = cv2.imencode('.jpg', depth_cropped_3d_colormap)[1].tostring()
            self.depth_feed = base64.b64encode(buffer).decode()
            #Timer.stop("imgencode")
            

            # *****************************************************************************************
            # we multiply by 1.4 and subtract -0.2 so that the player can reach the edges of the self game.
            # In other words, we shrunk the frame so that the edges of the self game can be reached without leaving the camera frame
            return (m/(np.size(cutoffImage,1)) * 1.4) -0.2
        print("depth failed")
        #Timer.stop("get_depth")
        return 0.5 # dummy value if we can't successfully get a good one

    def motion_loop(self):
        while True:
            # we need to publish if player is present and then position data?
            if(self.subscriber.game_state == 0): # waiting for confirmation of player
                arrayVals = np.array([]) # empty numpy array to store values to check if person is still
                has_bad_values = False

                for counter in range(0,30):
                    c_value = self.get_position()
                    if c_value == -0.5:
                        # c_value is our dummy value for not seeing a human blob
                        # continue loop back at waiting for interaction
                        has_bad_values = True
                    arrayVals = np.append(arrayVals, c_value)

                # if standard deviation of values from check_for_still_player was too variable (person walking by) or there was no person:
                if np.std(arrayVals) > 0.06 or has_bad_values:
                    # either there wasn't a person blob large enough, or the player wasn't still. Don't start game
                    print("          No still player.        ")
                    self.subscriber.publish("motion/presence", False)  
                    continue
                else:
                    self.subscriber.publish("motion/presence", True)    

            # let's run this about 60 times a second to approximately keep up with frame rate
            else: # if ready or running
                position = self.get_human_x()
                self.subscriber.publish("motion/position", position)
                time.sleep(0.016) # wait so we're not spamming as fast as the system can - approx 60 per second is more than enough for a max

    def __init__(self, config=Config.instance(), in_q = Queue(), pipeline = rs.pipeline(), decimation_filter = rs.decimation_filter(), crop_percentage_w = 1.0, crop_percentage_h = 1.0, clipping_distance_in_meters = 1.6):
        self.q = in_q

        # Realsense configuration
        self.pipeline = pipeline 
        self.decimation_filter = decimation_filter
        self.crop_percentage_w = crop_percentage_w
        self.crop_percentage_h = crop_percentage_h
        self.clipping_distance_in_meters = clipping_distance_in_meters
        self.clipping_distance = clipping_distance_in_meters

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.config = config

        self.configure_pipeline() # set up the pipeline for depth retrieval

        self.subscriber = MotionSubscriber()

        self.motion_thread = threading.Thread(target=self.motion_loop)
        self.motion_thread.start()

        self.subscriber.start() # loop the subscriber forever


def main(in_q):
    # main is separated out so that we can call it and pass in the queue from GUI
    config = Config.instance()
    instance = MotionDriver(config = config, in_q = in_q)

if __name__ == "__main__":
    main("")




"""

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
    for i in range(20):
        try:
            # print('trying wait_for_frames')
            frames = pipeline.wait_for_frames()
        except Exception as ed:
            print(ed)
            continue

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




    
    # This is the currently used method.
    # This finds the "biggest blob"/island and gets the x coordinate of their center of mass
    # It filters out everything below a certain height
    def get_human_x():
        #try to get the frame 50 times
        #return 0.5
        for i in range(50):
            #print(f"trials: {i}")
            #Timer.start("wait_frame")
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception:
                continue
            #Timer.stop("wait_frame")
            #Timer.start("get_frame")
            depth = frames.get_depth_frame()
            #Timer.stop("get_frame")
            #color = frames.get_color_frame()
            if not depth: continue

            #Timer.start("filter_frame")
            # filtering the image to make it less noisy and inconsistent
            depth_filtered = self.decimation_filter.process(depth)
            depth_image = np.asanyarray(depth_filtered.get_data())
            #Timer.stop("filter_frame")
            
            #Timer.start("crop_frame")
            # cropping the image based on a width and height percentage
            w,h = depth_image.shape
            ws, we = int(w/2 - (w * self.crop_percentage_w)/2), int(w/2 + (w * self.crop_percentage_w)/2)
            hs, he = int(h/2 - (h * self.crop_percentage_h)/2), int(h/2 + (h * self.crop_percentage_h)/2)
            #print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
            depth_cropped = depth_image[ws:we, hs:he]
            #depth_cropped = depth_image

            cutoffImage = np.where((depth_cropped < self.clipping_distance) & (depth_cropped > 0.1), True, False)
            #Timer.stop("crop_frame")

            #Timer.start("get_islands")
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
            # if we got no pixels in depth, return dumb value
            if countB <= 40: 
                return 0.5
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
            #Timer.stop("get_islands")
            
            #Timer.start("compare_islands")
            #print(islands)
            bigIsland = np.array([])
            for array in islands:
                if np.size(array,0) > np.size(bigIsland,0): bigIsland = array
            
            #print(np.median(bigIsland))
            m = (np.median(bigIsland))
            #Timer.stop("compare_islands")

            # DISPLAYING ******************************************************************************
            #Timer.start("align")
            print(frames.size())
            aligned_frames = self.align.process(frames)
            #Timer.stop("align")
            #Timer.start("extract")
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            #color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            #print(color_frame)
            #color_image = np.asanyarray(color_frame.get_data())
            #Timer.stop("extract")
            #Timer.start("stack")
            grey_color = 153
            grey_color2 = 40
            #depth_cropped_3d = np.dstack((depth_image,depth_image,depth_image))
            #Timer.stop("stack")
            #Timer.start("colorize")
            #bg_removed = np.where((depth_cropped_3d < self.clipping_distance) & (depth_cropped_3d > 0.1), color_image, grey_color )

            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW)

            depth_cropped_3d_actual = np.dstack((depth_cropped,depth_cropped,depth_cropped))
            depth_cropped_3d_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped_3d_actual, alpha=0.03), cv2.COLORMAP_RAINBOW)
            #Timer.stop("colorize")
            #Timer.start("drawline")
            depth_cropped_3d_colormap = np.where((depth_cropped_3d_actual < self.clipping_distance) & (depth_cropped_3d_actual > 0.1), depth_cropped_3d_colormap, grey_color2 )
            
            # Uncomment these lines to have a window showing the camera feeds
            # images = np.hstack((bg_removed, depth_colormap))
            
            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example',  images)
            # cv2.namedWindow('Filtered', cv2.WINDOW_NORMAL)
            depth_cropped_3d_colormap = cv2.line(depth_cropped_3d_colormap, (int(m),h), (int(m),0), (255,255,255), 1)
            #Timer.stop("drawline")
            
            # cv2.imshow('Filtered',  depth_cropped_3d_colormap)
            
            #Timer.start("imgencode")
            buffer = cv2.imencode('.jpg', depth_cropped_3d_colormap)[1].tostring()
            self.depth_feed = base64.b64encode(buffer).decode()
            #Timer.stop("imgencode")
            

            # *****************************************************************************************
            # we multiply by 1.4 and subtract -0.2 so that the player can reach the edges of the self game.
            # In other words, we shrunk the frame so that the edges of the self game can be reached without leaving the camera frame
            return (m/(np.size(cutoffImage,1)) * 1.4) -0.2
        print("depth failed")
        #Timer.stop("get_depth")
        return 0.5 # dummy value if we can't successfully get a good one

"""