import sys
from exhibit.shared.config import Config
from exhibit.camera.camera_subscriber import CameraSubscriber
import time
import numpy as np
import base64

import pyrealsense2 as rs
import cv2
import threading
from exhibit.shared.utils import Timer

from queue import Queue


class CameraDriver:

    def publish_inference(self):
        #Timer.start('inf')
        
        # check if a kill/quit message has been sent via the queue
        if not self.q.empty():
            dataQ = self.q.get()
            if dataQ == "endThreads":
                print('ai thread quitting')
                while not self.q.empty: # empty the rest of the q
                    dataQ = self.q.get()
                self.q.put('noneActive') 
                # a message that goes back to the main program to tell it that the ai_driver has stopped
                sys.exit()
                print('the sys exit didnt work')

        

        # Infer on flattened state vector
        x = diff_state.ravel()
        action, _, probs = self.agent.act(x)
        # Publish prediction
        if self.paddle1:
            self.state.publish("paddle1/action", {"action": str(action)})
            self.state.publish("paddle1/frame", {"frame": current_frame_id})
        elif self.paddle2:
            self.state.publish("paddle2/action", {"action": str(action)})
            self.state.publish("paddle2/frame", {"frame": current_frame_id})

        model_activation = self.agent.get_activation_packet()
        self.state.publish("ai/activation", model_activation)



    def __init__(self, config=Config.instance(), in_q = Queue(), pipeline = rs.pipeline(), decimation_filter = rs.decimation_filter(), crop_percentage_w = 1.0, crop_percentage_h = 0.1, clipping_distance_in_meters = 1.6):
        self.pipeline = pipeline # = pipeline
        self.decimation_filter = decimation_filter
        self.crop_percentage_w = crop_percentage_w
        self.crop_percentage_h = crop_percentage_h
        self.clipping_distance_in_meters = clipping_distance_in_meters
        self.clipping_distance = clipping_distance_in_meters
        self.config = config
        self.subscriber = CameraSubscriber() #= subscriber
        

        #self.state = AISubscriber(self.config, trigger_event=lambda: self.publish_inference())
        #self.last_frame_id = self.state.frame
        # self.last_tick = time.time()
        # self.frame_diffs = []
        # self.last_acted_frame = 0
        # self.inference_thread = threading.Thread(target=self.inference_loop)
        # self.inference_thread.start()
        # self.state.start()

        self.configure_pipeline()
    
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


    # checks if theres a big enough player sized blob
    #check_for_player(pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance)
    def check_for_player(self):
        #try to get the frame 50 times
        for i in range(50): 
            frames = self.pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            #color = frames.get_color_frame()
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
                return False

            return True # successfully found a player, return true
        return False # failed to get camera image, return false

    # This is the currently used method.
    # This finds the "biggest blob"/island and gets the x coordinate of their center of mass
    # It filters out everything below a certain height
    def get_human_x(self):
        #try to get the frame 50 times
        for i in range(50):
            print(f"trials: {i}")
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
            ws, we = int(w/2 - (w * self.crop_percentage_w)/2), int(w/2 + (w * Pong.crop_percentage_w)/2)
            hs, he = int(h/2 - (h * self.crop_percentage_h)/2), int(h/2 + (h * Pong.crop_percentage_h)/2)
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
            #bg_removed = np.where((depth_cropped_3d < Pong.clipping_distance) & (depth_cropped_3d > 0.1), color_image, grey_color )

            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW)

            depth_cropped_3d_actual = np.dstack((depth_cropped,depth_cropped,depth_cropped))
            depth_cropped_3d_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_cropped_3d_actual, alpha=0.03), cv2.COLORMAP_RAINBOW)
            #Timer.stop("colorize")
            #Timer.start("drawline")
            depth_cropped_3d_colormap = np.where((depth_cropped_3d_actual < Pong.clipping_distance) & (depth_cropped_3d_actual > 0.1), depth_cropped_3d_colormap, grey_color2 )
            
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
            # we multiply by 1.4 and subtract -0.2 so that the player can reach the edges of the Pong game.
            # In other words, we shrunk the frame so that the edges of the pong game can be reached without leaving the camera frame
            return (m/(np.size(cutoffImage,1)) * 1.4) -0.2
        print("depth failed")
        #Timer.stop("get_depth")
        return 0.5 # dummy value if we can't successfully get a good one


# checks if there is a player and returns their position from 0 to 1 so that we can tell if theyre walking through or still to play
# check_for_still_player(pipeline, decimation_filter, crop_percentage_w, crop_percentage_h, clipping_distance)

def check_for_still_player(self):
    #try to get the frame 50 times
    for i in range(20):
        try:
            # print('trying wait_for_frames')
            frames = self.pipeline.wait_for_frames()
        except Exception as ed:
            print(ed)
            continue

        depth = frames.get_depth_frame()
        #color = frames.get_color_frame()
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

def main(in_q):
    # main is separated out so that we can call it and pass in the queue from GUI
    config = Config.instance()
    instance = CameraDriver(config = config, in_q = in_q)

if __name__ == "__main__":
    main("")

