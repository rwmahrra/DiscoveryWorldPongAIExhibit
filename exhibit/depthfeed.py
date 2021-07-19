import pyrealsense2 as rs
import numpy as np

decimation_filter = rs.decimation_filter()
decimation_filter.set_option(rs.option.filter_magnitude, 6)

crop_percentage = .2

print("starting with crop at {}".format(crop_percentage * 100))
try:
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        # get an average depth value over a portion of the center
        depth_filtered = decimation_filter.process(depth)
        depth_image = np.asanyarray(depth_filtered.get_data())
        
        w,h = depth_image.shape
        ws, we = int(w/2 - (w * crop_percentage)), int(w/2 + (w * crop_percentage))
        hs, he = int(h/2 - (h * crop_percentage)), int(h/2 + (h * crop_percentage))
        print("dimension: {}, {}, width: {},{} height: {},{}".format(w,h,ws,we,hs,he))
        depth_cropped = depth_image[ws:we, hs:he]
        mean = np.mean(depth_cropped)
        print(mean)
except Exception as e:
    print(e)
finally:
    print("completed")
    pipeline.stop()
    exit(0)
