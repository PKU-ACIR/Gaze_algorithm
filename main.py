import time
import numpy as np
import cv2
import pyrealsense2 as rs
from matplotlib import pyplot


from haitai import HaiTai
import platform

from screen_test import test
from gaze_tracker import GAZE_TRACKER
from object_search_realsense import ObjectSearcher
#from Memory.gpu_mem_track import MemTracker




def realsense_configration():
    """
    realsense d455 configration
    :return: realsense pipeline and align configration
    """
    # Create a realsense pipeline
    d455_pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    # different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    # pipeline_wrapper = rs.pipeline_wrapper(d455_pipeline)
    # pipeline_profile = config.resolve(pipeline_wrapper)
    # device = pipeline_profile.get_device()
    # device_product_line = str(device.get_info(rs.camera_info.product_line))

    # configration of the resolution
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = d455_pipeline.start(config)

    # Getting the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: ", depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    d455_align = rs.align(align_to)

    return d455_pipeline, d455_align, depth_scale


def realsense_get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    return depth_frame, color_frame, depth_intrin


if __name__ == '__main__':
    
    if platform.system() == 'Windows':
        haitai = HaiTai('COM3')
    else:
        haitai = HaiTai('/dev/ttyUSB0')
    haitai.connect()
    print("success connect haitai")
    haitai.set_abs_angle(180, wait=True)
    
    #haitai = None
    

    pipeline, align, depth_scale = realsense_configration()

    gaze_tracker = GAZE_TRACKER(depth_scale)
    searcher = ObjectSearcher(haitai, pipeline, align, depth_scale)

    while True:
        time_s = time.time()

        depth_frame, color_frame, depth_intrin = realsense_get_frames(pipeline, align)
        color_profile = color_frame.get_profile()
        cvsprofile = rs.video_stream_profile(color_profile)
        color_intrin = cvsprofile.get_intrinsics()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        _,flag,focusing,attention_vec,face_center = gaze_tracker.run(color_image, depth_image, depth_intrin,True)
        time_e = time.time()
        if not flag and focusing:
            target_class, target_id = searcher.search_object(attention_vec, face_center)
            print("you are looking at: "+str(target_class)+" " + str(target_id))
            break
        

        if cv2.waitKey(33) == ord(' '):
            while(1):
                if cv2.waitKey(33) == ord('q'):
                    break
        if cv2.waitKey(33) == ord('z'):
            break
    gaze_tracker.writer.release()
    while True:
        searcher.visualizer.update_renderer()
        time.sleep(0.05)

    # cv2.destroyAllWindows()
    # haitai.close()
