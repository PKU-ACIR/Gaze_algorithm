from object_detector import ObjectDetector
from target_estimator import TargetEstimator
from visualizer import Visualizer

import torch
import torch.backends.cudnn as cudnn
import torchvision
import Detectron2
import pyrealsense2 as rs
import numpy as np

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from object_tracking.utils.parser import get_config
from object_tracking.deep_sort import DeepSort
import math
import cv2

import time

def get_rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def realsense_get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    return depth_frame, color_frame, depth_intrin





class ObjectSearcher():
    def __init__(self, haitai, pipeline, align, depth_scale):
        self.object_detector = ObjectDetector()
        self.target_estimator = TargetEstimator()
        self.visualizer = Visualizer()

        self.haitai = haitai
        self.pipeline = pipeline
        self.align = align
        self.depth_scale = depth_scale

        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
        self.color_writer = cv2.VideoWriter("output/color.avi",fourcc,20,(1280,720))
        self.depth_writer = cv2.VideoWriter("output/depth.avi",fourcc,20,(1280,720),False)

    def search_object(self, position_vec, start_point):
        """
        pitch = position[0] * np.pi / 180
        yaw = position[1] * np.pi / 180

        

        # TODO: position vector in the camera coordinates
        position_vec = -np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        position_vec = position_vec / np.linalg.norm(position_vec)
        """
        pitch = np.arcsin(-position_vec[1])
        yaw = np.arccos(-position_vec[2]/np.cos(pitch))

        self.visualizer.add_attention(start_point, yaw, pitch)

        if position_vec[0] > 0:
            rot_direction = 1
            rot_axis = np.array([-0.1,0,-0.1])
        else:
            rot_direction = -1
            rot_axis = np.array([0.2,0,-0.1])

        init_angle = 180
        rot_step = 10
        target_angle = init_angle
        
        #[0.039187602931393097, 0, -0.10893653558888251]
        #[0.017784037267085873, 0, -0.10841811135328348]
        #[0.0319833595449965, 0, -0.10559533782121161]
        #[0.03354704011483635, 0, -0.11156297570064148]
        #[0.014632300511966833, 0, -0.09645273452452803]
        #[0.010109732396608123, 0, -0.1016093404994408]
        #[0.025320590612195354, 0, -0.09311654409911752]
        #[0.019487799099584438, 0, -0.09361773338278424]
        #[0.006359998587457211, 0, -0.12121465262150927]
        #[-0.1695140830702915, 0, -0.10128169814490579]
        #[0.029450444980165077, 0, -0.1190666483482836]
        while True:
            

            depth_frame, color_frame, depth_intrin = realsense_get_frames(self.pipeline,self.align)
            self.haitai.set_abs_angle(target_angle, wait=True)
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            self.color_writer.write(color_image)
            self.depth_writer.write(depth_image)
            re,classes, points, ids = self.object_detector.get_prediction(color_image, depth_image, depth_intrin,
                                                                       self.depth_scale, total_points_num=10)

            if re == True and len(points) != 0:
                curr_angle = self.haitai.get_curr_angle()
                rotation_matrix = get_rotation_matrix(np.radians(curr_angle - init_angle))
                points += rot_axis
                rotated_points = points @ rotation_matrix.T
                rotated_points -= rot_axis

                points_direction = rotated_points - start_point  # (N, 3) array of direction from start to object
                
                points_direction_norm = (points_direction * points_direction).sum(axis=1) ** 0.5
                points_direction = points_direction / points_direction_norm[:, None]
                angles = np.arccos(points_direction @ position_vec.T) * 180 / np.pi

                self.target_estimator.update_info(rotated_points, classes, angles, ids)

            target_angle = target_angle + rot_direction * rot_step
            self.visualizer.update_renderer()

            if abs(target_angle - init_angle) > 180:
                break

        self.visualizer.add_points(self.target_estimator.all_points, self.target_estimator.all_ids)
        time_s = time.time()
        target_class, target_id = self.target_estimator.get_target_estimation()
        time_e = time.time()
        print("target_estimation:")
        print(time_e-time_s)

        return target_class, target_id
