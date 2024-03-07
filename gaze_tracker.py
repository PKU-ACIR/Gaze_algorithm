import time
import numpy as np
import cv2
import pyrealsense2 as rs
from matplotlib import pyplot

from eye_movement_analysis.probabilistic_model import rnn_method, thres_method

from gaze_estimation.utils import load_config
from gaze_estimation.common import MODEL3D

from gaze_estimation import GazeEstimationMethod, GazeEstimator
from gaze_estimation.common import (Face, FacePartsName, Visualizer)
from gaze_estimation.blink import Blink_Detector
from scipy.spatial import distance as dist

class GAZE_TRACKER():
    def __init__(self,depth_scale):
        self.target_class = None
        self.target_id = None

        self.fix_start = 0
        self.sacc_start = 0
        self.attention_state = []
        self.fix_thres = 0.5
        self.sacc_thres = 10
        self.key = 0

        self.focusing = False

        self.prop = rnn_method(5)

        self.depth_scale = depth_scale

        self.config = load_config()
        self.gaze_estimator = GazeEstimator(self.config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)
        
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model

        self.color_frame = []

        self.savefile = "test.txt"

        self.is_close = False
        self.is_print=True
        self.is_openmouth=False
        self.dismouth=10
        self.flag1=0
        self.flag2=0
        self.blink_detector = Blink_Detector()




        #self.searcher = ObjectSearcher(haitai, pipeline, align, depth_scale)
        
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
        self.writer = cv2.VideoWriter("gaze.avi",fourcc,10,(1280,720))
        
    def get_depth(self,index,depth_scale,depth_intrin,depth_image):
        face_center_xy = tuple(np.round(index).astype(np.int).tolist())
        face_center_x = face_center_xy[0]
        face_center_y = face_center_xy[1]
        head_depth = depth_image[face_center_y, face_center_x] * depth_scale
        face_center = rs.rs2_deproject_pixel_to_point(depth_intrin, [face_center_x, face_center_y], head_depth)

        return face_center

    def find_face_center(self,target_face,depth_scale,depth_intrin,depth_image,color_image):
        re_eye_1 = np.array(target_face.landmarks[33])
        re_eye_2 = np.array(target_face.landmarks[133])
        le_eye_1 = np.array(target_face.landmarks[362])
        le_eye_2 = np.array(target_face.landmarks[263])

        re_1 = np.array(self.get_depth(re_eye_1,depth_scale,depth_intrin,depth_image))
        re_2 = np.array(self.get_depth(re_eye_2,depth_scale,depth_intrin,depth_image))
        le_1 = np.array(self.get_depth(le_eye_1,depth_scale,depth_intrin,depth_image))
        le_2 = np.array(self.get_depth(le_eye_2,depth_scale,depth_intrin,depth_image))

        tmp2 = (re_1+re_2+le_1+le_2)/4
        re_eye_1 = tuple(np.round(re_eye_1).astype(np.int).tolist())
        le_eye_1 = tuple(np.round(le_eye_1).astype(np.int).tolist())

            
        face_center_xy = (re_eye_1+re_eye_2+le_eye_1+le_eye_2)/4
        face_center_xy = tuple(np.round(face_center_xy).astype(np.int).tolist())
        cv2.circle(color_image, face_center_xy,1, (0, 255, 255), cv2.FILLED)
        cv2.imshow("landmarks",color_image)
            
        face_center_x = face_center_xy[0]
        face_center_y = face_center_xy[1]

            
        head_depth = depth_image[face_center_y, face_center_x] * depth_scale
        face_center = rs.rs2_deproject_pixel_to_point(depth_intrin, [face_center_x, face_center_y], head_depth)
        if re_1[2] != 0 and re_2[2] != 0 and le_1[2] != 0 and le_2[2] != 0:
            face_center = tmp2
        return np.array(face_center)

    def run(self, color_image, depth_image, depth_intrin,has_depth=False):
        
        self.visualizer.set_image(color_image.copy())
        time_s_face = time.time()
        totaltime=0

        faces = self.gaze_estimator.detect_faces(color_image)
        #self.writer.write(color_image)
        time_e_face = time.time()
        face_time=time_e_face-time_s_face
        #print("face_detection:")
        #print(time_e_face-time_s_face)

        if faces:
            target_face = faces[0]
            leye = target_face.landmarks[[33,160,158,133,153,144]]
            reye = target_face.landmarks[[363,385,387,263,373,380]]
            mouth0 =target_face.landmarks[[13,14]]
            mydis = dist.euclidean(mouth0[0], mouth0[1])
            #print('mouth dif is :')
            #print(mydis)
            """ if(self.flag1==0):
                self.is_openmouth=0
            if(self.flag1==1 ):
                if(mydis<5):
                    self.is_openmouth=1
                    self.flag1=0
            else:
                self.is_openmouth=0
            if(mydis>8):
                self.flag1=1
            else:
                self.flag1=0 """
            if(mydis>self.dismouth):
              self.is_openmouth=True
            else:
                self.is_openmouth=False
            self.is_close,self.is_print = self.blink_detector.detect(leye,reye)


            gaze_estimate_time,head_pose_time=self.gaze_estimator.estimate_gaze(color_image, target_face)
            totaltime=face_time+head_pose_time+gaze_estimate_time
            print("total time:",face_time+head_pose_time+gaze_estimate_time)

            # gaze_v_cam = target_face.gaze_vector
            # gaze_v_cam = gaze_v_cam / np.linalg.norm(gaze_v_cam)
            angles = np.rad2deg(target_face.vector_to_angle(target_face.gaze_vector))  # [pitch, yaw]
            pitch = angles[0] * np.pi / 180
            yaw = angles[1] * np.pi / 180

            position_vec = target_face.gaze_vector

            #face_center_xy = target_face.landmarks[MODEL3D.NOSE_INDEX]
            if has_depth:
                face_center = self.find_face_center(target_face,self.depth_scale,depth_intrin,depth_image, color_image)
            else:
                face_center = target_face.center

            

            target_face.center = face_center

            self.prop.store(angles, face_center)
            time_s = time.time()
            flag, position, velocity = self.prop.analysis()
            time_e = time.time()
            #print("attention_analysis:")
            #print(time_e-time_s)

            if not flag:
                at_pitch = position[0] * np.pi / 180
                at_yaw = position[1] * np.pi / 180
                attention_vec = -np.array([np.cos(at_pitch) * np.sin(at_yaw), np.sin(at_pitch), np.cos(at_pitch) * np.cos(at_yaw)])
                attention_vec = attention_vec/np.linalg.norm(attention_vec)
            else:
                attention_vec = position_vec
           
            #self.draw_analysis_info(flag, self.focusing, target_face)
            self.draw_visualization(target_face)
            self.writer.write(self.visualizer.image)
            #cv2.imshow('frame', self.visualizer.image)
            self.color_frame = self.visualizer.image
            self.key = cv2.waitKey(1)
            print("center:",target_face.center)
            print("reye:",target_face.reye.center)
            print("leye:",target_face.leye.center)
            eye_center=(target_face.reye.center+target_face.leye.center)/2
            print("eye_center",eye_center)
            print(target_face.head_pose_rot.as_matrix())
           #true_vec=target_face.head_pose_rot.as_matrix().T@attention_vec
            true_vec=attention_vec
            true_vec_now=true_vec
            
            # not focusing
            if flag:
                self.focusing = False
                if self.sacc_start == 0:
                    self.sacc_start = time.time()
                if self.fix_start != 0:
                    self.attention_state.append(['focusing', time.time() - self.fix_start])
                    self.fix_start = 0
                if time.time() - self.sacc_start > self.sacc_thres:
                    self.attention_state.append(['not focusing', time.time() - self.sacc_start])
                    # print('why are you wondering')

            # focusing
            else:
                if self.fix_start == 0:
                    self.fix_start = time.time()
                if self.sacc_start != 0:
                    self.attention_state.append(['not focusing', time.time() - self.sacc_start])
                    self.sacc_start = 0
                if time.time() - self.fix_start > self.fix_thres:
                    # print('what are you thinking about')  
                    #cv2.imshow('gaze result', self.visualizer.image)
                    self.attention_state.append(['focusing', time.time() - self.fix_start])
                    print("focusing time statistics:")
                    self.focusing = True
                    #print(self.attention_state)
                    #self.target_class, self.target_id = self.searcher.search_object(position, face_center)
                    #print("you are looking at: "+str(self.target_class)+" " + str(self.target_id))
            return True,flag,self.focusing,attention_vec,eye_center,true_vec_now
        return False,True,self.focusing,0,0,true_vec_now

    def draw_visualization(self, face):
        
        self._draw_face_bbox(face)
        self._draw_head_pose(face)
        self._draw_landmarks(face)
        self._draw_face_template_model(face)
        self._draw_gaze_vector(face)

    def draw_analysis_info(self, flag, position, face):
        self._draw_flag(flag)
        if not isinstance(position, int):
            self._draw_position(face, position)
 
    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        # euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        # pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        # logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
        #             f'roll: {roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == 'ETH-XGaze':
            self.visualizer.draw_3d_arrowed_line(
                face.center, face.center + length * face.gaze_vector)
            # pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            # logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else:
            raise ValueError

    def _draw_flag(self, flag) -> None:
        if flag:
            text = "not focusing"
            color = (255, 0, 0)
        else:
            text = "focusing"
            color = (0, 255, 0)
        cv2.putText(self.visualizer.image, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 1)

    def _draw_position(self, face, position) -> None:
        length = self.config.demo.gaze_visualization_length
        pitch = position[0] * np.pi / 180
        yaw = position[1] * np.pi / 180
        position_vec = -np.array([
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw)
        ])
        if self.config.mode == 'ETH-XGaze':
            self.visualizer.draw_3d_line(face.center, face.center + length * position_vec, (255, 0, 0))

        else:
            raise ValueError





