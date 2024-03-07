import time
import numpy as np
import cv2
import pyrealsense2 as rs
from matplotlib import pyplot
import sys
sys.path.append("./RobotGazeFollowing")


import platform

from gaze_tracker_print import GAZE_TRACKER
from gaze_estimation.personal import Personal_Module
#from object_search_realsense import ObjectSearcher
#from Memory.gpu_mem_track import MemTracker


class ARC():
    def __init__(self):
        self.center = [0,0]
        self.pitch = []
        self.yaw = []
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.tracker = GAZE_TRACKER(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        # self.cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
        self.total_angle = 26*3
        #self.total_angle = 196.5
        self.a_h = 0.21
        self.width = 3440*3
        #self.width = 3880
        #self.a_h = 1.56
        self.arc_center = []
        self.user_pos = []
        self.u_c_angle = 0
        self.u_c_distance = 0
        self.R = 1.8
        #self.R = 1.4
        self.p_m = Personal_Module()
        self.cali_ready = False
        self.cali_start = 0
        self.now=0

    def calibration(self):
        if self.cali_start == 0:
            self.cali_start = time.time()
        rval,frame = self.cap.read()
        re,flag,focusing,attention_vec,face_center = self.tracker.run(frame, [], [],False)
        if not re:
            return False,False

        if focusing:
            pitch = np.arcsin(-attention_vec[1])
            yaw = np.arcsin(-attention_vec[0]/np.cos(pitch))
            self.pitch.append(pitch)
            self.yaw.append(yaw)
            self.now = time.time()
            if self.now-self.cali_start > 1:
                self.center[0] = np.average(self.pitch)
                self.center[1] = np.average(self.yaw)
                self.cali_ready = True
                return True,True
            return True,False
        else:
            self.cali_start=time.time()
            pitch = []
            yaw = []
            return True,False

    def personal_cali(self,cali_point):
        if self.cali_start == 0:
            self.cali_start = time.time()
        aim_target = []
        
        distance = 720-cali_point[0][1]
        pitch = np.arctan(distance*self.a_h/(1440*1.8))+self.center[0]
        del_yaw = (cali_point[0][0]/self.width-0.5)*self.total_angle
        yaw = del_yaw*np.pi/180+self.center[0]
        aim_target.append([pitch,yaw])
        current = 0

        rval,frame = self.cap.read()
        re,flag,focusing,attention_vec,face_center = self.tracker.run(frame, [], [],False)
        if not re:
            return False,False
        if focusing:
            pitch = np.arcsin(-attention_vec[1])
            yaw = np.arcsin(-attention_vec[0]/np.cos(pitch))
            self.p_m.add(pitch,yaw,aim_target[0][0],aim_target[0][1])
            self.now = time.time()
            if self.now-self.cali_start > 1:
                    return True,True
            return True,False
                
        else:
            self.cali_start=time.time()
            pitch = []
            yaw = []
            return True,False

    def calculate(self):
        rval,frame = self.cap.read()
        re,flag,focusing,attention_vec,face_center = self.tracker.run(frame, [], [],False)
        if not re and flag:
            return int(self.width/2),int(720)
        pitch = np.arcsin(-attention_vec[1])
        yaw = np.arcsin(-attention_vec[0]/np.cos(pitch))
        del_pitch = pitch-self.center[0]
        del_yaw = (yaw-self.center[1])*180/np.pi

        distance = 1440*np.tan(del_pitch)*1.8/self.a_h
        scale = del_yaw/self.total_angle
        x = self.width/2+scale*self.width
        y = 720-distance
        if x < 0:
            x = 22
        if x > self.width:
            x = 10280
        if y < 0:
            y = 160
        if y > 1440:
            y = 1280    
        return int(x),int(y)

    def free_calculate(self):
        rval,frame = self.cap.read()
        re,flag,focusing,attention_vec,face_center = self.tracker.run(frame, [], [],False)

        if not re and flag:
            return int(self.width/2),int(720)
        pitch = np.arcsin(-attention_vec[1])
        yaw = np.arcsin(-attention_vec[0]/np.cos(pitch))


        del_pitch = pitch-self.center[0]
        perpen_distance = self.u_c_distance*np.cos(self.u_c_angle)+np.sqrt(self.R**2-(self.u_c_distance*np.sin(self.u_c_angle))**2)
        distance = 1440*np.tan(del_pitch)*perpen_distance/self.a_h
        y = 720-distance

        yaw = np.arccos((self.u_c_distance**2+self.R**2-perpen_distance**2)/(2*self.u_c_distance*self.R))
        del_yaw = (yaw-self.center[1])*180/np.pi
        scale = del_yaw/self.total_angle
        x = self.width/2+scale*self.width


    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    



    t = ARC()
    t.calibration()
    print(t.calculate())



