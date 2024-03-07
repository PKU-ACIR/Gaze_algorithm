import open3d as o3d
import numpy as np
import cv2
import pygame
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import math
import pyrealsense2 as rs
from haitai import HaiTai
import time


import numpy as np
from numpy.linalg import solve

#from win32.win32api import  GetSystemMetrics

from gaze_tracker import GAZE_TRACKER



#%% 定义函数




def calibration(color_image,depth_image,pipeline, align, depth_scale,depth_intrin,color_intrin):
    crireria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.001)
 
    # 做一些3D点
    objp = np.zeros((6*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:6].T.reshape(-1, 2) 
    objp = objp * 0.074
    objpoints = []

    imgpoints = []
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray_tmp = (1280,720)
    #cv2.imshow("test",gray)
    #cv2.waitKey(1)
    print(gray_tmp)
    ret, corners = cv2.findChessboardCorners(gray, (6,6), None)
    max_depth = 6
    distance_sum = 0
    sum = 0
    direction_x = [0,0,0]
    direction_y = [0,0,0]
    if ret == False:
        return False,0,0,0,0,0,0,0,0
        #画出亚像素精度角点
    corners2 = cv2.cornerSubPix(gray, corners, (6,6), (-1, -1), crireria)
    imgpoints.append(corners2)
    objpoints.append(objp)
    cv2.drawChessboardCorners(color_image, (6,6), corners2, ret)  
    camera_intrin = np.array([[631.881, 0., 629.202],
           [0., 631.496, 362.336],
           [0., 0., 1.]])

    #cv2.imshow("chess",color_image)
    #cv2.waitKey(1)

    real_corners = np.zeros((36,3))
    for i in range(36):
        x = np.round(corners2[i][0][0])
        y = np.round(corners2[i][0][1])
        depth_value = depth_image[np.round(y).astype(np.int),np.round(x).astype(np.int)]*depth_scale
        if 0 < depth_value <= max_depth:
            curr_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], depth_value)
            real_corners[i] = curr_point

    for i in range(5):
        for j in range(5):
            if real_corners[i+j*6][0] == 0:
                continue
            if real_corners[i+j*6+1][0] != 0:
                
                tmp1 = real_corners[i+j*6]
                tmp2 = real_corners[i+j*6+1]
                if np.linalg.norm(real_corners[i+j*6]-real_corners[i+j*6+1]) > 0.1:
                    continue
                distance_sum += np.linalg.norm(real_corners[i+j*6]-real_corners[i+j*6+1])
                direction_x += real_corners[i+j*6]-real_corners[i+j*6+1]
                sum += 1
            if real_corners[i+(j+1)*6][0] != 0:
                if np.linalg.norm(real_corners[i+j*6]-real_corners[i+(j+1)*6]) > 0.1:
                    continue
                distance_sum += np.linalg.norm(real_corners[i+j*6]-real_corners[i+(j+1)*6])
                direction_y += real_corners[i+j*6]-real_corners[i+(j+1)*6]
                sum += 1

    

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_tmp, camera_intrin, np.array(color_intrin.coeffs),flags = cv2.CALIB_FIX_K3)

    gray_tmp = []
    distance = distance_sum/sum
    direction_x = -direction_x/np.linalg.norm(direction_x)
    direction_y = -direction_y/np.linalg.norm(direction_y)
    direction_z = np.cross(direction_x,direction_y)
    direction_z = direction_z/np.linalg.norm(direction_z)

    center = real_corners[0] - 2*direction_x*distance-2*direction_y*distance
    return True,center,distance,direction_x, direction_y, direction_z, real_corners, np.array(rvecs).reshape(3,1), np.array(tvecs).reshape(3,1)

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

def get_rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def calculate_position(direction,face_center,center,distance,direction_x,direction_y,direction_z,angle,axis):
    rotation_matrix = get_rotation_matrix(np.radians(angle))
    face_center = face_center+axis
    #face_center[0] = face_center[0] + 0.1
    rotated_face_center = face_center @ rotation_matrix.T
    
    rotated_direction = direction @ rotation_matrix.T
    #rotated_face_center[0] = rotated_face_center[0] - 0.1
    rotated_face_center = rotated_face_center - axis
    rotated_direction = rotated_direction/np.linalg.norm(rotated_direction)
    if np.dot(rotated_direction , direction_z)  == 0 or face_center[1] == 0:
        return [0,0],0
    m = np.dot(center-rotated_face_center,direction_z)/np.dot(rotated_direction,direction_z)
    surface_projection = m*rotated_direction+rotated_face_center
    tmp1 = np.dot(center-rotated_face_center,direction_z)
    tmp2 = np.dot(rotated_direction,direction_z)
    print("m direction")
    print(m)
    print(surface_projection)
    print(tmp1)
    print(tmp2)
    print(rotated_direction)

    
    #surface_projection = point-np.dot(center,direction_z)
    x = np.dot((surface_projection-center),direction_x)*100/distance
    y = np.dot((surface_projection-center),direction_y)*100/distance
    
    return [x,y],rotated_face_center

def solve_axis(start_corners,end_corners,angle):
    ang = np.radians(angle)
    cos = np.cos(ang)
    sin = np.sin(ang)
    A = np.mat([[1-cos,sin],[-sin,1-cos]])
    count = 0
    axis = np.array([[0.0],[0.0]])
    print(start_corners)
    print(end_corners)

    for i in range(len(start_corners)):
        if start_corners[i][2] != 0 and end_corners[i][2] != 0:
            b = np.mat([start_corners[i][0]*cos-start_corners[i][2]*sin-end_corners[i][0],start_corners[i][2]*cos+start_corners[i][0]*sin-end_corners[i][2]]).T
            tmp = np.array(solve(A,b))
            print("tmp")
            print(tmp)
            if  np.any(tmp > 0.5):
                continue
            axis += tmp
            count += 1
    print(axis)
    axis = axis/count
    axis = axis.tolist()
    print(axis)
    return [axis[0][0],0,axis[1][0]]

def pixel_to_distance(center,distance,direction_x,direction_y,direction_z,pixel_x,pixel_y):
    return center+(pixel_x*direction_x+pixel_y*direction_y)*distance/100
    



def test(gaze_tracker, pipeline, align, depth_scale,haitai):
    pygame.init()
    #size = width, height = GetSystemMetrics(0), GetSystemMetrics(1)
    width, height = 1920, 1080
    half_width = width/2
    half_height = height/2
    screen = pygame.display.set_mode((1920,1080),flags=pygame.FULLSCREEN)
    #screen = pygame.display.set_mode(size,0)
    color = (255, 255, 255)
    chessboard = pygame.image.load('resource/chessboard.png')  # 加载图片
    #chessboard_rect = chessboard.get_rect()
    chessboard_rect = (100,100)
    MY_EVENT = pygame.USEREVENT+1

    axis_distance = 0.1

    eye = pygame.image.load('resource/eye.png')
    aim = pygame.image.load('resource/aim.png')
    


    
    calibration_flag = False
    tracker_flag = False
    curr_angle = haitai.get_curr_angle()
    cal_angle = 10
    angle = 100
    circle_pos = [0,100]
    circle_color = (0,0,255)
    axis = [0,0,0]
    eye_rect = [0,0]
    aim_rect = [0,0]
    x,y = 0,0
    font = pygame.font.Font('freesansbold.ttf', 50)
    text = font.render('distance: 0', True, (0, 0, 255), (0, 255, 0))
    angle_text = font.render('angle: 0', True, (0, 0, 255), (0, 255, 0))
    
    while True:
        depth_frame, color_frame, depth_intrin = realsense_get_frames(pipeline, align)
        

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        if tracker_flag == True:
                
            re,flag,direction,face_center = gaze_tracker.run(color_image, depth_image, depth_intrin,True)
                        
            if re and not flag:
                print("direction")
                print(direction)
                x, y = pygame.mouse.get_pos()
                pos,rota = calculate_position(direction,face_center,center,distance,direction_x,direction_y,direction_z,angle,axis)
                print("pos")
                print(pos)
                print("face_center")
                print(face_center)
                print("rotated_face_center")
                print(rota)
                if pos[0] > 0 and pos[0] < width and pos[1] > 0 and pos[1] < height:
                    circle_pos = [np.round(pos[0]).astype(np.int),np.round(pos[1]).astype(np.int)]
                                
                    print("distance")
                    dis = np.linalg.norm(circle_pos-np.array([x,y]))
                    print(dis)
                    text = font.render('distance: '+ str(dis), True, (0, 0, 255), (0, 255, 0))
                                
                    
                    circle_color = (0,0,255)
                                
                else:
                    offset = pos - np. array([half_width,half_height]) 
                    offset[1] = -offset[1]
                    option1 = offset[0]*half_height/(offset[1])
                    option2 = offset[1]*half_width/(offset[0])

                    if np.abs(option1) < width/2:
                        circle_pos = [np.round(half_width+option1*np.sign(offset[1])).astype(np.int),np.round(half_height-half_height*np.sign(offset[1])).astype(np.int)]
                        circle_color = (0,255,255)
                                    
                    else:
                        circle_color = (0,255,255)
                        circle_pos = [np.round(half_width+half_width*np.sign(offset[0])).astype(np.int),np.round(half_height-option2*np.sign(offset[0])).astype(np.int)]
                detect_pos = pixel_to_distance(center,distance,direction_x,direction_y,direction_z,pos[0],pos[1])
                aim_pos = pixel_to_distance(center,distance,direction_x,direction_y,direction_z,x,y)
                tmp1 = face_center-detect_pos
                tmp2 = face_center-aim_pos
                
                delta_angle = np.arccos((tmp1 @ tmp2.T)/(np.linalg.norm(tmp1)*np.linalg.norm(tmp2)))*180.0/np.pi
                angle_text = font.render('angle: '+ str(delta_angle), True, (0, 0, 255), (0, 255, 0))
                eye_rect = [circle_pos[0]-150,circle_pos[1]-85]
                aim_rect = [x-100,y-100]
                
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            if not depth_frame or not color_frame:
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    screen = pygame.display.set_mode((640,480),0)
                if event.key == pygame.K_q:
                    pygame.quit()
                    return
                if event.key == pygame.K_c:
                    calibration_flag = not calibration_flag
                if event.key == pygame.K_x:
                    time.sleep(5)
                
                
                if event.key == pygame.K_z:
                    if tracker_flag == True:
                
                        re,flag,direction,face_center = gaze_tracker.run(color_image, depth_image, depth_intrin)
                        
                        if re:
                            print("direction")
                            print(direction)
                            x, y = pygame.mouse.get_pos()
                            pos,rota = calculate_position(direction,face_center,center,distance,direction_x,direction_y,direction_z,angle,axis)
                            print("pos")
                            print(pos)
                            print("face_center")
                            print(face_center)
                            print("rotated_face_center")
                            print(rota)
                            if pos[0] > 0 and pos[0] < width and pos[1] > 0 and pos[1] < height:
                                circle_pos = [np.round(pos[0]).astype(np.int),np.round(pos[1]).astype(np.int)]
                                
                                print("distance")
                                dis = np.linalg.norm(circle_pos-np.array([x,y]))
                                print(dis)
                                
                                text = font.render('distance: '+ str(dis), True, (0, 0, 255), (0, 255, 0))
                                
                            else:
                                offset = pos - np. array([half_width,half_height]) 
                                offset[1] = -offset[1]
                                option1 = offset[0]*half_height/offset[1]
                                option2 = offset[1]*half_width/offset[0]

                                if np.abs(option1) < half_width:
                                    circle_pos = [np.round(half_width+option1*np.sign(offset[1])).astype(np.int),np.round(half_height-half_height*np.sign(offset[1])).astype(np.int)]
                                    
                                else:
                                    circle_pos = [np.round(half_width+half_width*np.sign(offset[0])).astype(np.int),np.round(half_height-option2*np.sign(offset[0])).astype(np.int)]
                            detect_pos = pixel_to_distance(center,distance,direction_x,direction_y,direction_z,pos[0],pos[1])
                            aim_pos = pixel_to_distance(center,distance,direction_x,direction_y,direction_z,x,y)
                            tmp1 = face_center-detect_pos
                            tmp2 = face_center-aim_pos
                            delta_angle = np.arccos((tmp1 @ tmp2.T)/(np.linalg.norm(tmp1)*np.linalg.norm(tmp2)))*180.0/np.pi
                            angle_text = font.render('angle: '+ str(delta_angle), True, (0, 0, 255), (0, 255, 0))

                            eye_rect = [circle_pos[0]-150,circle_pos[1]-85]
                            aim_rect = [x-100,y-100]
                            
                if event.key == pygame.K_s:
                    color_profile = color_frame.get_profile()
                    cvsprofile = rs.video_stream_profile(color_profile)
                    color_intrin = cvsprofile.get_intrinsics()
                    
                    ret,center,distance,direction_x,direction_y,direction_z,real_corners,rvec_1,tvec_1 = calibration(color_image,depth_image,pipeline, align, depth_scale,depth_intrin,color_intrin)
                    


                    
                    
                    if distance != 0 and ret == True and tracker_flag == False:
                        rmat_1,_ = cv2.Rodrigues(rvec_1)
                        cv.imwrite('output/start.png',color_image)
                        target_angle = cal_angle + curr_angle
                        haitai.set_abs_angle(target_angle, wait=True)
                        target_angle = haitai.get_curr_angle()
                        cal_angle = target_angle-curr_angle
                        print(real_corners)
                        time.sleep(5)
                    
                        while(True):

                            depth_frame, color_frame, depth_intrin = realsense_get_frames(pipeline, align)
                            color_image = np.asanyarray(color_frame.get_data())
                            depth_image = np.asanyarray(depth_frame.get_data())
                            ret,ro_center,ro_distance,d_x,d_y,d_z,rotated_real_corners,rvec_2,tvec_2 = calibration(color_image,depth_image,pipeline, align, depth_scale,depth_intrin,color_intrin)
                            if ret == False:
                                continue
                            print(rotated_real_corners)
                            rmat_2,_ = cv2.Rodrigues(rvec_2)
                            r = np.dot(rmat_2,np.linalg.inv(rmat_1))
                            r_vec,_ = cv2.Rodrigues(r)
                            theta = np.linalg.norm(r_vec)
                            print(theta * 180 / np.pi)
                            print(r_vec/theta)
                            t = -np.dot(r,tvec_1)+tvec_2
                            print(t)
                            cv2.imwrite('output/end.png',color_image)
                            break
                            
                        

                        axis = solve_axis(real_corners,rotated_real_corners,cal_angle)
                        target_angle = curr_angle+angle
                        haitai.set_abs_angle(target_angle, wait=True)
                        target_angle = haitai.get_curr_angle()
                        angle = target_angle-curr_angle
                        tracker_flag = True
                        print(axis)
                        calibration_flag = not calibration_flag

        

        screen.fill(color)
        if tracker_flag and x != 0:
            screen.blit(eye,eye_rect)
            screen.blit(aim,aim_rect)
            screen.blit(text,(0,0))
            screen.blit(angle_text,(0,100))
            pygame.draw.line(screen,(0,0,0),[x,y],circle_pos,10)
        
        
        if calibration_flag:
            screen.blit(chessboard, chessboard_rect)
            
        pygame.display.flip()

if __name__ == '__main__':
    #if platform.system() == 'Windows':
    #    haitai = HaiTai('COM3')
    #else:
    #    haitai = HaiTai('/dev/ttyUSB0')
    #haitai.connect()
    #print("success connect haitai")
    #haitai.set_abs_angle(180, wait=True)
    haitai = None
    

    pipeline, align, depth_scale = realsense_configration()

    gaze_tracker = GAZE_TRACKER(depth_scale)
    test(gaze_tracker,pipeline,align,depth_scale,haitai)


