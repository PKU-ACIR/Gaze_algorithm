# encoding: utf-8
#!/usr/bin/python
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
 
#设置终止条件
crireria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.001)
 
# 做一些3D点
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
objpoints = []
imgpoints = []
 
images = glob.glob('./kinect_pictures/*.bmp')
gray_tmp = []
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_tmp = gray.shape
    cv2.imshow("test",img)
    print(gray_tmp)
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
 
    if ret == True:
        #画出亚像素精度角点
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (7,7), (-1, -1), crireria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, (7,7), corners2, ret)
 
        # 标定
        # Size imageSize, 在计算相机内部参数和畸变矩阵需要
        # cameraMatrix 为内部参数矩阵， 输入一个cvMat
        # distCoeffs为畸变矩阵.
        # 我们要使用的函数是 cv2.calibrateCamera()。它会返回摄像机矩阵，畸变系数，旋转和变换向量等。
        # mtx内参矩阵， dist畸变系数, rvecs旋转向量，外餐 tvecs平移向量，内参
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_tmp, None, None,flags = cv2.CALIB_FIX_K3)
 
print(fname, "rvecs", rvecs,"mtx",mtx)

np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

k = cv2.waitKey(500) & 0xff
h, w = img.shape[:2]

        # 畸变矫正
        # 如果缩放系数 alpha = 0，返回的非畸变图像会带有最少量的不想要的像素。
        # 它甚至有可能在图像角点去除一些像素。如果 alpha = 1，所有的像素都会被返回，
        # 还有一些黑图像。它还会返回一个 ROI 图像，我们可以用来对结果进行裁剪。
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
 
    if ret == True:
        cv2.imshow('img', img)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # roi只是一个元组
        x, y, w, h = roi
        print(roi)
 
        # 去除畸变
        if k == ord('q'):
            # undistort
            dst = cv2.undistort(img, mtx, dist, newCameraMatrix=newcameramtx)
            dst = dst[y:y+h, x:x+w]
            cv2.imshow('undistortimg', dst)
        else:
            # remapping。先找到畸变到非畸变的映射方程，再用重映射方程
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            dst = dst[y:y+h, x:x+w] #你可以试试注释这一步
            #cv2.imshow('undistortimg ', dst)
            cv2.waitKey()
            
 
        # 我们可以用反向投影对我们找到的参数的准确性评估，结果约接近0越好
        # 有了内部参数：畸变参数和旋转变换矩阵，我们可以用cv2.projectPoints()去
        # 把对象点转换到图像点，然后就可以计算变换得到图像与角点检测算法的绝对差了。
        # 然后我们计算所有标定图像的误差平均值。
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("total error: ", mean_error/len(objpoints))


