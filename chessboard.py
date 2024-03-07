import open3d as o3d
import numpy as np
import cv2
import pygame
import sys

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import math


def creatchessboard(size):
    distance = 100
    board_shape=(distance*size[0],distance*size[1])
    white=np.zeros(shape=(distance,distance))
    black=white+255
    board=np.zeros(shape=board_shape)
    l1=np.array([(1+pow(-1,i1+i2))/2 for i1 in range(0,size[0]) for i2 in range(0,size[1])]).reshape(size[0],size[1])
    x,y=np.where(l1==1)
    for i1 in range(0,len(x)):
        board[x[i1]*distance:(x[i1]+1)*distance,y[i1]*distance:(y[i1]+1)*distance]=black
    return board

im1=creatchessboard((7,7))

cv.imwrite("chessboard.png",im1)
