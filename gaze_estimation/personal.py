import numpy as np
from sklearn.linear_model import LinearRegression

class Personal_Module(object):
    def __init__(self):
        self.yaw_model = LinearRegression()
        self.pitch_model = LinearRegression()
        self.yaw_x = []
        self.yaw_y = []
        self.pitch_x = []
        self.pitch_y = []


    def add(self,pitch,yaw,g_p,g_y):
        self.yaw_x.append(yaw)
        self.yaw_y.append(g_y)
        self.pitch_x.append(pitch)
        self.pitch_y.append(g_p)

    def cali(self):
       
        self.yaw_model.fit(np.array(self.yaw_x).reshape(-1,1), np.array(self.yaw_y))
        self.pitch_model.fit(np.array(self.pitch_x).reshape(-1,1), np.array(self.pitch_y))

    def recify(self,gaze):
        
        return np.array([self.pitch_model.predict(np.array(gaze[0]).reshape(-1,1)),self.yaw_model.predict(np.array(gaze[1]).reshape(-1,1))])

