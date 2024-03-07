import numpy as np

from numpy.linalg import solve
import torch
import thop
from gaze_estimation.models import model



def solve_axis(start_corners,end_corners,angle):
    ang = np.radians(angle)
    cos = np.cos(ang)
    sin = np.sin(ang)
    A = np.mat([[1-cos,sin],[-sin,1-cos]])
    count = 0
    axis = [0,0]
    print(start_corners)
    print(end_corners)

    for i in range(len(start_corners)):
        if start_corners[i][2] != 0 and end_corners[i][2] != 0:
            b = np.mat([start_corners[i][0]*cos-start_corners[i][2]*sin-end_corners[i][0],start_corners[i][2]*cos+start_corners[i][0]*sin-end_corners[i][2]]).T
            tmp = solve(A,b)
            print("tmp")
            print(tmp)
            axis += tmp
            count += 1
    axis = axis/count
    axis = axis.tolist()
    return [axis[0][0],0,axis[1][0]]

def get_rotation_matrix(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]])

s_c = [[ 0.16460855,-0.19180554 ,0.65100002],
 [ 0.19208343 ,-0.19181767 ,0.65500003],
 [ 0.24468504, -0.18990126  ,0.66000003],
 [ 0.27100015, -0.18929729  ,0.66200006],
 [ 0.29877698, -0.18954626  ,0.66700006]]

e_c = [[-0.11212349,-0.19398792,  0.66800004],
 [-0.08783437, -0.19389595,  0.68200004],
 [-0.04031011, -0.19311325,  0.71000004],
 [-0.01615702, -0.19269167,  0.72500002],
 [ 0.00786054, -0.19269392,  0.73800004]]



angle = 25

model = model.Model().cuda()
inputs = torch.randn(1,3,224,224)   ####(360,640)
inputs=inputs.cuda()
macs, params = thop.profile(model,inputs=(inputs,))   ##verbose=False
print('The number of MACs is %s'%(macs/1e9))   ##### MB
print('The number of params is %s'%(params/1e6))   ##### MB
n_param = sum([p.nelement() for p in model.parameters()])
print(n_param)
vocab = 49
d_model=32
d_ff=512
h=8 
d_k=64
d_out = 1
encoder=4 * vocab *d_model +vocab + 6 * (16* vocab *d_model + h * (6*d_model*vocab*d_k - 4*vocab*d_k + 4*d_k*vocab*vocab + 3*vocab*vocab - vocab) +2 * vocab * d_model *d_model - vocab * d_model + 2 * vocab + 4 * vocab * d_model *d_ff)
print(encoder)
axis = solve_axis(s_c,e_c,angle)

rotation_matrix = get_rotation_matrix(np.radians(-angle))
tmp = np.add(s_c[0],axis)
tt = rotation_matrix.T
rota = tmp @ rotation_matrix.T -axis

print(rota)
