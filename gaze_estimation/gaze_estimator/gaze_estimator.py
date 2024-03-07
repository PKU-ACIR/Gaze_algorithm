import logging
from typing import List
#from gaze_estimation.visualizer import get_local

import numpy as np
import torch
import yacs.config

from gaze_estimation.types import GazeEstimationMethod
from gaze_estimation.models import create_model,model,model_gaze360,L2CS,eth_basic,swin,coatnet
from gaze_estimation.transforms import create_transform
from gaze_estimation.common import (Camera, Face,
                                                   FacePartsName)
import torchvision.transforms as T

import torchvision

from gaze_estimation.utils import get_3d_face_model

from gaze_estimation.head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
from easydict import EasyDict as edict
from torchvision import transforms
import time
import cv2
import torch.nn as nn

from torchstat import stat


from PIL import Image, ImageOps
from gaze_estimation.utils import load_config


#from gpu_mem_track import MemTracker


logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: yacs.config.CfgNode):
        self._config = config
        

        self._face_model3d = get_3d_face_model(config)

        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(
            config.gaze_estimator.normalized_camera_params)

        self._landmark_estimator = LandmarkEstimator(config)

        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance)

        #self._gaze_estimation_model = self._load_model()

        self._transform = create_transform(config)

        """
        self.ts_gaze_estimation_model = model.Model()
        self.ts_gaze_estimation_model.eval()
        self.ts_gaze_estimation_model.cuda()
        self.trans = T.Compose([
        T.ToTensor()
        ])
    
        
        self.ts_gaze_estimation_model.load_state_dict(
                torch.load(
                    "models/GazeTR-H-ETH.pt", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                )
            )
        """
        """
        self.baseline = eth_basic.Model()
        stat(self.baseline,(3,224,224))
        
        self.baseline.eval()
        self.baseline.cuda()
        
    
        
        self.baseline.load_state_dict(
                torch.load(
                    "models/eth_basic_34.pt", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                )
            )
        
        
        baseline_params = sum([v.numel() for k,v in self.baseline.state_dict().items()])
        
        self.ts_gaze_estimation_model = model.Model()

        
        self.ts_gaze_estimation_model.eval()
        self.ts_gaze_estimation_model.cuda()
        
    
        
        self.ts_gaze_estimation_model.load_state_dict(
                torch.load(
                    "models/res50_trans_23.pt", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                )
            )
        """
        


        self.trans = T.Compose([
        T.ToTensor()
        ])
        self.swin_transform_model = swin.GazeNet()
        self.swin_transform_model.eval()
        self.swin_transform_model.cuda()
        self.swin_transform_model.load_state_dict(
                torch.load(
                    "models/Iter_19_swin_peg.pt", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                )
            )
        

        """
        self.coatnet = coatnet.GazeNet()
        self.coatnet.eval()
        self.coatnet.cuda()
        self.coatnet.load_state_dict(
                torch.load(
                    "models/Iter_24_coatnet.pt", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                )
            )
        """
        
        """
        self.ts_gaze_estimation_model.load_state_dict(
                torch.load(
                    "models/GazeTR-H-ETH.pt", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                )
            )
            """
        """
        self.gaze_360 = model_gaze360.GazeStatic()
        g3_statedict = torch.load("models/Gaze360.pt", map_location={"cuda:0":"cuda:0"})

        self.gaze_360.load_state_dict(g3_statedict)
        self.gaze_360.cuda()
        
        self.gaze_360.eval()

        self.L2CS = L2CS.L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90)
        self.L2CS.load_state_dict(torch.load(
                    "models/L2CSNet_gaze360.pkl", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                ))
        self.L2CS.cuda()
        self.L2CS.eval()

        self.trans_L2CS = trans_yaw_pitch.Model()
        self.trans_L2CS.load_state_dict(torch.load(
                    "models/Iter_30_trans6.pt", 
                    map_location={f"cuda:{0}": f"cuda:{0}"}
                ))
        self.trans_L2CS.cuda()
        self.trans_L2CS.eval()
        """
        
    

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(self._config.gaze_estimator.checkpoint,
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(torch.device(self._config.device))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        time_s = time.time()
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        time_e = time.time()
        head_pose_time=time_e-time_s
        #print("head_pose:")
        #print(time_e-time_s)
        self._face_model3d.compute_face_eye_centers(face, self._config.mode)

        if self._config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in self.EYE_KEYS:
                eye = getattr(face, key.name.lower())
                self._head_pose_normalizer.normalize(image, eye)
            self._run_mpiigaze_model(face)
        elif self._config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self._head_pose_normalizer.normalize(image, face)
            self._run_mpiifacegaze_model(face)
        elif self._config.mode == 'ETH-XGaze':
            time_s = time.time()
            
            self._head_pose_normalizer.normalize(image, face)
            time_e = time.time()
            #print("normalize:")
            #print(time_e-time_s)
            time_s_1 = time.time()
            #self._run_ethxgaze_model(face)
            self._run_ethxgazetr_model(face)#(best)
            #self._run_L2CS_model(face)
            #self._run_trans_L2CS_model(face)
            #self._run_gaze360_model(face)
            time_e_1 = time.time()
            #print("gaze_estimation:")
            #print(time_e_1-time_s_1)
            return  time_e_1-time_s_1,head_pose_time


    def _run_L2CS_model(self, face: Face)-> None:
        transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        ])
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda()
        softmax = nn.Softmax(dim=1)

        img = cv2.resize(face.normalized_image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        image = transformations(im_pil).unsqueeze(0)

        device = torch.device(self._config.device)
        with torch.no_grad():
            image = image.to(device)
            gaze_pitch, gaze_yaw = self.L2CS(image)
            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)
                    
            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
            pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
            yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        face.normalized_gaze_angles = [pitch_predicted,yaw_predicted]
        face.angle_to_vector()
        face.denormalize_gaze_vector()


    def _run_trans_L2CS_model(self, face: Face)-> None:
        image = self.trans(face.normalized_image).unsqueeze(0)
        #cv2.imshow("normalized_face",face.normalized_image)
        device = torch.device(self._config.device)
        data = edict()
        data.face = image.to(device)
        
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda()
        softmax = nn.Softmax(dim=1)



        device = torch.device(self._config.device)
        with torch.no_grad():
            image = image.to(device)
            gaze_pitch, gaze_yaw = self.trans_L2CS(data)
            pitch_predicted = softmax(gaze_pitch)
            yaw_predicted = softmax(gaze_yaw)
                    
            # Get continuous predictions in degrees.
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
            pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
            yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        face.normalized_gaze_angles = [pitch_predicted,yaw_predicted]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.EYE_KEYS:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1]
                normalized_head_pose *= np.array([1, -1])
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._config.device)
        with torch.no_grad():
            images = images.to(device)
            head_poses = head_poses.to(device)
            predictions = self._gaze_estimation_model(images, head_poses)
            predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()

    def _run_mpiifacegaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        with torch.no_grad():
            image = image.to(device)
            prediction = self._gaze_estimation_model(image)
            prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

    @torch.no_grad()
    def _run_ethxgaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device(self._config.device)
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()
        print("ethxgaze:")
        print(prediction)
        face.normalized_gaze_angles = prediction[0]
        
        face.angle_to_vector()
        face.denormalize_gaze_vector()

    @torch.no_grad()
    def _run_ethxgazetr_model(self,face:Face)-> None:
        
        image = self.trans(face.normalized_image).unsqueeze(0)
        #cv2.imshow("normalized_face",face.normalized_image)
        device = torch.device(self._config.device)
        """
        data = edict()
        data.face = image.to(device)
        """
        
        #prediction = self.ts_gaze_estimation_model(data)
        prediction = self.swin_transform_model(image.to(device))
        prediction = prediction.cpu().numpy()
       # print("swin:")
       # print(prediction)

        face.normalized_gaze_angles = np.array([prediction[0][0],prediction[0][1]])
        #face.normalized_gaze_angles = np.array([prediction[0][1],prediction[0][0]])

        face.angle_to_vector()
        face.denormalize_gaze_vector()

    @torch.no_grad()
    def _run_gaze360_model(self,face:Face)-> None:
        image = self._transform(face.normalized_image).unsqueeze(0)
        fimg = cv2.resize(face.normalized_image, (448, 448))/255.0
        fimg = fimg.transpose(2, 0, 1)
        image = torch.from_numpy(fimg).type(torch.FloatTensor).unsqueeze(0)
        device = torch.device(self._config.device)
        data = edict()
        data.face = image.to(device)
        
        prediction,_ = self.gaze_360(data)
        prediction = prediction.cpu().numpy()
        face.normalized_gaze_angles = prediction[0]
        
        face.angle_to_vector()
        face.denormalize_gaze_vector()


if __name__ == "main":
    GazeEstimator(load_config())
