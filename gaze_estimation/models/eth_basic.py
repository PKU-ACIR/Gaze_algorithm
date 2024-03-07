import torch
import torch.nn as nn 
import numpy as np
import math
import copy
from gaze_estimation.models.resnet_eth import resnet50,resnet18


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.base_model = resnet50(pretrained=True)

        self.feed = nn.Sequential(
            nn.Linear(2048, 2)
        )
            
        self.loss_op = nn.L1Loss()
        



    def forward(self, x_in):
        feature = self.base_model(x_in["face"])

        feature = feature.view(feature.size(0), -1)
        gaze = self.feed(feature)
        return gaze

    def loss(self, x_in, label):
        gaze = self.forward(x_in)
        loss = self.loss_op(gaze, label) 
        
        return loss

