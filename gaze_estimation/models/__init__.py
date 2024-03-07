import importlib
import os
import sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import timm
import torch
# from omegaconf import DictConfig
import yacs.config
from . import model,eth_basic,swin,coatnet,PVT,twins



# def create_model(config: yacs.config.CfgNode) -> torch.nn.Module:
#     dataset_name = config.mode.lower()
#     module = importlib.import_module(
#         f'gaze_estimation.models.{dataset_name}.{config.model.name}')
#     model = module.Model(config)
#     device = torch.device(config.device)
#     model.to(device)
#     return model

def create_model(mode,model_name) -> torch.nn.Module:
    
    if model_name == "swin":
        model = swin.GazeNet()
    elif model_name == "swin_peg":
        model = swin.GazeNet()
    elif model_name == "swin_full_peg":
        model = swin.GazeNet()
    elif model_name == "coatnet":
        model = coatnet.GazeNet()
    elif model_name == "PVT":
        model = PVT.GazeNet()
    elif model_name == "twins":
        model = twins.GazeNet()


    else:
        raise ValueError

    return model
