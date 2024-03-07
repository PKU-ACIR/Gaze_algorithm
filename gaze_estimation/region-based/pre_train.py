
import sys, os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import yaml
import cv2

from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse
from model import RSN,GazeNet


def random_selection():


def main(config):
    # ===============================> Setup <================================

    

    print("===> Model building <===")
    rsn = RSN()
    rsn.train();rsn.cuda()

    
    # Pretrain
    """pretrain = config.pretrain
    if pretrain.enable and pretrain.device:
        net.load_state_dict(
                torch.load(
                    pretrain.path, 
                    map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"}
                )
            )
    elif pretrain.enable and not pretrain.device:
        net.load_state_dict(
                torch.load(pretrain.path)
            )"""
        
    batch_size = 90
    print("===> optimizer building <===")
    optimizer = optim.Adam(
                    rsn.parameters(),
                    lr=10e-4, 
                    betas=(0.9,0.95)
                )
    params.decay = 0.1
    params.decay_step = 10
    scheduler = optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=params.decay_step, 
                    gamma=params.decay
                )

    """if params.warmup:
        scheduler = GradualWarmupScheduler(
                    optimizer, 
                    multiplier=1, 
                    total_epoch=params.warmup, 
                    after_scheduler=scheduler


                )"""

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint/{savename}")
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # =======================================> Training < ==========================
    print("===> Training <===")
    length = len(dataset); total = length * params.epoch
    timer = ctools.TimeCounter(total)


    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch+1):
            for i, (data, anno) in enumerate(dataset):

                # ------------------forward--------------------
                data["face"] = data["face"].cuda()

                anno = anno.cuda()
 
                

                # -----------------backward--------------------
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                rest = timer.step()/3600

                # -----------------loger----------------------
                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " +\
                          f"[{i}/{length}] " +\
                          f"loss:{loss}" +\
                          f"lr:{ctools.GetLR(optimizer)} "+\
                          f"rest time:{rest:.2f}h"

                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()

            scheduler.step()

            if epoch % save.step == 0:
                torch.save(
                        net.state_dict(), 
                        os.path.join(savepath, f"Iter_{epoch}_{save.model_name}.pt")
                    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    parser.add_argument('-p', '--person', type=int,
                        help='The tested person.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    config = config.train
    config.person = args.person
    
    print("=====================>> (Begin) Training params << =======================")

    print(ctools.DictDumps(config))

    print("=====================>> (End) Traning params << =======================")

    main(config)

