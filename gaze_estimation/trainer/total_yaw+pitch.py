import sys,os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import models.trans_yaw_pitch as model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import yaml
import cv2
import ctools as ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse
from tensorboardX import SummaryWriter

def main(config):

    #  ===================>> Setup <<=================================

    dataloader = importlib.import_module("reader." + config.reader)

    torch.cuda.set_device(config.device) 
    cudnn.benchmark = True

    data = config.data
    save = config.save
    params = config.params
    writer = SummaryWriter('log_res50_y_p')
    

    print("===> Read data <===")

    if data.isFolder:
        data, _ = ctools.readfolder(data)

    dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=1
                )


    print("===> Model building <===")
    net = model.Model()
    net.train(); net.cuda()


    # Pretrain 
    pretrain = config.pretrain

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
                )


    print("===> optimizer building <===")
    optimizer = optim.Adam(
                    net.parameters(),
                    lr=params.lr, 
                    betas=(0.9,0.999)
                )
  
    scheduler = optim.lr_scheduler.StepLR( 
                    optimizer, 
                    step_size=params.decay_step, 
                    gamma=params.decay
                )

    if params.warmup:
        scheduler = GradualWarmupScheduler( 
                        optimizer, 
                        multiplier=1, 
                        total_epoch=params.warmup, 
                        after_scheduler=scheduler
                    )

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
 
    # =====================================>> Training << ====================================
    print("===> Training <===")

    length = len(dataset)
    total = length * params.epoch
    timer = ctools.TimeCounter(total)


    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor)
    r_loss_p = 0
    r_loss_y = 0
    board_count = 0


    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch+1):
            for i, (data, label,cont_label) in enumerate(dataset):

                # -------------- forward -------------
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()




                label = label.cuda() 
                cont_label = cont_label.cuda()

                loss_pitch,loss_yaw = net.loss(data, label,cont_label)

                # -------------- Backward ------------
                loss_seq = [loss_pitch, loss_yaw]
                grad_seq = [torch.tensor(1.0).cuda() for _ in range(len(loss_seq))]
                

                optimizer.zero_grad()
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer.step()
                rest = timer.step()/3600

                r_loss_p += loss_pitch.item()
                r_loss_y += loss_yaw.item()

                if i % 200 == 0:
                    writer.add_scalar('Train/Loss_Pitch', r_loss_p/200, board_count)
                    writer.add_scalar('Train/Loss_Yaw', r_loss_y/200, board_count)
                    board_count += 1
                    r_loss_p = 0
                    r_loss_y = 0

                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch+30}]: " + \
                          f"[{i}/{length}] " +\
                          f"loss_pitch:{loss_pitch} " +\
                          f"loss_yaw:{loss_yaw} " +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"rest time:{rest:.2f}h"
                    
                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()

            scheduler.step()

            if epoch % save.step == 0:
                torch.save(
                        net.state_dict(), 
                        os.path.join(
                            savepath, 
                            f"Iter_{epoch+30}_{save.model_name}.pt"
                            )
                        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    #config = edict(yaml.load(open("gaze_estimation/config/train/config_eth.yaml"), Loader=yaml.FullLoader))
    print("=====================>> (Begin) Training params << =======================")
    print(ctools.DictDumps(config))
    print("=====================>> (End) Traning params << =======================")

    main(config.train)

