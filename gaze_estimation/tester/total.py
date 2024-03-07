import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
#import models.model as model
import models.swin as model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools as ctools, gtools as gtools
import argparse
from tensorboardX import SummaryWriter



def main(train, test):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load
    writer = SummaryWriter(load.tensor_path)

    
 

    # ===============================> Read Data <=========================
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 32, num_workers=0, shuffle=False)

    modelpath = os.path.join(train.save.metapath,
                                train.save.folder, f"checkpoint/")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{test.savename}")

  
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):

        print(f"Test {saveiter}")

        net = model.GazeNet()
        """
        statedict = torch.load(
                        "models/GazeTR-H-ETH.pt", 
                        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
                    )
        """
        statedict = torch.load(
                        os.path.join(modelpath, 
                            f"Iter_{saveiter}_{train.save.model_name}.pt"), 
                        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
                    )
        

        net.cuda(); net.load_state_dict(statedict); net.eval()

        length = len(dataset); accs = 0; count = 0
        personal_accs = 0;personal_count = 0
        personal_yaw = 0;personal_pitch = 0
        yaw=0;pitch=0
        current = "rec000"

        logname = f"{saveiter}.log"

        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")
        name =  ["rec000"]
        

        with torch.no_grad():
            for j, (data, label) in enumerate(dataset):
                gts = label.cuda()
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                if not test.personal:
                    
                    gazes = net(data)

                    for k, gaze in enumerate(gazes):

                        gaze = gaze.cpu().detach().numpy()
                        gt = gts.cpu().numpy()[k]
  
                        count += 1
                        accs += gtools.angular(
                                gtools.gazeto3d(gaze),
                                gtools.gazeto3d(gt)
                            )
                        yaw += abs((gaze[1]-gt[1])*180.0/np.pi)
                        pitch += abs((gaze[0]-gt[0])*180.0/np.pi)

                        gaze = [str(u) for u in gaze] 
                        gt = [str(u) for u in gt] 
                        log = name + [",".join(gaze)] + [",".join(gt)]
                        outfile.write(" ".join(log) + "\n")
                    loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
                    writer.add_scalar("pitch", pitch/count, saveiter)
                    writer.add_scalar("yaw", yaw/count, saveiter)
                    writer.add_scalar("total", accs/count, saveiter)
                    continue


                names =  np.array(data.name)
                if not np.all(names == current):
                    count += personal_count
                    accs += personal_accs
                    yaw += personal_yaw
                    pitch += personal_pitch
                    
                    loger = f"[{saveiter}] Total Num: {personal_count}, avg: {personal_accs/personal_count}, Name:{current}"
                    writer.add_scalar(current, personal_accs/personal_count, saveiter)
                    writer.add_scalar(current+"yaw", personal_yaw/personal_count, saveiter)
                    writer.add_scalar(current+"pitch", personal_pitch/personal_count, saveiter)
                    personal_accs = 0;personal_count = 0
                    personal_yaw = 0;personal_pitch = 0
                    current = names[-1]
                    continue
                

                


                gts = label.cuda()
           
                gazes = net(data)

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]
  
                    personal_count += 1
                    personal_accs += gtools.angular(
                                gtools.gazeto3d(gaze),
                                gtools.gazeto3d(gt)
                            )
                    personal_yaw += abs((gaze[1]-gt[1])*180.0/np.pi)
                    personal_pitch += abs((gaze[0]-gt[0])*180.0/np.pi)

                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile.write(" ".join(log) + "\n")

            
            if test.personal:
                count += personal_count
                accs += personal_accs
            
                loger = f"[{saveiter}] Total Num: {personal_count}, avg: {personal_accs/personal_count}, Name:{current}"
                writer.add_scalar(current, personal_accs/personal_count, saveiter)
                writer.add_scalar(current+"yaw", personal_yaw/personal_count, saveiter)
                writer.add_scalar(current+"pitch", personal_pitch/personal_count, saveiter)
                loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
                writer.add_scalar("pitch", pitch/count, saveiter)
                writer.add_scalar("yaw", yaw/count, saveiter)
                writer.add_scalar("total", accs/count, saveiter)
            

            outfile.write(loger)
            print(loger)
        outfile.close()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))
    """
    train_conf = edict(yaml.load(open("gaze_estimation/config/train/config_eth.yaml"), Loader=yaml.FullLoader))
    test_conf = edict(yaml.load(open("gaze_estimation/config/test/config_eth.yaml"), Loader=yaml.FullLoader))
    """
    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf.test)

 
