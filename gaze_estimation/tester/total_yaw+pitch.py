import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import models.trans_yaw_pitch as model
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
import time

def main(train, test):

    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    data = test.data
    load = test.load
    writer = SummaryWriter('tensorboard/log_res50_y_p_mpii')
    

    # ===============================> Read Data <=========================
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 

    print(f"==> Test: {data.label} <==")
    dataset = reader.loader(data, 128, num_workers=1, shuffle=False)

    modelpath = os.path.join(train.save.metapath,
                                train.save.folder, f"checkpoint/")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{test.savename}"+str(time.time()))

  
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Test <=============================

    begin = load.begin_step; end = load.end_step; step = load.steps

    for saveiter in range(begin, end+step, step):

        print(f"Test {saveiter}")

        net = model.Model()

        statedict = torch.load(
                        os.path.join(modelpath, 
                            f"Iter_{saveiter}_{train.save.model_name}.pt"), 
                        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
                    )
        """
        statedict = torch.load(
                        "models/Iter_30_trans6.pt", 
                        map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
                    )
        """
        net.cuda(); net.load_state_dict(statedict); net.eval()
        

        length = len(dataset); accs = 0; count = 0
        
        
        personal_accs = 0;personal_count = 0
        personal_yaw = 0;personal_pitch = 0
        current = "subject0000"

        logname = f"{saveiter}.log"

        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")
        

        idx_tensor = [idx for idx in range(90)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda()

        softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            for j, (data, label,cont_label) in enumerate(dataset):
                names =  np.array(data["name"])

                """if not np.all(names == current):
                    count += personal_count
                    accs += personal_accs
                    loger = f"[{saveiter}] Total Num: {personal_count}, avg: {personal_accs/personal_count}, Name:{current}"
                    writer.add_scalar(current, personal_accs/personal_count, saveiter)
                    writer.add_scalar(current+"yaw", personal_yaw/personal_count, saveiter)
                    writer.add_scalar(current+"pitch", personal_pitch/personal_count, saveiter)
                    personal_accs = 0;personal_count = 0
                    personal_yaw = 0;personal_pitch = 0
                    current = names[-1]
                    continue
                """

                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                names =  data["name"]
           
                pitchs,yaws = net(data)

                for k in range(len(pitchs)):

                    pitch = pitchs[k].unsqueeze(0)
                    yaw = yaws[k].unsqueeze(0)
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                    gaze = [pitch_predicted,yaw_predicted]

                    c_l = cont_label.numpy()[k]

                    gt = [c_l[0]* np.pi/180.0,c_l[1]* np.pi/180.0]

                    personal_count += 1                
                    personal_accs += gtools.angular(
                                gtools.gazeto3d(np.array(gaze)),
                                gtools.gazeto3d(np.array(gt))
                            )
                    personal_yaw += abs((yaw_predicted-gt[1])*180.0/np.pi)
                    personal_pitch += abs((pitch_predicted-gt[0])*180.0/np.pi)
            
                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile.write(" ".join(log) + "\n")

            count += personal_count
            accs += personal_accs
            """
            loger = f"[{saveiter}] Total Num: {personal_count}, avg: {personal_accs/personal_count}, Name:{current}"
            writer.add_scalar(current, personal_accs/personal_count, saveiter)
            writer.add_scalar(current+"yaw", personal_yaw/personal_count, saveiter)
            writer.add_scalar(current+"pitch", personal_pitch/personal_count, saveiter)
            """
            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            writer.add_scalar("total",accs/count,saveiter)
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
    test_conf = edict(yaml.load(open("gaze_estimation/config/test/config_mpii.yaml"), Loader=yaml.FullLoader))
    """
    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(test_conf))
    print("=======================>(End) Config for test<======================")

    main(train_conf.train, test_conf.test)

 
