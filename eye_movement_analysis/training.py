import scipy.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import h5py
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader

from probabilistic_model import RNN
from sklearn.model_selection import StratifiedShuffleSplit
import os
from tensorboardX import SummaryWriter



def test(num):
    outfile = open("train_log/rnn_wt_in.log",'w')
    batch_size = 256
    criterion = nn.BCELoss()
    rnn = RNN(5,10).cuda()
    
    data_test = np.load('data_test.npy')
    label_test = np.load('label_test.npy')
    test_dataset = MyDataset(data_test,label_test)
    test_loader = DataLoader(test_dataset,batch_size,num_workers = 1)
    rnn.eval()
    test_losses = []
    TP,TN,FN,FP = 0,0,0,0
    for i in range():

        rnn.load_state_dict(torch.load('pro_model/rnn{}.pt'.format(i)))
        with torch.no_grad():
            for batch_count, (test_data, test_label) in enumerate(test_loader):
                test_data = test_data[:,:,0:4].float().cuda()
                test_label = test_label.float().cuda()           
                h = torch.zeros([1,batch_size,10]).float().cuda()
                pred,h = rnn(test_data,h)
                pred = F.sigmoid(pred).squeeze(0)
                loss = criterion(pred,test_label)

                TP += ((pred > 0.5) & (test_label == 1)).cpu().sum().item()
                TN += ((pred < 0.5) & (test_label == 0)).cpu().sum().item()
                FN += ((pred < 0.5) & (test_label == 1)).cpu().sum().item()
                FP += ((pred > 0.5) & (test_label == 0)).cpu().sum().item()

                
                test_losses.append(np.mean(loss.cpu().numpy()))
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            test_loss_avg = np.mean(test_losses)
            log = 'model:{}      loss: {}   precision: {}    recall {}    F1 {}    acc {} '.format(i,test_loss_avg,p,r,F1,acc)
            print(log)
            outfile.write(log + "\n")

def thres_test(thres):
    data_test = np.load('datasets/data_test.npy')
    label_test = np.load('datasets/label_test.npy')
    TP,TN,FN,FP = 0,0,0,0
    total_v = np.sqrt(data_test[:,2] ** 2 + data_test[:,3] ** 2)

    TP = ((total_v > thres) & (label_test == 1)).sum()
    TN = ((total_v < thres) & (label_test == 0)).sum()
    FN = ((total_v < thres) & (label_test == 1)).sum()
    FP = ((total_v > thres) & (label_test == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('precision: {}    recall {}    F1 {}    acc {} '.format(p,r,F1,acc))

def thres_test_2(thres1,thres2,outfile):
    data_test = np.load('datasets/data_test.npy')
    label_test = np.load('datasets/label_test.npy')
    TP,TN,FN,FP = 0,0,0,0
    v1 = np.abs(data_test[:,2])
    v2 = np.abs(data_test[:,3])

    TP = (np.logical_and(v1 > thres1, v2 > thres2) & (label_test == 1)).sum()
    TN = (np.logical_or(v1 <= thres1, v2 <= thres2) & (label_test == 0)).sum()
    FN = (np.logical_or(v1 <= thres1, v2 <= thres2) & (label_test == 1)).sum()
    FP = (np.logical_and(v1 > thres1, v2 > thres2) & (label_test == 0)).sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    outfile.write('i:{}  j: {} precision: {}    recall {}    F1 {}    acc {} \n'.format(thres1,thres2,p,r,F1,acc))
    



class MyDataset(Dataset): 
  def __init__(self, data,label):

    # Read source data 
    self.data = data
    self.label = label




  def __len__(self):

    return len(self.data)


  def __getitem__(self, idx):


    return self.data[idx], self.label[idx]

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pt, target):
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def train():
    data_train = np.load('datasets/data_train.npy')
    label_train = np.load('datasets/label_train.npy')

    data_test = np.load('datasets/data_train.npy')
    label_test = np.load('datasets/label_train.npy')
    writer = SummaryWriter("train_board/rnn_random_new_slow")

    train_dataset = MyDataset(data_train,label_train)
    test_dataset = MyDataset(data_test,label_test)

    









    epochs = 40
    lr = 0.001
    distance = 700
    batch_size = 256

    rnn = RNN(5,10).cuda()
    criterion = BCEFocalLoss()
    optim = torch.optim.Adam(rnn.parameters(),lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR( 
                    optim, 
                    step_size=10, 
                    gamma=0.5
                )
    outfile = open("train_log/rnn_random_new_slow.log",'w')

    train_loader = DataLoader(train_dataset,batch_size,num_workers = 1)
    test_loader = DataLoader(test_dataset,batch_size,num_workers = 1)
    

    loss_items = []
    test_loss_items = []
    test_acc_items = []
    board_count = 0
    test_count = 0

    model_num = 0
    

    for i in range(epochs):
        running_loss = 0
        for j, (data, label) in enumerate(train_loader):
            data = data.float().cuda()
            #data = data[:,:,0:4].float().cuda()
            label = label.float().cuda()
            h = torch.zeros([1,label.shape[0],10]).float().cuda()
            pred,h = rnn(data,h)
            pred = F.sigmoid(pred).squeeze(0)
            loss = criterion(pred,label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss
            
            if j % 1000 == 0:
                writer.add_scalar('Train/Loss', running_loss/1000, board_count)
                board_count += 1
                running_loss = 0
                test_losses = []

                TP,TN,FN,FP = 0,0,0,0
                with torch.no_grad():
                    for batch_count, (test_data, test_label) in enumerate(test_loader):
                        test_data = test_data.float().cuda()
                        #test_data = test_data[:,:,0:4].float().cuda()
                        test_label = test_label.float().cuda()
                        h = torch.zeros([1,test_label.shape[0],10]).float().cuda()
                        pred,h = rnn(test_data,h)
                        pred = F.sigmoid(pred).squeeze(0)
                        loss = criterion(pred,test_label)

                        TP += ((pred > 0.5) & (test_label == 1)).cpu().sum().item()
                        TN += ((pred < 0.5) & (test_label == 0)).cpu().sum().item()
                        FN += ((pred < 0.5) & (test_label == 1)).cpu().sum().item()
                        FP += ((pred > 0.5) & (test_label == 0)).cpu().sum().item()

                
                        test_losses.append(np.mean(loss.cpu().numpy()))
                    p = TP / (TP + FP)
                    r = TP / (TP + FN)
                    F1 = 2 * r * p / (r + p)
                    acc = (TP + TN) / (TP + TN + FP + FN)
                    test_loss_avg = np.mean(test_losses)
                    test_count += 1
                    writer.add_scalar('Test/Loss', test_loss_avg, test_count)
                    writer.add_scalar('Precision', p, test_count)
                    writer.add_scalar('Recall', r, test_count)
                    writer.add_scalar('F1_score', F1, test_count)
                    writer.add_scalar('Accuracy', acc, test_count)

                    log = 'epoch:{}    eval_step:{}   loss: {}   precision: {}    recall {}    F1 {}    acc {} '.format(i,j/1000,test_loss_avg,p,r,F1,acc)
                    print(log)
                    outfile.write(log + "\n")

            if j % 2000 == 0:
                if not os.path.exists('pro_model_random_new_slow/'):
                    os.makedirs('pro_model_random_new_slow/')
                save_path = "pro_model_random_new_slow/rnn{}.pt".format(model_num)
                model_num += 1
                torch.save(rnn.state_dict(),save_path)

if __name__ == '__main__':
    #train()

    outfile = open("thres_test.txt",'w')
    for i in range(300):
        for j in range(300):

            thres_test_2(i,j,outfile)

    




