import torch
import torch.nn as nn


input_size = [224,224]
class RSN(nn.module):
    def __init__(self):
        super(RSN, self).__init__()
        self.select_num = 3
        self.location_pool = 49
        self.output= 3 * 49
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.output)

    def forward(self, x_in):
        base_out = self.base_model(x_in["face"])
        base_out = base_out.view(input.size(0),self.select_num,self.location_pool)
        

class GazeNet(nn.module):
    def __init__(self):
        super(GazeNet,self).__init__()
        self.select_num = 3
        self.base_model = resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(3*self.select_num, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.loc_conv1 = nn.Conv2d(self.select_num,3,kernel_size= 4,stride = 4,padding=0)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.loc_conv2 = nn.Conv2d(self.select_num,1,kernel_size= 2,stride = 1,padding=0)
        self.fc1 = nn.Linear(49,250)
        self.fc2 = nn.Linear(250,1250)
        self.final_fc = nn.Linear(1250+512,2)

    def forward(self,images,locs):
        features = self.base_model.extract(images)
        loc_f = self.loc_conv1(locs)
        loc_f = self.bn1(loc_f)
        loc_f = self.relu(loc_f)
        loc_f = self.conv2(loc_f)
        loc_f = self.relu(loc_f)
        loc_f = self.fc1(loc_f)
        loc_f = self.relu(loc_f)
        loc_f = self.fc2(loc_f)

        all_features = nn.contenate(features,loc_f)
        out = self.final_fc(all_features)











