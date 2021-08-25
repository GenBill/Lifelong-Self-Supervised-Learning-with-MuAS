from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader

class shared_model_res18(nn.Module):

    def __init__(self):
        super().__init__()
        res_net = models.resnet18(pretrained = False)
        self.tmodel = nn.Module()
        self.tmodel.add_module('features', nn.Sequential(*(list(res_net.children())[:-1])) )
        self.tmodel.add_module('classifier', res_net.fc )
        self.reg_params = {}

    def forward(self, x):
        return self.tmodel(x)

class mlptail(nn.Module):
    def __init__(self, in_features, mid_features, out_features):
        super(mlptail, self).__init__()
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, mid_features), 
            nn.LeakyReLU(),
            nn.Linear(mid_features, mid_features), 
            nn.LeakyReLU(),
            nn.Linear(mid_features, out_features)
        )
        
    def forward(self, x):
        # x -> x.view(-1, self.in_features)
        return self.fc(x.view(-1, self.in_features))

net1 = models.resnet18(pretrained=True)
net2 = models.alexnet(pretrained=True)
net3 = mlptail(10,10,2)

print(net1)
