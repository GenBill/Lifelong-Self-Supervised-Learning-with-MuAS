from __future__ import print_function, division
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import math
import os
import copy
import argparse
import random
import numpy as np
import warnings
import torch.utils.data as data
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyTemplateNet(nn.Module):
    def __init__(self):
        super(MyTemplateNet, self).__init__() # 第一句话，调用父类的构造函数
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

net = MyTemplateNet().to(device) # 构造模型


Frost_list = []
def Params_Listmaker(params, Frost_list):
    # 具体实现细节还要改！
    # for layer in params:
        Norm_list = np.array([])
        for p in tqdm(params):
            # Norm_list.append(sum(abs(p)).item())
            print(p)
            np.append(Norm_list, torch.mean(torch.abs(p)).item())
        sort_index = np.argsort(Norm_list)
        print(Norm_list)
        # num_discard = len(sort_index)//10 + len(sort_index)%10
        num_discard = len(sort_index)//1 # + len(sort_index)%1
        for i in range(num_discard):
            # Frost : params[index[i]]
            Frost_list.append(params[sort_index[i]])

def Params_Frost(params, Frost_list):
    Params_Listmaker(params, Frost_list)
    for ice in Frost_list:
        ice.detach_()       # Frost ! 

def Params_Lava(Frost_list):
    for ice in Frost_list:
        ice.undetach_()     # Lava !
    Frost_list = []

# list(MyLinear_mod(2, 3).parameters())
# print(list(net.parameters()))

# X_torch = torch.from_numpy(X).float().to(device)
# Y_torch = torch.from_numpy(Y).unsqueeze(1).float().to(device)

'''
params = list(net.named_parameters())   # get the index by debuging
for i in range(len(params)):
    print(i, params[i][0])      # name
    print(params[i][1].data)    # data
'''



optimizer = optim.SGD(
    [
        {'params': net.parameters(), 'lr': 0., 'momentum': 0., 'weight_decay': 0.}
    ]   , lr=1e-3, momentum=0.8, weight_decay=0.001
)
criterion = nn.MSELoss(reduction='mean')

X_torch = torch.rand(1,2).float().to(device)
Y_torch = torch.rand(1,1).float().to(device)

pred = net(X_torch)
loss = criterion(pred, Y_torch)

optimizer.zero_grad()
loss.backward()
print('this')
print(net.fc1.weight.grad)
net.fc1.weight.grad = net.fc1.weight.grad-1
print(net.fc1.weight.grad)

optimizer.step()


params = list(net.named_parameters())   # get the index by debuging
for params in net.parameters():
    print(params.data)

print('--------2021 08 19--------')
for params in net.parameters():
    print(params.name)