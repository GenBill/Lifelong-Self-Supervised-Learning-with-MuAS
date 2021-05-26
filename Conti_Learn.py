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

plt.ion()  # interactive mode
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_data(ax, X, Y):
    plt.axis('off')
    ax.scatter(X[:, 0], X[:, 1], s=1, c=Y, cmap='bone')

from sklearn.datasets import make_moons
X, Y = make_moons(n_samples=2000, noise=0.1)
# %matplotlib notebook
# %matplotlib inline

x_min, x_max = -1.5, 2.5
y_min, y_max = -1, 1.5

'''
fig, ax = plt.subplots(1, 1, facecolor='#4B6EA9')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plot_data(ax, X, Y)
plt.show()
'''

# plot the decision boundary of our classifier
def plot_decision_boundary(ax, X, Y, classifier):
    # Define the grid on which we will evaluate our classifier
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),
                        np.arange(y_min, y_max, .01))

    # the grid for plotting the decision boundary should be now made of tensors.
    # to_forward = np.array(list(zip(xx.ravel(), yy.ravel())))
    to_forward = torch.from_numpy(np.array(list(zip(xx.ravel(), yy.ravel())))).float().to(device)
    # forward pass on the grid, then convert to numpy for plotting
    Z = classifier.forward(to_forward)
    # Z = torch.argmax(Z, dim=1).unsqueeze(1)
    Z = Z.reshape(xx.shape).to('cpu')

    # plot contour lines of the values of our classifier on the grid
    ax.contourf(xx, yy, Z>0.5, cmap='Blues')
    
    # then plot the dataset
    plot_data(ax, X, Y)


class MyTemplateNet(nn.Module):
    def __init__(self):
        super(MyTemplateNet, self).__init__() # 第一句话，调用父类的构造函数
        self.fc1 = nn.Linear(2, 8)
        # self.fc2 = nn.Linear(8, 8)
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

X_torch = torch.from_numpy(X).float().to(device)
Y_torch = torch.from_numpy(Y).unsqueeze(1).float().to(device)

optimizer = optim.Adam(
    [
        {'params': (p for name, p in net.named_parameters() if 'weight' in name), 'lr': 1e-1, 'momentum': 0., 'weight_decay': 1e-4},
        {'params': (p for name, p in net.named_parameters() if 'bias' in name), 'lr': 1e-1, 'momentum': 0., 'weight_decay': 0.}
    ]   # , lr=1e-3, momentum=0.8, weight_decay=0.001
)
criterion = nn.MSELoss(reduction='mean')

#net.fc1.weight


'''
params = list(net.named_parameters())   # get the index by debuging
print(len(params))
for i in range(4):
    print(i, params[i][0])      # name
    print(params[i][1].data)    # data


Params_Frost(net.parameters(), Frost_list)
pred = net(X_torch[0:16])
loss = criterion(pred, Y_torch[0:16])
optimizer.zero_grad()
loss.backward()
optimizer.step()
Params_Lava(Frost_list)
# print(2, 'Done!')
# print(list(net.parameters()))

pred = net(X_torch[0:16])
loss = criterion(pred, Y_torch[0:16])
optimizer.zero_grad()
loss.backward()
optimizer.step()
# print(3, 'Done!')
# print(list(net.parameters()))
# print('Done!')
'''