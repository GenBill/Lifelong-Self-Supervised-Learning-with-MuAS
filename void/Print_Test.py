from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader

this_Epoch = 10000
epoch_num = 10000
'''
for epoch in range(epoch_num) :
    if (epoch+1) % 100 == 0:
        print('./TimHu/models256/'+'Epoch_'+str(this_Epoch + epoch)+'.pth')
'''
for epoch in range(1,epoch_num) :
    if epoch % 1000 == 0:
        print('./TimHu/models256/'+'Epoch_'+str(this_Epoch + epoch)+'.pth')


net = models.resnet18(pretrained=True)
params_net = list(net.named_parameters())
print(params_net[-1][0])
print(params_net[-2][0])


Frost_Percent = 0.1
print(str(int(Frost_Percent*100)))