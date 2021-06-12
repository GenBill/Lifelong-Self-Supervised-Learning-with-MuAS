from agent.train_rota import rotaloader
from agent.train_patch import patchloader
from agent.train_jigpa import jigpaloader
from agent.train_jigro import jigroloader
from agent.train_step import Step

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.transforms as transforms
from torchvision import datasets, models
import matplotlib.pyplot as plt

import os
import argparse
import random
import numpy as np
import warnings

from PIL import Image
plt.ion()  # interactive mode
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--netCont', default='', help="path to net (for continue training)")
parser.add_argument('--rotaCont', default='', help="path to fc_layer")
parser.add_argument('--jigroCont', default='', help="path to fc_layer")
parser.add_argument('--patchCont', default='', help="path to fc_layer")
parser.add_argument('--jigpaCont', default='', help="path to fc_layer")
# parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args(args=[])
opt.lr = 1e-3
opt.manualSeed = 2077
# opt.netCont = './models/net_epoch_56.pth'


out_dir = './StepRot/models'
log_out_dir = './StepRot/logs'

try:
    os.makedirs(out_dir)
    os.makedirs(log_out_dir)
except OSError:
    pass


file = open("{}/training_logs.txt".format(log_out_dir), "w+")
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000) 
file.write("Random Seed: {} \n".format(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


cudnn.benchmark = True
image_size = (224, 224)
data_root = '../Kaggle265'     # '../Dataset/Kaggle265'
batch_size = 512
num_epochs = 50

# Initiate dataset and dataset transform
data_pre_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
    ]),
}
data_post_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.6086, 0.4920, 0.4619], std=[0.2577, 0.2381, 0.2408])
    ]),
}
patch_dim = 96
gap = 6
jitter = 6

loader_rota = rotaloader(data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_patch = patchloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_jigpa = jigpaloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_jigro = jigroloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file.write("using " + str(device) + "\n")
file.flush()

# Model Initialization
# 仅支持 Res-Net !!!
model_all = models.resnet18(pretrained=True)
num_ftrs = model_all.fc.in_features

def make_MLP(input_ftrs, hidden_ftrs, output_ftrs, layers=1):
    modules_list = []
    modules_list.append(nn.Flatten())
    if layers==1:
        modules_list.append(nn.Linear(input_ftrs, output_ftrs))
    else:
        modules_list.append(nn.Linear(input_ftrs, hidden_ftrs))
        for _ in range(layers-2):
            modules_list.append(nn.LeakyReLU())
            modules_list.append(nn.Linear(hidden_ftrs, hidden_ftrs))
        modules_list.append(nn.LeakyReLU())
        modules_list.append(nn.Linear(hidden_ftrs, output_ftrs))
    
    modules_list.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*modules_list)

model_ft = nn.Sequential(*(list(model_all.children())[:-1]))
fc_rota = make_MLP(num_ftrs, num_ftrs, output_ftrs=4, layers=1)
fc_patch = make_MLP(2*num_ftrs, 2*num_ftrs, output_ftrs=8, layers=2)
fc_jigpa = make_MLP(4*num_ftrs, 4*num_ftrs, output_ftrs=24, layers=4)
### 警告：类间距不同，不可粗暴计算损失，需要重写损失函数 ###
fc_jigro = make_MLP(4*num_ftrs, 4*num_ftrs, output_ftrs=96, layers=8)   # output_ftrs=24

if torch.cuda.device_count() > 1: 
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_ft = nn.DataParallel(model_ft)
    fc_rota = nn.DataParallel(fc_rota)
    fc_patch = nn.DataParallel(fc_patch)
    fc_jigpa = nn.DataParallel(fc_jigpa)
    fc_jigro = nn.DataParallel(fc_jigro)

model_ft = model_ft.to(device)
fc_rota = fc_rota.to(device)
fc_patch = fc_patch.to(device)
fc_jigpa = fc_jigpa.to(device)
fc_jigro = fc_jigro.to(device)

# Load state : model & fc_layer
def loadstate(model, fc_layer, net_Cont, fc_Cont, device, file):
    if net_Cont != '':
        model.load_state_dict(torch.load(net_Cont, map_location=device))
        print('Loaded model state ...')
        file.write('Loaded model state ...')

    if fc_Cont != '':
        fc_layer.load_state_dict(torch.load(fc_Cont, map_location=device))
        print('Loaded fc_layer state ...')
        file.write('Loaded fc_layer state ...')

loadstate(model_ft, fc_rota, opt.netCont, opt.rotaCont, device, file)

# Model trainer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(
    [
        {'params': model_ft.parameters(), 'lr': opt.lr, 'weight_decay': 1e-8},
        {'params': fc_rota.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
        {'params': fc_patch.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
        {'params': fc_jigpa.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
        {'params': fc_jigro.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
    ]
)
optimizer_rota = optim.Adagrad(
    [
        {'params': model_ft.parameters(), 'lr': opt.lr, 'weight_decay': 1e-8},
        {'params': fc_rota.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
    ]
)
optimizer_patch = optim.Adagrad(
    [
        {'params': model_ft.parameters(), 'lr': opt.lr, 'weight_decay': 1e-8},
        {'params': fc_patch.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
    ]
)
optimizer_jigpa = optim.Adagrad(
    [
        {'params': model_ft.parameters(), 'lr': opt.lr, 'weight_decay': 1e-8},
        {'params': fc_jigpa.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
    ]
)
optimizer_jigro = optim.Adagrad(
    [
        {'params': model_ft.parameters(), 'lr': opt.lr, 'weight_decay': 1e-8},
        {'params': fc_jigro.parameters(), 'lr': opt.lr, 'weight_decay': 1e-12},
    ]
)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.2)

### 警告：施工现场 ###

# model_ft, fc_rota = Step(
#     model_ft, fc_rota, 'rota',
#     myloader_rota, criterion, optimizer, scheduler,
#     device, out_dir, file, 10, num_epochs
# )
print('Training ...')

# 'rota' , 'patch' , 'jigpa' , 'jigro'
model_ft, fc_jigro = Step(
    model_ft, fc_jigro, 'jigro',
    loader_jigro, criterion, optimizer, scheduler,
    device, out_dir, file, saveinterval=2, num_epochs=num_epochs
)

