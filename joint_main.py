from agent.train_step import LaStep
from agent.train_step import JointStep
from agent import *

import torch
import torch.nn as nn
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
parser.add_argument('--cuda', default='', help="cuda : ?")
parser.add_argument('--powerword', default='rota', help="Power Word, decide what to do")

parser.add_argument('--lr_net', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--weight_net', type=float, default=1e-8, help="weight decay")
parser.add_argument('--lr_fc', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--weight_fc', type=float, default=1e-8, help="weight decay")

parser.add_argument('--netCont', default='', help="path to net (for continue training)")
parser.add_argument('--plainCont', default='', help="path to plain fc_layer (for continue training)")
parser.add_argument('--rotaCont', default='', help="path to fc_layer rota")
parser.add_argument('--jigroCont', default='', help="path to fc_layer patch")
parser.add_argument('--patchCont', default='', help="path to fc_layer jigpa")
parser.add_argument('--jigpaCont', default='', help="path to fc_layer jigro")
parser.add_argument('--contraCont', default='', help="path to fc_layer jigro")
# parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--joint', type=int, default=1, help="joint on")
parser.add_argument('--pretrain', type=int, default=1, help="pretrain on")

# opt = parser.parse_args(args=[])
opt = parser.parse_args()
opt.manualSeed = 2077
# opt.netCont = './models/net_epoch_56.pth'

if opt.joint==0:
    if opt.pretrain==1:
        out_dir = './Joint/Wopre/models'
        log_out_dir = './Joint/Wopre/Wopre'
    else :
        out_dir = './Joint/Wono/models'
        log_out_dir = './Joint/Wono/Wono'

elif opt.joint==1:
    if opt.pretrain==1:
        out_dir = './Joint/Jopre/models'
        log_out_dir = './Joint/Jopre/Jopre'
    else :
        out_dir = './Joint/Jono/models'
        log_out_dir = './Joint/Jono/Jono'

elif opt.joint==2:
    if opt.pretrain==1:
        out_dir = './Joint/JoJopre/models'
        log_out_dir = './Joint/JoJopre/JoJopre'
    else :
        out_dir = './Joint/JoJono/models'
        log_out_dir = './Joint/JoJono/JoJono'
        
else:
    if opt.pretrain==1:
        out_dir = './Joint/Onepre/models'
        log_out_dir = './Joint/Onepre/Onepre'
    else :
        out_dir = './Joint/Oneno/models'
        log_out_dir = './Joint/Oneno/Oneno'
        

try:
    os.makedirs(out_dir)
except OSError:
    pass


file = open("{}_logs.txt".format(log_out_dir), "w+")
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000) 
file.write("Random Seed: {} \n".format(opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


cudnn.benchmark = True
image_size = (224, 224)
data_root = '../Kaggle265'     # '../Dataset/Kaggle265'
batch_size = 512

patch_dim = 96
contra_dim = 128
gap = 6
jitter = 6

saveinterval = 1
num_epochs = 100
fine_epochs = 20

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file.write("using " + str(device) + "\n")
file.flush()

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
loader_plain = plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_rota = rotaloader(data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_patch = patchloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_jigpa = jigpaloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_jigro = jigroloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
loader_contra = contraloader(patch_dim, data_root, data_pre_transforms, data_post_transforms, batch_size)

# Model Initialization
# 仅支持 Res-Net !!!
model_all = models.resnet18(pretrained=opt.pretrain)
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
fc_plain = make_MLP(num_ftrs, num_ftrs, output_ftrs=265, layers=1)

fc_rota = make_MLP(num_ftrs, num_ftrs, output_ftrs=4, layers=1)
fc_patch = make_MLP(2*num_ftrs, 2*num_ftrs, output_ftrs=8, layers=2)
fc_jigpa = make_MLP(4*num_ftrs, 4*num_ftrs, output_ftrs=24, layers=4)
### 警告：类间距不同，不可粗暴计算损失，需要重写损失函数 ###
fc_jigro = make_MLP(4*num_ftrs, 4*num_ftrs, output_ftrs=96, layers=8)   # output_ftrs=24
# 对比学习部分
fc_contra = make_MLP(2*num_ftrs, 2*num_ftrs, output_ftrs=1, layers=2)

if torch.cuda.device_count() > 1: 
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_ft = nn.DataParallel(model_ft)
    fc_plain = nn.DataParallel(fc_plain)
    fc_rota = nn.DataParallel(fc_rota)
    fc_patch = nn.DataParallel(fc_patch)
    fc_jigpa = nn.DataParallel(fc_jigpa)
    fc_jigro = nn.DataParallel(fc_jigro)

model_ft = model_ft.to(device)
fc_plain = fc_plain.to(device)
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

loadstate(model_ft, fc_plain, opt.netCont, opt.plainCont, device, file)
loadstate(model_ft, fc_rota, opt.netCont, opt.rotaCont, device, file)
loadstate(model_ft, fc_patch, opt.netCont, opt.patchCont, device, file)
loadstate(model_ft, fc_jigpa, opt.netCont, opt.jigpaCont, device, file)
loadstate(model_ft, fc_jigro, opt.netCont, opt.jigroCont, device, file)
loadstate(model_ft, fc_contra, opt.netCont, opt.contraCont, device, file)

# Model trainer
criterion = nn.CrossEntropyLoss()
milestones = [50, 100, 150]
milegamma = 0.2

optimizer_all = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_plain.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
    {'params': fc_rota.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
    {'params': fc_patch.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
    {'params': fc_jigpa.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
    {'params': fc_jigro.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler_all = lr_scheduler.MultiStepLR(optimizer_all, milestones, milegamma)

optimizer_plain = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_plain.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler_plain = lr_scheduler.MultiStepLR(optimizer_plain, milestones, milegamma)

optimizer_rota = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_rota.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler_rota = lr_scheduler.MultiStepLR(optimizer_rota, milestones, milegamma)

optimizer_patch = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_patch.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler_patch = lr_scheduler.MultiStepLR(optimizer_patch, milestones, milegamma)

optimizer_jigpa = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_jigpa.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler_jigpa = lr_scheduler.MultiStepLR(optimizer_jigpa, milestones, milegamma)

optimizer_jigro = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_jigro.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler_jigro = lr_scheduler.MultiStepLR(optimizer_jigro, milestones, milegamma)

optimizer_contra = optim.Adam([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'weight_decay': opt.weight_net},
    {'params': fc_contra.parameters(), 'lr': opt.lr_fc, 'weight_decay': opt.weight_fc},
])
scheduler_contra = lr_scheduler.MultiStepLR(optimizer_contra, milestones, milegamma)

# print('Training ... {}\n'.format(opt.powerword))
# 'rota' , 'patch' , 'jigpa' , 'jigro'
if opt.joint==0:
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro = LaStep(
            loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, loader_contra, 
            'plain', model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
            optimizer_plain, optimizer_rota, optimizer_patch, optimizer_jigpa, optimizer_jigro, optimizer_contra, 
            scheduler_plain, scheduler_rota, scheduler_patch, scheduler_jigpa, scheduler_jigro, scheduler_contra, 
            criterion, device, out_dir, file, saveinterval, 0, num_epochs+fine_epochs
        )

elif opt.joint==1:
    powerword = ['rota', 'patch', 'jigpa', 'jigro']
    for i in range(num_epochs):
        model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro = LaStep(
            loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, loader_contra, 
            powerword[i%4], model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
            optimizer_plain, optimizer_rota, optimizer_patch, optimizer_jigpa, optimizer_jigro, optimizer_contra, 
            scheduler_plain, scheduler_rota, scheduler_patch, scheduler_jigpa, scheduler_jigro, scheduler_contra, 
            criterion, device, out_dir, file, saveinterval, i, 1
        )
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro = LaStep(
            loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, loader_contra, 
            'plain', model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
            optimizer_plain, optimizer_rota, optimizer_patch, optimizer_jigpa, optimizer_jigro, optimizer_contra, 
            scheduler_plain, scheduler_rota, scheduler_patch, scheduler_jigpa, scheduler_jigro, scheduler_contra, 
            criterion, device, out_dir, file, saveinterval, num_epochs, fine_epochs
        )

elif opt.joint==2:
    powerword = ['rota', 'patch', 'jigpa', 'jigro', 'plain']
    for i in range(num_epochs+fine_epochs):
        model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro = LaStep(
            loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, loader_contra, 
            powerword[i%5], model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
            optimizer_plain, optimizer_rota, optimizer_patch, optimizer_jigpa, optimizer_jigro, optimizer_contra, 
            scheduler_plain, scheduler_rota, scheduler_patch, scheduler_jigpa, scheduler_jigro, scheduler_contra, 
            criterion, device, out_dir, file, saveinterval, i, 1
        )

elif opt.joint==3:
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro = JointStep(
        loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, loader_contra, 
        model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
        optimizer_all, scheduler_all, criterion, device, out_dir, file, saveinterval, 0, num_epochs+fine_epochs
    )

