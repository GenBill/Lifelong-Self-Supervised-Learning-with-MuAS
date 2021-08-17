# 警告：修改全连接层的层数 layers = 1 -> 2

from agent import LaStep
from agent import JointStep
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
# parser.add_argument('--powerword', default='rota', help="Power Word, decide what to do")

parser.add_argument('--batchsize', type=int, default=512, help="set batch size")
parser.add_argument('--numworkers', type=int, default=4, help="set num workers")
parser.add_argument('--epochs_0', type=int, default=200, help="set num epochs")
parser.add_argument('--epochs_1', type=int, default=40, help="set num epochs")

parser.add_argument('--lr_net', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--lr_fc', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--weight_net', type=float, default=1e-4, help="weight decay")
parser.add_argument('--weight_fc', type=float, default=1e-4, help="weight decay")

parser.add_argument('--netCont', default='', help="path to net (for continue training)")
parser.add_argument('--plainCont', default='', help="path to plain fc_layer (for continue training)")
parser.add_argument('--rotaCont', default='', help="path to fc_layer rota")
parser.add_argument('--jigroCont', default='', help="path to fc_layer patch")
parser.add_argument('--patchCont', default='', help="path to fc_layer jigpa")
parser.add_argument('--jigpaCont', default='', help="path to fc_layer jigro")
parser.add_argument('--contraCont', default='', help="path to fc_layer jigro")
# parser.add_argument('--manualSeed', type=int, help='manual seed')

# parser.add_argument('--joint', type=int, default=1, help="joint on")
parser.add_argument('--pretrain', type=int, default=1, help="pretrain on")

# opt = parser.parse_args(args=[])
opt = parser.parse_args()
opt.manualSeed = 2077
# opt.netCont = './models/net_epoch_56.pth'

if opt.pretrain==1:
    dirname = 'Demipre'
else:
    dirname = 'Demino'
out_dir = '../Joint_{}/{}/models'.format(opt.batchsize, dirname)
log_out_dir = '../Joint_{}/{}/{}'.format(opt.batchsize, dirname, dirname)

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
data_root = '../../Kaggle265'   # '../Dataset/Kaggle265'
batch_size = opt.batchsize      # 512, 256
num_workers = opt.numworkers    # 4

patch_dim = 96
contra_dim = 128
gap = 6
jitter = 6

saveinterval = 1
num_epochs = opt.epochs_0
fine_epochs = opt.epochs_1

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

loader_joint = DJloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
loader_test = testloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)

loader_plain = plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)

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
fc_plain = make_MLP(num_ftrs, num_ftrs, output_ftrs=265, layers=2)      # layers=1)

fc_rota = make_MLP(num_ftrs, num_ftrs, output_ftrs=4, layers=2)         # layers=1)
fc_patch = make_MLP(2*num_ftrs, 2*num_ftrs, output_ftrs=8, layers=2)
fc_jigpa = make_MLP(4*num_ftrs, 4*num_ftrs, output_ftrs=24, layers=4)
### 警告：Jigro类间距不同，不可粗暴计算损失，需要重写损失函数 ###
# Bye Jigro
# fc_jigro = make_MLP(4*num_ftrs, 4*num_ftrs, output_ftrs=96, layers=8)   # output_ftrs=24
# 对比学习部分
fc_contra = make_MLP(2*num_ftrs, 2*num_ftrs, output_ftrs=2, layers=2)

if torch.cuda.device_count() > 1: 
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_ft = nn.DataParallel(model_ft)
    fc_plain = nn.DataParallel(fc_plain)
    fc_rota = nn.DataParallel(fc_rota)
    fc_patch = nn.DataParallel(fc_patch)
    fc_jigpa = nn.DataParallel(fc_jigpa)
    fc_contra = nn.DataParallel(fc_contra)

model_ft = model_ft.to(device)
fc_plain = fc_plain.to(device)
fc_rota = fc_rota.to(device)
fc_patch = fc_patch.to(device)
fc_jigpa = fc_jigpa.to(device)
fc_contra = fc_contra.to(device)

# Load state : model & fc_layer
def loadstate(model, net_Cont, device, file):
    if net_Cont != '':
        model.load_state_dict(torch.load(net_Cont, map_location=device))
        print('Loaded model/fc state ...')
        file.write('Loaded model/fc state ...')

loadstate(model_ft, opt.netCont, device, file)
loadstate(fc_plain, opt.plainCont, device, file)
loadstate(fc_rota, opt.rotaCont, device, file)
loadstate(fc_patch, opt.patchCont, device, file)
loadstate(fc_jigpa, opt.jigpaCont, device, file)
# loadstate(fc_jigro, opt.jigroCont, device, file)
loadstate(fc_contra, opt.contraCont, device, file)

# Model trainer
criterion = nn.CrossEntropyLoss()
milestones = [5, 20, 40, 80, 120, 200, 300, 400, 800, 1600]
milegamma = 0.6
optimizer_all = optim.SGD([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'momentum': opt.momentum, 'weight_decay': opt.weight_net},
    # {'params': fc_plain.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
    {'params': fc_rota.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
    {'params': fc_patch.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
    {'params': fc_jigpa.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
    {'params': fc_contra.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
])
scheduler_all = lr_scheduler.MultiStepLR(optimizer_all, milestones, milegamma)

# milestones = [10, 20, 30, 40]
# milegamma = 0.6
optimizer_plain = optim.SGD([
    # {'params': model_ft.parameters(), 'lr': opt.lr_net, 'momentum': opt.momentum, 'weight_decay': opt.weight_net},
    {'params': fc_plain.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
])
scheduler_plain = lr_scheduler.MultiStepLR(optimizer_plain, milestones, milegamma)

optimizer_rota = optim.SGD([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'momentum': opt.momentum, 'weight_decay': opt.weight_net},
    {'params': fc_rota.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
])
scheduler_rota = lr_scheduler.MultiStepLR(optimizer_rota, milestones, milegamma)

optimizer_patch = optim.SGD([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'momentum': opt.momentum, 'weight_decay': opt.weight_net},
    {'params': fc_patch.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
])
scheduler_patch = lr_scheduler.MultiStepLR(optimizer_patch, milestones, milegamma)

optimizer_jigpa = optim.SGD([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'momentum': opt.momentum, 'weight_decay': opt.weight_net},
    {'params': fc_jigpa.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
])
scheduler_jigpa = lr_scheduler.MultiStepLR(optimizer_jigpa, milestones, milegamma)

optimizer_contra = optim.SGD([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'momentum': opt.momentum, 'weight_decay': opt.weight_net},
    {'params': fc_contra.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
])
scheduler_contra = lr_scheduler.MultiStepLR(optimizer_contra, milestones, milegamma)

optimizer_contra = optim.SGD([
    {'params': model_ft.parameters(), 'lr': opt.lr_net, 'momentum': opt.momentum, 'weight_decay': opt.weight_net},
    {'params': fc_contra.parameters(), 'lr': opt.lr_fc, 'momentum': opt.momentum, 'weight_decay': opt.weight_fc},
])
scheduler_contra = lr_scheduler.MultiStepLR(optimizer_contra, milestones, milegamma)

# print('Training ... {}\n'.format(opt.powerword))
# 'rota' , 'patch' , 'jigpa' , 'contra'

model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_contra = demitrain(
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_contra, 
    loader_joint, loader_test, 
    # 警告：optimizer_all 不含 fc_plain
    # 警告：optimizer_0 仅优化 fc_plain
    optimizer_all, optimizer_plain, optimizer_rota, optimizer_patch, optimizer_jigpa, optimizer_contra, 
    scheduler_all, scheduler_plain, scheduler_rota, scheduler_patch, scheduler_jigpa, scheduler_contra, 
    criterion, device, out_dir, file, saveinterval, 500, num_epochs)

milestones = [5, 10, 20, 40, 80, 100]
milegamma = 0.8
optimizer_finetune = optim.SGD([
    {'params': model_ft.parameters(), 'lr': 5e-3, 'momentum': 0.9, 'weight_decay': opt.weight_net},
    {'params': fc_plain.parameters(), 'lr': 5e-3, 'momentum': 0.9, 'weight_decay': opt.weight_fc},
])
scheduler_finetune = lr_scheduler.MultiStepLR(optimizer_finetune, milestones, milegamma)

model_ft, fc_plain = plaintrain(
    model_ft, fc_plain, 
    loader_plain, criterion, optimizer_finetune, scheduler_finetune, 
    device, out_dir, file, saveinterval, 500, fine_epochs
)