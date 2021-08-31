from agent import SingleStep
from agent.mission import plainloader
from evap import evap
from onlytest import onlytest

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

parser.add_argument('--batchsize', type=int, default=512, help="set batch size")
parser.add_argument('--numworkers', type=int, default=4, help="set num workers")
parser.add_argument('--num_epoch_0', type=int, default=20, help="num_epoch")
parser.add_argument('--num_epoch_1', type=int, default=100, help="num_epoch")
parser.add_argument('--quick_flag', type=bool, default=True, help="quick_flag")

parser.add_argument('--lr', type=float, default=1e-2, help='learning rate, default=0.001')
parser.add_argument('--momentum', type=float, default=0., help="momentum")
parser.add_argument('--weight', type=float, default=1e-4, help="weight decay")
parser.add_argument('--alpha', type=float, default=0.5, help="alpha")

parser.add_argument('--manualSeed', type=int, default=2077, help='manual seed')

parser.add_argument('--pretrain', type=bool, default=True, help="pretrain on")
parser.add_argument('--classnum', type=int, default=265, help="set class num")

# opt = parser.parse_args(args=[])
opt = parser.parse_args()
opt.classnum = 100

dirname = 'Evap'
if opt.pretrain:
    dirname = dirname + 'pre'
else :
    dirname = dirname + 'no'
    
out_dir = '../Single_{}/{}/models'.format(opt.batchsize, dirname)
log_out_dir = '../Single_{}/{}/{}'.format(opt.batchsize, dirname, dirname)

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
data_root = '~/Datasets/miniImageNet'
# data_root = '../../Kaggle265'     # '../Datasets/Kaggle265'
batch_size = opt.batchsize      # 512, 256
num_workers = opt.numworkers    # 4

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file.write("using " + str(device) + "\n")
file.flush()


# Model Initialization
# 仅支持 Res-Net !!!
student = models.resnet18(pretrained=opt.pretrain)
student.fc = nn.Linear(student.fc.in_features, opt.classnum).to(device)

model_all_0 = models.resnet18(pretrained=opt.pretrain)
model_all_1 = models.resnet18(pretrained=opt.pretrain)
model_all_2 = models.resnet18(pretrained=opt.pretrain)
model_all_3 = models.resnet18(pretrained=opt.pretrain)


model_ft_0 = nn.Sequential(*(list(model_all_0.children())[:-1]))
model_ft_1 = nn.Sequential(*(list(model_all_1.children())[:-1]))
model_ft_2 = nn.Sequential(*(list(model_all_2.children())[:-1]))
model_ft_3 = nn.Sequential(*(list(model_all_3.children())[:-1]))


if torch.cuda.device_count() > 1: 
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_ft_0 = nn.DataParallel(model_ft_0)
    model_ft_1 = nn.DataParallel(model_ft_1)
    model_ft_2 = nn.DataParallel(model_ft_2)
    model_ft_3 = nn.DataParallel(model_ft_3)
    student = nn.DataParallel(student)

model_ft_0 = model_ft_0.to(device)
model_ft_1 = model_ft_1.to(device)
model_ft_2 = model_ft_2.to(device)
model_ft_3 = model_ft_3.to(device)
student = student.to(device)

# Load state : model & fc_layer
def loadstate(model, net_Cont, device, file):
    if net_Cont != '':
        model.load_state_dict(torch.load(net_Cont, map_location=device))
        print('Loaded model state ...')
        file.write('Loaded model state ...')

loadstate(model_ft_0, '../Single_256/Rotano/models/model_epoch_199.pth', device, file)
loadstate(model_ft_1, '../Single_256/Patchno/models/model_epoch_199.pth', device, file)
loadstate(model_ft_2, '../Single_256/Jigpano/models/model_epoch_199.pth', device, file)
loadstate(model_ft_3, '../Single_256/Jigrono/models/model_epoch_199.pth', device, file)

# Model trainer
criterion = nn.CrossEntropyLoss()
milestones = [50, 100, 150, 200]
milegamma = 0.6


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

loader_plain = plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
model_list = [model_all_0, model_all_1, model_all_2, model_all_3]

criterion = nn.CrossEntropyLoss()
# Train Student
student = evap(opt, loader_plain['train'], criterion, model_list, student, device)

# Test Student
student = onlytest(loader_plain['test'], criterion, student, device)

