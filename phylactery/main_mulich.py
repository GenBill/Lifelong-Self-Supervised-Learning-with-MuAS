#!/usr/bin/env python
# coding: utf-8

from agent.mission.m_plain import plainloader
import torch
torch.backends.cudnn.benchmark=True

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import argparse 
import numpy as np
from random import shuffle

import copy

# import sys 
# sys.path.append('./utils')
from MuAS.utils.model_utils import *
from MuAS.utils.mas_utils import *

from MuAS.model_class import *
from MuAS.optimizer_lib import *
from MuAS.model_train import *
from MuAS.mas import *

from cometopower import cometopower

parser = argparse.ArgumentParser(description='Test file')
parser.add_argument('--cuda', default='', type=str, help = 'Set the GPU index')
parser.add_argument('--batch_size', default=32, type=int, help = 'The batch size you want to use')
parser.add_argument('--num_workers', default=8, type=int, help = 'The num workers you want to use')
parser.add_argument('--num_freeze_layers', default=2, type=int, help = 'Number of layers you want to frozen in the feature extractor of the model')
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs you want to train the model on')
parser.add_argument('--last_epochs', default=10, type=int, help='Number of epochs you want to Finetune')

parser.add_argument('--init_lr', default=0.001, type=float, help='Initial learning rate for training the model')
parser.add_argument('--reg_lambda', default=0.01, type=float, help='Regularization parameter')
parser.add_argument('--miu', default=0.99, type=float, help='Initial MiuAS')

parser.add_argument('--circle', default=1, type=int, help='Initial eLich Circle')

args = parser.parse_args()
cuda_index = args.cuda
batch_size = args.batch_size
num_workers = args.num_workers

no_of_layers = args.num_freeze_layers
num_epochs = args.num_epochs
last_epochs = args.last_epochs

lr = args.init_lr
reg_lambda = args.reg_lambda
miu = args.miu

dloaders_train = []
dloaders_test = []

num_classes = []

data_dir = '~/Datasets/miniImageNet'
# data_path = "~/Datasets/Kaggle265"

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_size = (224, 224)
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

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dset_size_train, dset_size_test = cometopower('none', data_dir, data_pre_transforms, data_post_transforms, batch_size, num_workers)
loader_plain = plainloader(data_dir, data_pre_transforms, data_post_transforms, batch_size, num_workers)

powerlist = args.circle * ['rota', 'patch', 'jigpa', 'jigro'] + ['plain']
num_in_times = args.circle * [1, 2, 4, 4] + [1]
num_classes = args.circle * [4, 8, 24, 96] + [100]

for powerword in powerlist:
    
    dset_loaders = cometopower(powerword, data_dir, data_pre_transforms, data_post_transforms, batch_size, num_workers)
    tr_dset_loaders = dset_loaders['train']
    te_dset_loaders = dset_loaders['test']

    #append the dataloaders of these tasks
    dloaders_train.append(tr_dset_loaders)
    dloaders_test.append(te_dset_loaders)

#get the number of tasks in the sequence
no_of_tasks = len(powerlist)

# model = shared_model(models.alexnet(pretrained = True))
model = shared_model()


#train the model on the given number of tasks
for task in range(no_of_tasks):
    print ("Training the model on task {}".format(task+1))

    dataloader_train = dloaders_train[task]
    dataloader_test = dloaders_test[task]
    no_of_in_times = num_in_times[task]
    no_of_classes = num_classes[task]

    model = model_init(no_of_in_times, no_of_classes, device)
    if task < no_of_tasks-1 :
        mulich_train(model, task+1, num_epochs, no_of_layers, no_of_classes, 
            dataloader_train, dataloader_test, loader_plain['train'], # include ['train'] and ['test']
            dset_size_train, dset_size_test, 
            device, lr, reg_lambda, miu)
    else :
        mulich_train(model, task+1, last_epochs, no_of_layers, no_of_classes, 
        dataloader_train, dataloader_test, loader_plain['train'], # include ['train'] and ['test']
        dset_size_train, dset_size_test, 
        device, lr, reg_lambda, miu)
    

print ("The training process on the {} tasks is completed".format(no_of_tasks))

print ("Testing the model now")

#test the model out on the test sets of the tasks
for task in range(no_of_tasks):
    print ("Testing the model on task {}".format(task))

    dataloader = dloaders_test[task]
    dset_size = dset_size_test # dsets_test[task-1]
    no_of_in_times = num_in_times[task]
    no_of_classes = num_classes[task]
    
    # now_performance - old_performance
    forgetting = compute_forgetting(task+1, no_of_in_times, dataloader, dset_size, device)

    print ("The forgetting undergone on task {} is {:.4f}%".format(task+1, forgetting*100))
    








