#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from collections import OrderedDict

import os
import shutil

#The idea is to have classification layers for different tasks


#class specific features are only limited to the last linear layer of the model
class classification_head(nn.Module):
    """
    Each task has a seperate classification head which houses the features that
    are specific to that particular task. These features are unshared across tasks
    as described in section 5.1 of the paper
    """
    
    def __init__(self, in_features, mid_features, out_features):
        super(classification_head, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, mid_features), 
            nn.Linear(mid_features, mid_features), 
            nn.Linear(mid_features, out_features)
        )

    def forward(self, x):
        return x
        
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


class shared_model_alex(nn.Module):

    def __init__(self):
        super().__init__()
        self.tmodel = models.alexnet(pretrained = True)
        self.reg_params = {}

    def forward(self, x):
        return self.tmodel(x)

class shared_model_res18(nn.Module):

    def __init__(self):
        super().__init__()
        res_net = models.resnet18(pretrained = False)
        self.tmodel = nn.Module()
        self.tmodel.add_module('features', nn.Sequential(*(list(res_net.children())[:-2])))
        self.tmodel.add_module('classifier', nn.Sequential(*(list(res_net.children())[-2:])))
        self.reg_params = {}

    def forward(self, x):
        return self.tmodel(x)

class shared_model(nn.Module):

    def __init__(self):
        super().__init__()
        res_net = models.resnet18(pretrained = False)
        self.tmodel = nn.Sequential(OrderedDict([
            ('features', nn.Sequential(*(list(res_net.children())[:-1]))),
            ('classifier', nn.Sequential(*(list(res_net.children())[-1:]))),
        ]))
        self.reg_params = {}

    def forward(self, x):
        return self.tmodel(x)
