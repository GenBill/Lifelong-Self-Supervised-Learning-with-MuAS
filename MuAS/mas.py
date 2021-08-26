#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil

import sys
import time

# sys.path.append('./utils')
from MuAS.utils.model_utils import *
from MuAS.utils.mas_utils import *

from MuAS.optimizer_lib import *
from MuAS.model_train import *


def mas_train(model, task_no, num_epochs, no_of_layers, no_of_classes, 
    dataloader_train, dataloader_test, dset_size_train, dset_size_test, 
    device, lr = 0.001, reg_lambda = 0.01, miu = 0.99):
    """
    Inputs:
    1) model: A reference to the model that is being exposed to the data for the task
    2) task_no: The task that is being exposed to the model identified by it's number
    3) no_of_layers: The number of layers that you want to freeze in the feature extractor of the Alexnet
    4) no_of_classes: The number of classes in the task  
    5) dataloader_train: Dataloader that feeds training data to the model
    6) dataloader_test: Dataloader that feeds test data to the model
    6) dset_size_train: The size of the task (size of the dataset belonging to the training task)
    7) dset_size_test: The size of the task (size of the dataset belonging to the test set)
    7) use_gpu: Set the flag to `True` if you want to train the model on GPU

    Outputs:
    1) model: Returns a trained model

    Function: Trains the model on a particular task and deals with different tasks in the sequence
    """

    #this is the task to which the model is exposed
    if (task_no == 1):
        #initialize the reg_params for this task
        if no_of_layers<0:
            model = init_reg_params(model, device, [])
        else:
            model, freeze_layers = create_freeze_layers(model, no_of_layers)
            model = init_reg_params(model, device, freeze_layers)

    else:
        #inititialize the reg_params for this task
        model = init_reg_params_across_tasks(model, device)

    #get the optimizer
    optimizer_sp = local_sgd(model.tmodel.parameters(), reg_lambda, lr)
    train_model(model, task_no, no_of_classes, optimizer_sp, model_criterion, 
        dataloader_train, dataloader_test, dset_size_train, dset_size_test, 
        num_epochs, device, lr, reg_lambda, miu)

    if (task_no > 1):

        model = consolidate_reg_params(model, device)

    return model


def mulich_train(model, task_no, num_epochs, no_of_layers, no_of_classes, 
    dataloader_train, dataloader_test, loader_plain, dset_size_train, dset_size_test, 
    device, lr = 0.001, reg_lambda = 0.01, miu = 0.99):
    """
    Inputs:
    1) model: A reference to the model that is being exposed to the data for the task
    2) task_no: The task that is being exposed to the model identified by it's number
    3) no_of_layers: The number of layers that you want to freeze in the feature extractor of the Alexnet
    4) no_of_classes: The number of classes in the task  
    5) dataloader_train: Dataloader that feeds training data to the model
    6) dataloader_test: Dataloader that feeds test data to the model
    6) dset_size_train: The size of the task (size of the dataset belonging to the training task)
    7) dset_size_test: The size of the task (size of the dataset belonging to the test set)
    7) use_gpu: Set the flag to `True` if you want to train the model on GPU

    Outputs:
    1) model: Returns a trained model

    Function: Trains the model on a particular task and deals with different tasks in the sequence
    """

    #this is the task to which the model is exposed
    if (task_no == 1):
        #initialize the reg_params for this task
        # if no_of_layers<0:
        #     model, freeze_layers = create_freeze_layers(model, no_of_layers)
        model, freeze_layers = create_freeze_layers(model, no_of_layers)
        model = init_reg_params(model, device, freeze_layers)

    else:
        #inititialize the reg_params for this task
        model = init_reg_params_across_tasks(model, device)

    #get the optimizer
    optimizer_sp = local_sgd(model.tmodel.parameters(), reg_lambda, lr)
    optimizer_mlp = local_sgd(model.mlptail.parameters(), reg_lambda, lr)
    elich_train_model(model, task_no, no_of_classes, optimizer_sp, optimizer_mlp, model_criterion, 
        dataloader_train, dataloader_test, loader_plain, dset_size_train, dset_size_test, 
        num_epochs, device, lr, reg_lambda, miu)

    if (task_no > 1):

        model = consolidate_reg_params(model, device)

    return model

def compute_forgetting(task_no, in_times, dataloader, dset_size, device):
    """
    Inputs
    1) task_no: The task number on which you want to compute the forgetting 
    2) dataloader: The dataloader that feeds in the data to the model

    Outputs
    1) forgetting: The amount of forgetting undergone by the model

    Function: Computes the "forgetting" that the model has on the 
    """
    
    #get the results file
    store_path = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
    model_path = os.path.join(os.getcwd(), "models")
    # device = torch.device("cuda:0" if use_gpu else "cpu")

    #get the old performance
    file_object = open(os.path.join(store_path, "performance.txt"), 'r')
    old_performance = file_object.read()
    file_object.close()

    #load the model for inference
    model = model_inference(task_no, in_times, device)
    model.to(device)

    running_corrects = 0

    for data in dataloader:
        # input_data, labels = data
        if len(data)==2:
            input_data_0, labels = data
            input_data_0 = input_data_0.to(device)
            input_data = [input_data_0]

        if len(data)==3:
            input_data_0, input_data_1, labels = data
            input_data_0 = input_data_0.to(device)
            input_data_1 = input_data_1.to(device)
            input_data = [input_data_0, input_data_1]

        if len(data)==5:
            input_data_0, input_data_1, input_data_2, input_data_3, labels = data
            input_data_0 = input_data_0.to(device)
            input_data_1 = input_data_1.to(device)
            input_data_2 = input_data_2.to(device)
            input_data_3 = input_data_3.to(device)
            input_data = [input_data_0, input_data_1, input_data_2, input_data_3]

        del data

        # input_data = input_data.to(device)
        labels = labels.to(device)
        
        moutputs = model.tmodel(input_data[0])
        for i in range(1, len(input_data)):
            moutputs = torch.cat((moutputs, model.tmodel(input_data[i])), dim=1)
        output = model.mlptail(moutputs)
        # output = model.tmodel(input_data)
        del input_data

        _, preds = torch.max(output, 1)

        running_corrects += torch.sum(preds == labels.data)
        del preds
        del labels

    epoch_accuracy = running_corrects.double()/dset_size

    old_performance = float(old_performance) 
    forgetting = epoch_accuracy.item() - old_performance

    return forgetting
