from .train_rota import *
from .train_patch import *
from .train_jigpa import *
from .train_jigro import *
from .train_contra import *

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

def LaStep(
    image_size, data_root, batch_size, patch_dim, contra_dim, gap, jitter, 
    powerword, model_ft, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
    criterion, lr, weight_decay, milestones, milegamma, 
    device, out_dir, file, saveinterval, num_epochs, 
):
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
    
    if powerword=='rota':
        loader_rota = rotaloader(data_root, data_pre_transforms, data_post_transforms, batch_size)
        optimizer = optim.Adam(
            [
                {'params': model_ft.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': fc_rota.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            ]
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, milegamma)
        model_ft, fc_rota = rotatrain(
            model_ft, fc_rota, 
            loader_rota, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )

    elif powerword=='patch':
        loader_patch = patchloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
        optimizer = optim.Adam(
            [
                {'params': model_ft.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': fc_patch.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            ]
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, milegamma)
        model_ft, fc_patch = patchtrain(
            model_ft, fc_patch, 
            loader_patch, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )

    elif powerword=='jigpa':
        loader_jigpa = jigpaloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
        optimizer = optim.Adam(
            [
                {'params': model_ft.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': fc_jigpa.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            ]
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, milegamma)
        model_ft, fc_jigpa = jigpatrain(
            model_ft, fc_jigpa, 
            loader_jigpa, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )

    elif powerword=='jigro':
        loader_jigro = jigroloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
        optimizer = optim.Adam(
            [
                {'params': model_ft.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': fc_jigro.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            ]
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, milegamma)
        model_ft, fc_jigro = jigrotrain(
            model_ft, fc_jigro, 
            loader_jigro, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )

    elif powerword=='contra':
        loader_contra = contraloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size)
        optimizer = optim.Adam(
            [
                {'params': model_ft.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': fc_contra.parameters(), 'lr': lr, 'weight_decay': weight_decay},
            ]
        )
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, milegamma)
        model_ft, fc_contra = contratrain(
            model_ft, fc_contra, 
            loader_contra, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )
    
    return model_ft, fc_rota, fc_patch, fc_jigpa, fc_jigro

    # default num_step == 0,1,2,3
    # else return 0


### 警告：废弃管道 ###

'''
def Step(
    powerword, model_ft, fc_rota, fc_patch, fc_jigpa, fc_jigro,
    loader_rota, loader_patch, loader_jigpa, loader_jigro, 
    criterion, optimizer, scheduler,
    device, out_dir, file, saveinterval, num_epochs,
):
    # default num_step == 0,1,2,3
    if powerword=='rota':
        model_ft, fc_rota = rotatrain(
            model_ft, fc_rota, 
            loader_rota, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )

    elif powerword=='patch':
        model_ft, fc_patch = patchtrain(
            model_ft, fc_patch, 
            loader_patch, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )

    elif powerword=='jigpa':
        model_ft, fc_jigpa = jigpatrain(
            model_ft, fc_jigpa, 
            loader_jigpa, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )

    elif powerword=='jigro':
        model_ft, fc_jigro = jigrotrain(
            model_ft, fc_jigro, 
            loader_jigro, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )
    
    return model_ft, fc_rota, fc_patch, fc_jigpa, fc_jigro

    # default num_step == 0,1,2,3
    # else return 0
'''