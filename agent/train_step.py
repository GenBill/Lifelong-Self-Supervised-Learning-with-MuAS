from .mission import *

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

def LaStep(
    loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, loader_contra, 
    powerword, model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
    optimizer_plain, optimizer_rota, optimizer_patch, optimizer_jigpa, optimizer_jigro, optimizer_contra, 
    scheduler_plain, scheduler_rota, scheduler_patch, scheduler_jigpa, scheduler_jigro, scheduler_contra, 
    criterion, device, out_dir, file, saveinterval, last_epochs, num_epochs, 
):

    if powerword=='plain':
        model_ft, fc_plain = plaintrain(
            model_ft, fc_plain, 
            loader_plain, criterion, optimizer_plain, scheduler_plain, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='rota':
        model_ft, fc_rota = rotatrain(
            model_ft, fc_rota, 
            loader_rota, criterion, optimizer_rota, scheduler_rota, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='patch':
        model_ft, fc_patch = patchtrain(
            model_ft, fc_patch, 
            loader_patch, criterion, optimizer_patch, scheduler_patch, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='jigpa':
        model_ft, fc_jigpa = jigpatrain(
            model_ft, fc_jigpa, 
            loader_jigpa, criterion, optimizer_jigpa, scheduler_jigpa, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='jigro':
        model_ft, fc_jigro = jigrotrain(
            model_ft, fc_jigro, 
            loader_jigro, criterion, optimizer_jigro, scheduler_jigro, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='contra':
        model_ft, fc_contra = contratrain(
            model_ft, fc_contra, 
            loader_contra, criterion, optimizer_contra, scheduler_contra, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )
    
    return model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro

    # default num_step == 0,1,2,3
    # else return 0


def JointStep(
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
    loader_joint, optimizer, scheduler, criterion, device, out_dir, file, saveinterval, last_epochs, num_epochs, 
):
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro = jointtrain(
            model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, 
            loader_joint, optimizer, scheduler, criterion, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )
    return model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro


def SingleStep(
    image_size, data_root, batch_size, patch_dim, contra_dim, gap, jitter, 
    powerword, model_ft, fc_layer, criterion, 
    lr_0, weight_0, lr_1, weight_1, milestones, milegamma, 
    device, out_dir, file, saveinterval, last_epochs, num_epochs, num_workers
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
    optimizer = optim.Adam(
        [
            {'params': model_ft.parameters(), 'lr': lr_0, 'weight_decay': weight_0},
            {'params': fc_layer.parameters(), 'lr': lr_1, 'weight_decay': weight_1},
        ]
    )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, milegamma)

    if powerword=='plain':
        loader_plain = plainloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        model_ft, fc_layer = plaintrain(
            model_ft, fc_layer, 
            loader_plain, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='rota':
        loader_rota = rotaloader(data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        model_ft, fc_layer = rotatrain(
            model_ft, fc_layer, 
            loader_rota, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='patch':
        loader_patch = patchloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        model_ft, fc_layer = patchtrain(
            model_ft, fc_layer, 
            loader_patch, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='jigpa':
        loader_jigpa = jigpaloader(patch_dim, gap, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        model_ft, fc_layer = jigpatrain(
            model_ft, fc_layer, 
            loader_jigpa, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='jigro':
        loader_jigro = jigroloader(patch_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        model_ft, fc_layer = jigrotrain(
            model_ft, fc_layer, 
            loader_jigro, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )

    elif powerword=='contra':
        loader_contra = contraloader(contra_dim, jitter, data_root, data_pre_transforms, data_post_transforms, batch_size, num_workers)
        model_ft, fc_layer = contratrain(
            model_ft, fc_layer, 
            loader_contra, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )
    
    return model_ft, fc_layer

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