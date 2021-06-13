from .train_plain import *
from .train_rota import *
from .train_patch import *
from .train_jigpa import *
from .train_jigro import *
from .train_contra import *

from .train_joint import *

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
    loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, loader_contra, 
    model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, fc_contra, 
    optimizer, scheduler, criterion, device, out_dir, file, saveinterval, last_epochs, num_epochs, 
):
    model_ft, fc_contra = jointtrain(
            loader_plain, loader_rota, loader_patch, loader_jigpa, loader_jigro, 
            model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro, 
            criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, last_epochs, num_epochs
        )
    return model_ft, fc_plain, fc_rota, fc_patch, fc_jigpa, fc_jigro
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