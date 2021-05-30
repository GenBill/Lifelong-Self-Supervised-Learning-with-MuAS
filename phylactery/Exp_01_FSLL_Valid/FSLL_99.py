from tqdm import tqdm
import numpy as np
import os
import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader

# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")

manualSeed = 2077     # random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# copy from Frost_Func.py
# Sector Frost_Func
def print_params(net):
    params = list(net.named_parameters())   # get the index by debuging
    for i in range(len(params)):
        print(i, params[i][0])      # name
        print(params[i][1].data)    # data

def BW_tensor(this_tensor, percent):
    # percent : 将超过min(percent)的数值设置为不更新
    length = this_tensor.numel()
    temp_tensor = torch.abs(this_tensor.view(-1))
    Kmin_list = torch.topk(temp_tensor, math.floor(length*percent), largest=False, sorted=True)
    if len(Kmin_list.values)!=0 :
        threshold = Kmin_list.values[-1]
        return (torch.abs(this_tensor)<=threshold).float()
    else :
        return torch.zeros(1)    # this_tensor<0 + this_tensor>=0

def Get_shadow_net(net, percent=0.5):
    shadow_net = copy.deepcopy(net)
    for params in shadow_net.parameters():
        params.data = BW_tensor(params.data, percent)
        params.requires_grad_(False)
    return shadow_net

def Unfreeze_net(shadow_net, sign_str=''):
    params_shadow = list(shadow_net.named_parameters())
    for i in range(len(params_shadow)) :
        this_str = params_shadow[i][0]
        if sign_str in this_str:
            params_shadow[i][1].data *= 0.
            params_shadow[i][1].data += 1.
    return shadow_net


def Get_old_net(net):
    old_net = copy.deepcopy(net)
    for params in old_net.parameters():
        params.requires_grad_(False)
    return old_net


def Frost_iter(net, shadow_net, old_net, X, target, loss_Func, optimizer, reg_lamb=5):
    # 构造参数索引 params_list
    params_net = list(net.named_parameters())
    params_old = list(old_net.named_parameters())
    params_shadow = list(shadow_net.named_parameters())
    
    # 计算基本损失
    pred = F.softmax(net(X), 1)
    loss = loss_Func(pred, target)
    # print(pred[0])

    # 添加遗忘损失
    # for i in range(len(params_net)-2):
    #     loss += reg_lamb * torch.sum(torch.abs(params_net[i][1] - params_old[i][1]))
    
    # 计算 grad
    optimizer.zero_grad()
    loss.backward()

    # 过滤 grad
    for i in range(len(params_net)-2):
        params_net[i][1].grad = params_net[i][1].grad * params_shadow[i][1].data
    
    # End Sub
    optimizer.step()
    # acc_mat = torch.argmax(pred, 1)==torch.argmax(target, 1)
    acc_mat = torch.argmax(pred, 1)==target
    acc_time = torch.sum(acc_mat)

    return loss.item(), acc_time.item()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 任务：CIFAR10，CUB200（暂定）
# ResNet-18 装载
this_Epoch = 0
checkpoint_path = '/home/zhangxuanming/eLich/Saved_models/valid_FSLL_99'      # 'E:/Laplace/Dataset/Kaggle265/valid'
logs = open(checkpoint_path+'/training_logs.txt', "w+")
logs.write("Random Seed: {} \n".format(manualSeed))
logs.write("Loaded Epoch {} and continuing training\n".format(this_Epoch))

Trainset_path = '/home/zhangxuanming/Kaggle265/train'
Testset_path = '/home/zhangxuanming/Kaggle265/test'
Validset_path = '/home/zhangxuanming/Kaggle265/valid'

batch_size = 512
num_epochs = 1000

# 载入数据

data_transform = transforms.Compose([
    # transforms.Resize(224),           # 缩放图片(Image)，保持长宽比不变，最短边为32像素
    # transforms.CenterCrop(224),       # 从图片中间切出32*32的图片
    transforms.ToTensor(),              # 将图片(Image)转成Tensor，归一化至[0, 1]
    transforms.Normalize(
        mean = [0.492, 0.461, 0.417], 
        std = [0.256, 0.248, 0.251]
    )                                   # 标准化至[-1, 1]，规定均值和标准差
])


data_train = datasets.ImageFolder(root = Trainset_path, transform = data_transform)

# 就是普通的载入数据
data_train = datasets.ImageFolder(root = Trainset_path, transform = data_transform)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
data_size = data_train.__len__()

# 载入模型
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 265)
net = net.to(device)

if torch.cuda.device_count() > 1: 
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    logs.write("Now we use {} GPUs!\n\n".format(torch.cuda.device_count()))
net = nn.DataParallel(net)


if this_Epoch != 0 :
    net.load_state_dict(torch.load('%s/Epoch_%d.pth' % (checkpoint_path, this_Epoch), map_location=device))

# 就是普通的载入模型
optimizer = optim.Adam(
    [
        {'params': (p for name, p in net.named_parameters() if 'weight' in name), 'lr': 1e-3, 'momentum': 0.6, 'weight_decay': 1e-5},
        {'params': (p for name, p in net.named_parameters() if 'bias' in name), 'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 0.}
    ]   , lr=1e-3, weight_decay=1e-5
)

criterion = nn.CrossEntropyLoss(reduction='mean')   # nn.MSELoss(reduction='mean')
loss_list = []
accrate_list = []

# Train Step
# 镜像
old_net = Get_old_net(net)
shadow_net = Get_shadow_net(net, 1.)
shadow_net = Unfreeze_net(shadow_net, 'fc')

'''
sign_str = 'fc'
params_shadow = list(shadow_net.named_parameters())
params_net = list(net.named_parameters())
print(params_net[-1][1].name)
print(params_net[-2][1].name)

for i in range(len(params_shadow)):
    this_str = params_shadow[i][0]
    if sign_str in this_str:
        print(params_shadow[i][1].data)
'''

# Epoch On !
for epoch in range(1,num_epochs+1):
    epoch_loss = 0
    epoch_acc = 0
    for batch_num, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # targets = F.one_hot(labels,265).float().to(device)

        this_loss, this_acc = Frost_iter(net, shadow_net, old_net, inputs, labels, criterion, optimizer, reg_lamb=0.)
        epoch_loss += this_loss
        epoch_acc += this_acc
    loss_list.append(epoch_loss)
    accrate_list.append(epoch_acc/data_size)

    if epoch % 1 == 0 :
        print('Epoch: {} \nAcc: {:.4f}, Loss: {:.4f}'.format(epoch, epoch_acc/data_size, epoch_loss/(data_size//batch_size)))
        logs.write('\nEpoch: {} \nAcc: {:.4f}, Loss: {:.4f}\n'.format(epoch, epoch_acc/data_size, epoch_loss/(data_size//batch_size)))
        logs.flush()

    if epoch % 10 == 0 :
        torch.save(net.state_dict(), '%s/Epoch_%d.pth' % (checkpoint_path, this_Epoch+epoch))

logs.close()