from tqdm import tqdm
import numpy as np
import os
import copy
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='FSLL Valid')
parser.add_argument('--cuda', '-c', help='cuda Num', default='0')
parser.add_argument('--seed', '-s', help='manual Seed', default=2077)
parser.add_argument('--frost', '-f', help='Frost Stone', default=1.0)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda      # '0'

# 固定随机种子
manualSeed = args.seed      # 2077     # random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 将超过 Frost_stone 的参数设置为不更新
Frost_stone = float(args.frost)     # 0.8

if Frost_stone < 0.10 :
    Frost_str = '0'+str(int(Frost_stone*10))
elif Frost_stone > 0.99 :
    Frost_str = '99'
else :
    Frost_str = str(int(Frost_stone*100))


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
    # for i in range(len(params_net)-2):
    #     params_net[i][1].grad = params_net[i][1].grad * params_shadow[i][1].data
    
    # End Sub
    optimizer.step()
    # acc_mat = torch.argmax(pred, 1)==torch.argmax(target, 1)
    acc_mat = torch.argmax(pred, 1)==target
    acc_time = torch.sum(acc_mat)

    return loss.item(), acc_time.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 任务：CIFAR10，CUB200（暂定）
# ResNet-18 装载
this_Epoch = 0
checkpoint_path = '/home/zhangxuanming/eLich/Saved_models/valid_FSLL_'+Frost_str      # 'E:/Laplace/Dataset/Kaggle265/valid'
logs_path = '/home/zhangxuanming/eLich/Saved_logs'
plot_path = '/home/zhangxuanming/eLich/Saved_plot'

try:
    os.makedirs(checkpoint_path)
    os.makedirs(logs_path)
    os.makedirs(plot_path)
except OSError:
    pass

logs = open(logs_path+'/logs_'+Frost_str+'.txt', "w+")
logs.write("Random Seed: {} \n".format(manualSeed))
logs.write("Loaded Epoch {} and continuing training\n".format(this_Epoch))

Trainset_path = '/home/zhangxuanming/Kaggle265/train'
Testset_path = '/home/zhangxuanming/Kaggle265/test'
Validset_path = '/home/zhangxuanming/Kaggle265/valid'

batch_size = 256
num_epochs = 400

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
data_size = data_train.__len__()
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)

data_test = datasets.ImageFolder(root = Testset_path, transform = data_transform)
test_size = data_test.__len__()
test_loader = torch.utils.data.DataLoader(data_test, batch_size=test_size, shuffle=False, num_workers=4)

# 载入模型
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 265)
net = net.to(device)

if torch.cuda.device_count() > 1: 
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    logs.write("Now we use {} GPUs!\n".format(torch.cuda.device_count()))
net = nn.DataParallel(net)


if this_Epoch != 0 :
    net.load_state_dict(torch.load('%s/Epoch_%d.pth' % (checkpoint_path, this_Epoch), map_location=device))

# 就是普通的载入模型
optimizer = optim.Adam(
    [
        {'params': (p for name, p in net.named_parameters() if 'weight' in name), 'lr': 1e-3, 'momentum': 0.6, 'weight_decay': 1e-4},
        {'params': (p for name, p in net.named_parameters() if 'bias' in name), 'lr': 5e-3, 'momentum': 0.9, 'weight_decay': 1e-8}
    ]   , lr=1e-3, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[240, 360], gamma=0.2, last_epoch=-1)

criterion = nn.CrossEntropyLoss(reduction='mean')   # nn.MSELoss(reduction='mean')
loss_list = []
accrate_list = []
testloss_list = []
testaccrate_list = []

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
    net.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch_num, (inputs, labels) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # targets = F.one_hot(labels,265).float().to(device)

        this_loss, this_acc = Frost_iter(net, shadow_net, old_net, inputs, labels, criterion, optimizer, reg_lamb=0.)
        epoch_loss += this_loss
        epoch_acc += this_acc
    loss_list.append(epoch_loss/data_size*batch_size)
    accrate_list.append(epoch_acc/data_size)
    scheduler.step()

    ## Test
    net.eval()
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        for batch_num, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            test_pred = net(inputs)
            this_loss = criterion(F.softmax(test_pred, 1), labels)

            acc_mat = torch.argmax(test_pred, 1)==labels
            acc_time = torch.sum(acc_mat)

            test_loss += this_loss.item()
            test_acc += acc_time.item()
        testloss_list.append(test_loss)
        testaccrate_list.append(test_acc/test_size)

    if epoch % 1 == 0 :
        print('Epoch: {} \nAcc: {:.4f}, Loss: {:.4f}'.format(epoch, epoch_acc/data_size, epoch_loss/data_size*batch_size))
        print('Test Acc: {:.4f}, Test Loss: {:.4f}'.format(test_acc/test_size, test_loss))
        
        logs.write('\nEpoch: {} \nAcc: {:.4f}, Loss: {:.4f}\n'.format(epoch, epoch_acc/data_size, epoch_loss/data_size*batch_size))
        logs.write('Test Acc: {:.4f}, Test Loss: {:.4f}\n'.format(test_acc/test_size, test_loss))
        
        logs.flush()

    if epoch % 10 == 0 :
        torch.save(net.state_dict(), '%s/Epoch_%d.pth' % (checkpoint_path, this_Epoch+epoch))

logs.close()


plt.figure()
plt.plot(loss_list)
plt.plot(testloss_list)
plt.savefig(plot_path+'/loss_'+Frost_str+'.png')

plt.figure()
plt.plot(accrate_list)
plt.plot(testaccrate_list)
plt.savefig(plot_path+'/acc_'+Frost_str+'.png')