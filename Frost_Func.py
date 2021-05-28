from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def print_params(net):
    params = list(net.named_parameters())   # get the index by debuging
    for i in range(len(params)):
        print(i, params[i][0])      # name
        print(params[i][1].data)    # data

def BW_tensor(this_tensor, divide):
    # divide : 将超过min(1/div)的数值设置为不更新
    length = this_tensor.numel()
    temp_tensor = torch.abs(this_tensor.view(-1))
    Kmin_list = torch.topk(temp_tensor, length//divide, largest=False, sorted=True)
    if len(Kmin_list.values)!=0 :
        threshold = Kmin_list.values[-1]
        return (torch.abs(this_tensor)<=threshold).float()
    else :
        return torch.zeros(1)    # this_tensor<0 + this_tensor>=0

def Get_shadow_net(net, divide=2):
    shadow_net = copy.deepcopy(net)
    for params in shadow_net.parameters():
        params.data = BW_tensor(params.data, divide)
        params.requires_grad_(False)
    return shadow_net

def Get_old_net(net):
    old_net = copy.deepcopy(net)
    for params in old_net.parameters():
        params.requires_grad_(False)
    return old_net

'''
def params_diff(net, old_net):
    # zero part of _grad
    params_net = list(net.named_parameters())
    params_old = list(old_net.named_parameters())
    loss = 0
    for i in range(len(params_net)):
        loss += sum(abs(params_net[i][1].data - params_old[i][1].data))
    return loss
'''

def Frost_iter(net, shadow_net, old_net, X, target, loss_Func, optimizer, reg_lamb=5):
    # 构造参数索引 params_list
    params_net = list(net.named_parameters())
    params_old = list(old_net.named_parameters())
    params_shadow = list(shadow_net.named_parameters())
    
    # 计算基本损失
    pred = net(X)
    loss = loss_Func(pred, target)

    # 添加遗忘损失
    for i in range(len(params_net)):
        loss += reg_lamb * torch.sum(torch.abs(params_net[i][1] - params_old[i][1]))
    
    # 计算 grad
    optimizer.zero_grad()
    loss.backward()

    # 过滤 grad
    for i in range(len(params_net)):
        params_net[i][1].grad = params_net[i][1].grad * params_shadow[i][1].data
    
    # End Sub
    optimizer.step()

# 简单的Net测试
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class MyTemplateNet(nn.Module):
    def __init__(self):
        super(MyTemplateNet, self).__init__() # 第一句话，调用父类的构造函数
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

net = MyTemplateNet().to(device) # 构造模型
optimizer = optim.SGD(
    [
        {'params': net.parameters(), 'lr': 0.1, 'momentum': 0.8, 'weight_decay': 0.}
    ]   , lr=1e-1, momentum=0.8, weight_decay=0.0001
)
criterion = nn.MSELoss(reduction='mean')

old_net = Get_old_net(net)
shadow_net = Get_shadow_net(net, 2)

# Epoch On !
print('pre-net : ')
print_params(net)

X_torch = torch.rand(1,2).float().to(device)
Y_torch = torch.rand(1,1).float().to(device)
Frost_iter(net, shadow_net, old_net, X_torch, Y_torch, criterion, optimizer, reg_lamb=1e-2)

X_torch = torch.rand(1,2).float().to(device)
Y_torch = torch.rand(1,1).float().to(device)
Frost_iter(net, shadow_net, old_net, X_torch, Y_torch, criterion, optimizer, reg_lamb=1e-2)

X_torch = torch.rand(1,2).float().to(device)
Y_torch = torch.rand(1,1).float().to(device)
Frost_iter(net, shadow_net, old_net, X_torch, Y_torch, criterion, optimizer, reg_lamb=1e-2)

X_torch = torch.rand(1,2).float().to(device)
Y_torch = torch.rand(1,1).float().to(device)
Frost_iter(net, shadow_net, old_net, X_torch, Y_torch, criterion, optimizer, reg_lamb=1e-2)

print('trained-net : ')
print_params(net)

