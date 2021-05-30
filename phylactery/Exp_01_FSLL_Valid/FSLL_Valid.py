from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader

# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")

# copy from Frost_Func.py
# Sector Frost_Func
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
    acc_mat = torch.argmax(pred, 1)==torch.argmax(target, 1)
    acc_time = torch.sum(acc_mat)

    return loss.item(), acc_time.item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 任务：CIFAR10，CUB200（暂定）
# ResNet-18 装载
this_Epoch = 0
checkpoint_path = 'E:/Laplace/Dataset/Kaggle265/models' # './TimHu/models'

Trainset_path = 'E:/Laplace/Dataset/Kaggle265/train'    # './Kaggle265/train'
Testset_path = 'E:/Laplace/Dataset/Kaggle265/test'      # './Kaggle265/test'
Validset_path = 'E:/Laplace/Dataset/Kaggle265/valid'    # './Kaggle265/valid'

batch_size = 64
num_epochs = 200

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
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)



# 载入模型
net = models.resnet50(pretrained=True)

if this_Epoch != 0 :
    net.load_state_dict(torch.load('%s/Epoch_%d.pth' % (checkpoint_path, this_Epoch), map_location=device))

# 就是普通的载入模型
optimizer = optim.SGD(
    [
        {'params': net.parameters(), 'lr': 0.1, 'momentum': 0.8, 'weight_decay': 0.}
    ]   , lr=1e-1, momentum=0.8, weight_decay=0.0001
)
criterion = nn.MSELoss(reduction='mean')
loss_list = []
accrate_list = []

# Train Step
# 镜像
old_net = Get_old_net(net)
shadow_net = Get_shadow_net(net, 2)

# Epoch On !
for epoch in tqdm(range(1,num_epochs)):

    for batch_num, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        this_loss, this_acc = Frost_iter(net, shadow_net, old_net, inputs, labels, criterion, optimizer, reg_lamb=1e-2)
        loss_list.append(this_loss)
        accrate_list.append(this_acc/batch_size)

        if epoch % 10 == 0 :
            print('Epoch:', epoch, ', acc:', this_acc/batch_size, ', loss:', this_loss)
        if epoch % 1000 == 0 :
            torch.save(net.state_dict(), '%s/Epoch_%d.pth' % (checkpoint_path, this_Epoch+epoch))


