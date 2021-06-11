# Elastic Weight Consolidation
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch import autograd

# remove EWC !!!
# from EWC_class import ElasticWeightConsolidation
from EWC.EWC_class import ElasticWeightConsolidation

# 固定随机种子
manualSeed = 2077   # random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def accu(model, lastlayer, dataloader, device):
    model = model.eval()    # .to(device)
    acc = 0
    for input, target in dataloader:
        o = lastlayer(model(input.to(device)))
        acc += (o.argmax(dim=1).long() == target.to(device)).float().mean()
    return acc / len(dataloader)

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        return self.act(self.lin(x))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class BaseModel(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(BaseModel, self).__init__()
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=True)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=True)
        # self.lin3 = nn.Linear(num_hidden, num_outputs)
    def forward(self, x):
        return self.lin2(self.lin1(self.f1(x)))

class LastLayer(nn.Module):
    def __init__(self, num_hidden, num_outputs):
        super(LastLayer, self).__init__()
        self.lin = nn.Linear(num_hidden, num_outputs)
    def forward(self, x):
        return self.lin(x)


crit = nn.CrossEntropyLoss()
mynet = BaseModel(28*28, 100).to(device)
last1 = LastLayer(100, 10).to(device)
last2 = LastLayer(100, 10).to(device)
optimizer = optim.Adam(
    [
        {'params': mynet.parameters(), 'lr': 1e-3, 'momentum': 0.6, 'weight_decay': 1e-8},
        {'params': last1.parameters(), 'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 1e-8},
        {'params': last2.parameters(), 'lr': 1e-3, 'momentum': 0.9, 'weight_decay': 1e-8},
    ]   , lr=1e-3, weight_decay=1e-8
)
ewc = ElasticWeightConsolidation(mynet, crit=crit, optimizer=optimizer, device=device)

for _ in range(4):
    for input, target in tqdm(train_loader):
        ewc.forward_backward_update(input, target, last1)
task1_acc = accu(ewc.model, last1, test_loader, device).item()
print('Task1_Acc = ', task1_acc)

ewc.register_ewc_params(mnist_train, 100, 300)

f_mnist_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms.ToTensor())
f_mnist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transforms.ToTensor())
f_train_loader = DataLoader(f_mnist_train, batch_size = 100, shuffle=True)
f_test_loader = DataLoader(f_mnist_test, batch_size = 100, shuffle=False)

for _ in range(4):
    for input, target in tqdm(f_train_loader):
        ewc.forward_backward_update(input, target, last2)

ewc.register_ewc_params(f_mnist_train, 100, 300)

task2_acc = accu(ewc.model, last2, f_test_loader, device).item()
task1_acc = accu(ewc.model, last1, test_loader, device).item()
print('Task2_Acc = ', task2_acc)
print('Task1_Acc = ', task1_acc)