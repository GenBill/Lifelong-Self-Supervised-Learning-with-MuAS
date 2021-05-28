from numpy import copy
import torch

def BW_tensor(this_tensor, divide):
    # divide : 将超过min(1/div)的数值设置为不更新
    length = this_tensor.numel()
    temp_tensor = this_tensor.view(-1)
    threshold = torch.topk(temp_tensor, length//divide, largest=False, sorted=True).values[-1]
    return this_tensor<=threshold

def Get_BW_net(net, device):
    shadow_net = copy.deepcopy(net)
    for params in shadow_net.parameters():
        params.data = BW_tensor(params.data, 2)
    print("Shadow Run !")
    return shadow_net

def params_diff(net, old_net):
    # zero part of _grad
    params_net = list(net.named_parameters())
    params_old = list(old_net.named_parameters())
    loss = 0
    for i in range(len(params_net)):
        loss += sum(abs(params_net[i][1].data - params_old[i][1].data))
    return loss

def Frost_iter(net, shadow_net, old_net, X, target, loss_Func, optimizer, reg_lamb=5):
    # 计算基本损失
    pred = net(X)
    loss = loss_Func(pred, target)

    # 添加遗忘损失
    params_net = list(net.named_parameters())
    params_old = list(old_net.named_parameters())
    for i in range(len(params_net)):
        loss += reg_lamb * sum(abs(params_net[i][1].data - params_old[i][1].data))
    
    # 计算 grad
    optimizer.zero_grad()
    loss.backward()

    # zero part of _grad
    params_net = list(net.named_parameters())
    params_shadow = list(shadow_net.named_parameters())
    for i in range(len(params_net)):
        print(params_net[i][1].grad)    # data
        params_net[i][1].grad = params_net[i][1].grad * params_shadow[i][1].data.float()
        print(params_net[i][1].grad)    # data

    # 
    optimizer.step()

    # 明天用简单的Net写个测试