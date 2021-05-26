from numpy import copy
import torch

def BW_tensor(this_tensor, divide):
    # divide : 将超过min(1/div)的数值设置为不更新
    length = this_tensor.numel()
    temp_tensor = this_tensor.view(-1)
    threshold = torch.topk(temp_tensor, length//divide, largest=False, sorted=True).values[-1]
    return this_tensor<=threshold

def Get_BW_net(net, device):
    shadow_new = copy.deepcopy(net)
    for params in shadow_new.parameters():
        params.data = BW_tensor(params.data, 2)
    print("Shadow Run !")
    return shadow_new

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A = torch.rand(3,4)

print(A)
print(BW_tensor(A, 2))