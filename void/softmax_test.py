import torch

w = torch.rand(4)
ws = torch.softmax(w, 0)
print(w, ws)
