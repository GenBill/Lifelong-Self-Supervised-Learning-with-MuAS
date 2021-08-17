import torch
import torch.nn as nn
import random

outputs = torch.tensor([0.1, 0.1, 0.9, 0.9])
labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
print(outputs, labels)

loss = nn.BCELoss()(outputs, labels)
print('loss = ', loss.item())

ra = random.randint(0, 7)
print(ra)