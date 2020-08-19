import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 150, 3, padding=1)
        self.conv2 = nn.Conv2d(150, 150, 3, padding=1)
        self.conv3 = nn.Conv2d(150, 150, 3, padding=1)
        self.conv4 = nn.Conv2d(150, 1, 3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x