import torch
import torch.nn as nn
import torch.nn.functional as F

class Model8x8(nn.Module):
    def __init__(self):
        super(Model8x8, self).__init__()
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

class Model51x51(nn.Module):
    def __init__(self):
        super(Model51x51, self).__init__()
        self.conv1 = nn.Conv2d(1, 2500, 7, padding=3)
        self.conv2 = nn.Conv2d(2500, 1024, 5, padding=2)
        self.conv3 = nn.Conv2d(1024, 512, 3, padding=1)
        # self.conv4 = nn.Conv2d(512, 256, 3, padding=1)
        # self.conv5 = nn.Conv2d(256, 128, 3, padding=1)
        # self.conv6 = nn.Conv2d(128, 32, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        x = self.conv7(x)
        return x
