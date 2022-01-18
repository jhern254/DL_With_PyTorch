import os
import numpy as np
import collections
import logging
import datetime
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)

#class NetDepth(nn.Module):
#    def __init__(self, n_chans1=32):
#        super().__init__()
#        self.n_chans1 = n_chans1
#        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
#        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, 3, 1)
#        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2, 3, 1)
#        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32) # TODO: make note of how to get first arg
#        self.fc2 = nn.Linear(32, 2)
#
#    def forward(self, x):
#        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
#        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
#        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
#        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
#        out = torch.relu(self.fc1(out))
#        out = self.fc2(out)
#        return out

class NetDepth(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 2,
                               kernel_size=3, padding=1)
        # 16 x 16
        self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
        out = out.view(-1, 4 * 4 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out



def SayHello():
    print("Hello World\n")








