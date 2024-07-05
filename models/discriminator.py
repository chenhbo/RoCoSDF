"""Discriminator from UNSR: https://arxiv.org/abs/2401.05915.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from torch.nn import Conv1d,Conv2d


############Define Discriminator network based on UNSR#######################
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x  

    def sdf(self, x):
        return self.forward(x) 
    
