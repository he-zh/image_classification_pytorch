#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
# import torch.utils.model_zoo as model_zoo

class LeNet(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 10): # 3, 32, 32
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=5, stride=1, padding=0), # 6, 28, 28
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 6, 14, 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0), # 16, 10, 10
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16, 5, 5
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, out_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def lenet(pretrained=False, **kwargs):
    model = LeNet(**kwargs)
    if pretrained:
        print("no pretrained model could be used, begin training from scratch")
    return model