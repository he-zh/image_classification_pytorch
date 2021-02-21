#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torchvision
def conv_block(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1) # H,W
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channel, out_channel):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channel + i * out_channel
            net.append(conv_block(in_c,out_channel))
        
        self.net = nn.ModuleList(net)
        self.out_channel = in_channel + num_convs * out_channel

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X,Y),1)
        return X

def transition_block(in_channel, out_channel):
    return nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2) # out_channel, H/2, W/2
    )

class DenseNet(nn.Module):
    def __init__(self, num_blocks=(6,12,24,16), in_channel=3, out_channel=10, num_init_features=64,
                growth_rate=32):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channel, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        num_features = num_init_features
        for i,num_layers in enumerate(num_blocks):
            block = DenseBlock(num_layers, num_features, growth_rate) 
            self.features.add_module('dense_block%d'%(i+1), block)
            num_features = num_features + num_layers * growth_rate 
            if i != len(num_blocks)-1 :
                self.features.add_module('transition_block%d'%(i+1), transition_block(num_features, num_features//2))
                num_features = num_features//2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        self.relu = nn.ReLU(inplace=True)

        self.classifer = nn.Linear(num_features, out_channel)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        

    def forward(self, x):

        features = self.features(x)
        out = self.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifer(out)

        return out

def densenet(pretrained=False, **kwargs):
    model = DenseNet((6,12,8), growth_rate=32, num_init_features=64, **kwargs)
    if pretrained == True:
        print("no pretrained model could be used, begin training from scratch")
    return model

def densenet121(pretrained=False, **kwargs):
    model = DenseNet((6,12,24,16), growth_rate=32, num_init_features=64, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.densenet121(pretrained=pretrained)
        pre_model.classifier = model.classifier
    return model

def densenet161(pretrained=False, **kwargs):
    model = DenseNet((6,12,36,24), growth_rate=48, num_init_features=96, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.densenet161(pretrained=pretrained)
        pre_model.classifier = model.classifier
    return model

def densenet169(pretrained=False, **kwargs):
    model = DenseNet((6,12,32,32), growth_rate=32, num_init_features=64, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.densenet169(pretrained=pretrained)
        pre_model.classifier = model.classifier
    return model

def densenet201(pretrained=False, **kwargs):
    model = DenseNet((6,12,48,32), growth_rate=32, num_init_features=64, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.densenet201(pretrained=pretrained)
        pre_model.classifier = model.classifier
    return model
