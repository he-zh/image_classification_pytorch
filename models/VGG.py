from torch import nn
import torch.utils.model_zoo as model_zoo
import torchvision
import torch
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def basic_block(num_conv, in_channel, out_channel):
    blk=[]
    blk.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
    for i in range(1, num_conv):
        blk.append(nn.Conv2d(out_channel,out_channel, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # H/2,W/2
    return nn.Sequential(*blk)    

def vgg_block(conv_arch):
    block = nn.Sequential()
    for i, (num_conv, in_channel, out_channel) in enumerate(conv_arch):
        block.add_module("basic_block_"+str(i+1), basic_block(num_conv,in_channel,out_channel))
    
    return block


class VGG(nn.Module):
    def __init__(self, conv_arch, in_channel=3, out_channel=10, fc_features=512*7*7, fc_hidden_units=4096, drop_rate=0.5, init_weights=True):
        super(VGG, self).__init__()
        self.vgg_block = vgg_block(conv_arch)
        feature_channel = conv_arch[-1][-1]
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        fc_features = feature_channel * 7 *7
        self.classifier = nn.Sequential(
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(fc_hidden_units, out_channel)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.vgg_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg(pretrained=False, **kwargs):
    conv_arch = ((1,3,64),(1,64,128),(1,128,256))
    model = VGG(conv_arch,fc_hidden_units=2048, **kwargs)
    if pretrained == True:
        print("no pretrained model could be used, begin training from scratch")
    return model

def vgg11(pretrained=False, **kwargs):
    conv_arch = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
    model = VGG(conv_arch, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.vgg11(pretrained=pretrained)
        pre_model.classifier = model.classifier
        model = pre_model
        
    return model

def vgg13(pretrained=False, **kwargs):
    conv_arch = ((2, 3, 64), (2, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
    model = VGG(conv_arch, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.vgg13(pretrained=pretrained)
        pre_model.classifier = model.classifier
    return model

def vgg16(pretrained=False, **kwargs):
    conv_arch = ((2, 3, 64), (2, 64, 128), (3, 128, 256), (3, 256, 512), (3, 512, 512))
    model = VGG(conv_arch, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.vgg16(pretrained=pretrained)
        pre_model.classifier = model.classifier
    return model

def vgg19(pretrained=False, **kwargs):
    conv_arch = ((2, 3, 64), (2, 64, 128), (4, 128, 256), (4, 256, 512), (4, 512, 512))
    model = VGG(conv_arch, **kwargs)
    if pretrained == True:
        pre_model = torchvision.models.vgg19(pretrained=pretrained)
        pre_model.classifier = model.classifier
    return model

