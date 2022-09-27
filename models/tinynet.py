from models.utils.misc import ForkedPdb
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .evonorm import EvoNormSample2d as evonorm_s0
import torch
from torch.nn import functional as F
__all__ = ['tinynet', 'tinynet5']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    
def normalization(planes, groups=2, norm_type='evonorm'):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(planes)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(groups, planes)
    elif norm_type == 'evonorm':
        return evonorm_s0(planes)
    else:
        raise NotImplementedError


class tinynet(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes=10, dataset='cifar10'):
        super(tinynet, self).__init__()
        self.features = features
        self.avgpool  = nn.AvgPool2d(4) #nn.Conv2d(128, 128, stride = 4, kernel_size=4, padding=0)
        self.linear  = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),)
            #nn.Linear(64, num_classes))
        self.classifier = nn.Linear(64, num_classes)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()


    def forward(self, x):
        x   = self.features(x)
        x   = self.avgpool(x)
        #print(x.size())
        x   = x.view(x.size(0), -1)
        x   = self.linear(x)
        out = self.classifier(x)
        return out

def make_layers(cfg, norm_type, groups, norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)] 
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if norm==True:
                norm_layer = normalization(planes=v, groups=groups, norm_type=norm_type)
                if norm_type=='evonorm':
                    layers += [conv2d, norm_layer]
                else:
                    layers+=[conv2d, norm_layer, nn.ReLU(inplace=True)]
            else:
                layers+=[conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [32, 'M', 64, 'M', 128, 'M' ],
    
}


def tinynet5(num_classes=10, dataset='cifar10', norm_type='evonorm', groups=2, norm=True):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    if "none" in norm_type.lower():
        norm = False
    return tinynet(make_layers(cfg['A'], norm_type, groups, norm), num_classes, dataset)