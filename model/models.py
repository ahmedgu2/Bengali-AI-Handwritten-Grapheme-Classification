import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class ResNet34(nn.Module):

    def __init__(self, pretrained = True):
        super(ResNet34, self).__init__()

        if pretrained:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        self.fc1 = nn.Linear(512, 168)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 7)


    def forward(self, x):
        bs = x.shape[0]
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(bs, -1)
        
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)

        return y1, y2, y3


class SeResNext(nn.Module):
    
    def __init__(self, pretrained = True):
        super(SeResNext, self).__init__()

        if pretrained:
            self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)

        self.fc1 = nn.Linear(2048, 168)
        self.fc2 = nn.Linear(2048, 11)
        self.fc3 = nn.Linear(2048, 7)


    def forward(self, x):
        bs = x.shape[0]
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).view(bs, -1)
        
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)

        return y1, y2, y3