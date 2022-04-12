import numpy as np
import torch
import torchvision

from torch import nn

class Seg1(nn.Module):
    def __init__(self,backbone,num_out=10,pretrained=False):
        super(Seg1,self).__init__()
        if backbone == 'resnet50':
            self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained,progress=True)
        elif backbone == 'resnet101':
            self.model = torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained, progress=True)
        self.sigm = nn.Sigmoid()
        _ = self.model.classifier[-1]
        self.model.classifier[-1] = nn.Conv2d(_.in_channels,num_out,kernel_size=_.kernel_size,stride=_.stride)
        nn.init.xavier_uniform_(self.model.classifier[-1].weight)
    def forward(self,x):
        x = self.model(x)
        try:
            if 'out' in x.keys():
                x = x['out']
        except:
            pass
        return self.sigm(x)