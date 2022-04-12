import torch 
import torch.nn as nn
import torch.nn.functional as fn


class Dummy(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Dummy, self).__init__()
        self.conv_1 =  nn.Conv2d(n_channels,1,kernel_size=(3,3),padding='same')

    def forward(self, x):
        return fn.sigmoid(self.conv_1(x))