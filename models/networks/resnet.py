from operator import sub
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau 
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy




def ResnetGenerator(input_nc, output_nc, ngf=64, norm_layer=layers.BatchNormalization, use_dropout=False, n_blocks=6, padding_type='zero'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == tfa.layers.InstanceNormalization
        else:
            use_bias = norm_layer == tfa.layers.InstanceNormalization

        model = [
            layers.Input(shape=(None, None, input_nc)),
            ReflectionPad2D(3),
            layers.Conv2D(ngf, kernel_size=7, padding='valid', use_bias=use_bias),
            norm_layer(),
            layers.ReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [
                layers.ZeroPadding2D(padding=(1,1)),
                layers.Conv2D(ngf * mult * 2, kernel_size=3, strides=2, use_bias=use_bias),
                norm_layer(),
                layers.ReLU()
                ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [
                #layers.ZeroPadding2D(padding=(1,1)),
                layers.Conv2DTranspose(
                    int(ngf * mult / 2),
                    kernel_size=3, 
                    strides=2,
                    padding='same',
                    output_padding=1,
                    use_bias=use_bias),
                norm_layer(),
                layers.ReLU()
                ]
        model += [ReflectionPad2D(3)]
        model += [layers.Conv2D(output_nc, kernel_size=7)]
        model += [layers.Activation('tanh')]

        return tf.keras.Sequential(layers=model)


class ResnetBlock(layers.Layer):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad2D(1)]
        #elif padding_type == 'replicate':
        #    conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [layers.ZeroPadding2D(padding=(1,1))]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [layers.Conv2D(dim, kernel_size=3, use_bias=use_bias), norm_layer(), layers.ReLU()]
        if use_dropout:
            conv_block += [layers.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad2D(1)]
        #elif padding_type == 'replicate':
        #    conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [layers.ZeroPadding2D(padding=(1,1))]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [layers.Conv2D(dim, kernel_size=3, use_bias=use_bias), norm_layer()]

        return conv_block

    def call(self, x):
        """Forward function (with skip connections)"""
        x_ = x
        for layer_ in self.conv_block:
            x_ = layer_(x_)
        out = x + x_  # add skip connections
        return out