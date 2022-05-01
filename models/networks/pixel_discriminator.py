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



def PixelDiscriminator(input_nc, ndf=64, norm_layer=layers.BatchNormalization):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        input_ = layers.Input(shape=(None, None, input_nc))
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == tfa.layers.InstanceNormalization
        else:
            use_bias = norm_layer == tfa.layers.InstanceNormalization

        net = [
            layers.Conv2D(ndf, kernel_size=1, strides=1, padding='valid'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(ndf * 2, kernel_size=1, strides=1, padding='valid', use_bias=use_bias),
            norm_layer(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=1, strides=1, padding='valid', use_bias=use_bias)
            ]
        out_convs = []

        for layer_ in net:
            in_ = layer_(in_)
            if type(layer_) == tf.keras.layers.Conv2D:
                out_convs.append(layer_.output)
        
        return tf.keras.models.Model(inputs = input_, outputs = [in_]+out_convs)
