from tensorflow.python.keras import layers

import functools
import tensorflow as tf
import tensorflow_addons as tfa

def NLayerDiscriminator(input_nc, ndf=64, n_layers=3, norm_layer=layers.BatchNormalization):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == tfa.layers.InstanceNormalization
        else:
            use_bias = norm_layer == tfa.layers.InstanceNormalization

        kw = 4
        padw = 1

        input_ = layers.Input(shape=(None, None, input_nc))

        sequence = [
            layers.ZeroPadding2D(padding=(padw,padw)),
            layers.Conv2D(ndf, kernel_size=kw, strides=2, padding='valid'), 
            layers.LeakyReLU(0.2)
            ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                layers.ZeroPadding2D(padding=(padw,padw)),
                layers.Conv2D(ndf * nf_mult, kernel_size=kw, strides=2, padding='valid', use_bias=use_bias),
                norm_layer(),
                layers.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            layers.ZeroPadding2D(padding=(padw,padw)),
            layers.Conv2D(ndf * nf_mult, kernel_size=kw, strides=1, use_bias=use_bias),
            norm_layer(),
            layers.LeakyReLU(0.2)
        ]

        sequence += [
            layers.ZeroPadding2D(padding=(padw,padw)),
            layers.Conv2D(1, kernel_size=kw, strides=1, padding='valid')
            ]  # output 1 channel prediction map

        out_convs = []

        in_ = input_

        for layer_ in sequence:
            in_ = layer_(in_)
            if type(layer_) == tf.keras.layers.Conv2D:
                out_convs.append(layer_.output)
        
        return tf.keras.models.Model(inputs = input_, outputs = [in_]+out_convs)
