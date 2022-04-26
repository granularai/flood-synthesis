from tensorflow.python.keras import layers

import tensorflow as tf
import tensorflow_addons as tfa

def UnetGenerator(input_nc, output_nc, num_downs, ngf=64, norm_layer=layers.BatchNormalization, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        # construct unet structure
        convs = []
        input_ = layers.Input(shape=(None, None, input_nc))
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        convs += unet_block.convs
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            convs += unet_block.convs
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        return tf.keras.Sequential([input_, unet_block])

class UnetSkipConnectionBlock(tf.keras.layers.Layer):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=layers.BatchNormalization, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == tfa.layers.InstanceNormalization
        else:
            use_bias = norm_layer == tfa.layers.InstanceNormalization
        if input_nc is None:
            input_nc = outer_nc
        

        downconv = [
            layers.ZeroPadding2D(padding=(1,1)),
            layers.Conv2D(inner_nc, kernel_size=4,
                             strides=2, use_bias=use_bias)]

        downrelu = layers.LeakyReLU(0.2)
        downnorm = norm_layer()
        uprelu = layers.ReLU()
        upnorm = norm_layer()

        if outermost:
            upconv = [
                #layers.ZeroPadding2D(padding=(1,1)),
                layers.Conv2DTranspose(outer_nc,
                                        kernel_size=4, strides=2,padding='same',
                                        )
            ]
            down = [*downconv]
            up = [uprelu, *upconv, layers.Activation('tanh')]
            model = down + [self.submodule] + up
        elif innermost:
            upconv = [
                #layers.ZeroPadding2D(padding=(1,1)),
                layers.Conv2DTranspose(outer_nc,
                                        kernel_size=4, strides=2,padding='same',
                                        use_bias=use_bias),
            ]
            down = [downrelu, *downconv]
            up = [uprelu, *upconv, upnorm]
            model = down + up
        else:
            upconv = [
                #layers.ZeroPadding2D(padding=(1,1)),
                layers.Conv2DTranspose(outer_nc,
                                        kernel_size=4, strides=2,padding='same',
                                        use_bias=use_bias)
            ]
            down = [downrelu, *downconv, downnorm]
            up = [uprelu, *upconv, upnorm]

            if use_dropout:
                model = down + [self.submodule] + up + [layers.Dropout(0.5)]
            else:
                model = down + [self.submodule] + up

        #self.model = tf.keras.models.Sequential(layers = model)
        self.net = model

    def call(self, x):
        x_ = x
        for layer_ in self.net:
            x_ = layer_(x_)
        if self.outermost:
            return x_
        else:   # add skip connections
            return tf.concat([x, x_], axis=-1)
