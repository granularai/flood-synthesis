import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau 
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay
from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy


###############################################################################
# Helper Functions
###############################################################################


class Identity(layers.Layer):
    def __init__(self):
        super(Identity, self).__init__()
    def call(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(layers.BatchNormalization, scale=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(tfa.layers.InstanceNormalization, scale=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch, lr):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = LearningRateScheduler(lambda_rule)
    elif opt.lr_policy == 'step':
        def func():
            schedule = ExponentialDecay(opt.lr, decay_steps=opt.lr_decay_iters, decay_rate=0.1)
            def lambda_rule(epoch, lr):
                return schedule(epoch)
            return lambda_rule
        scheduler = LearningRateScheduler(func())
    elif opt.lr_policy == 'plateau':
        # TODO: replace val loss with appropriate metric
        scheduler = ReduceLROnPlateau(monitor='loss_total', mode='min', factor=0.2, min_delta=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        def func():
            schedule = CosineDecay(opt.lr, decay_steps=opt.n_epochs, alpha=0)
            def lambda_rule(epoch, lr):
                return schedule(epoch)
            return lambda_rule
        scheduler = LearningRateScheduler(func())
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_512':
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return net #init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return net #init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(tf.keras.losses.Loss):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, reduction=None):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = MeanSquaredError()
        elif gan_mode == 'vanilla':
            self.loss = BinaryCrossentropy(from_logits=True)
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return tf.ones_like(prediction)*target_tensor

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = - tf.reduce_mean(prediction)
            else:
                loss = tf.reduce_mean(prediction)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = tf.random.uniform(shape=(real_data.shape[0], 1))
            alpha = tf.reshape(tf.boadcast_to(alpha, real_data.shape[0], tf.size(real_data) // real_data.shape[0]),real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv = tf.Variable(interpolatesv)
        disc_interpolates = netD(interpolatesv)
        # TODO: add gradient tape
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, paddings=1):
        super(ReflectionPad2D, self).__init__()
        self.paddings = [paddings for i in range(4)]

    def call(self, input):
        l, r, t, b = self.paddings
        return tf.pad(input, paddings=[[0,0], [t,b], [l,r], [0,0]], mode='REFLECT')


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
        input_ = layers.Input(shape=(None, None, input_nc))
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
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
            model = down + [submodule] + up
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
                model = down + [submodule] + up + [layers.Dropout(0.5)]
            else:
                model = down + [submodule] + up

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
        sequence = [
            layers.Input(shape=(None, None, input_nc)),
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
        return tf.keras.Sequential(layers = sequence)



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
            input_,
            layers.Conv2D(ndf, kernel_size=1, strides=1, padding='valid'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(ndf * 2, kernel_size=1, strides=1, padding='valid', use_bias=use_bias),
            norm_layer(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, kernel_size=1, strides=1, padding='valid', use_bias=use_bias)
            ]
        return tf.keras.Sequential(layers = net)
