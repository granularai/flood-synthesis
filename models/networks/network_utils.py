from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau 
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay

import functools
import tensorflow as tf
import tensorflow_addons as tfa

class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, paddings=1):
        super(ReflectionPad2D, self).__init__()
        self.paddings = [paddings for i in range(4)]

    def call(self, input):
        l, r, t, b = self.paddings
        return tf.pad(input, paddings=[[0,0], [t,b], [l,r], [0,0]], mode='REFLECT')


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



class Identity(layers.Layer):
    def __init__(self):
        super(Identity, self).__init__()
    def call(self, x):
        return x


def get_scheduler(opt, platue_metric = ''):
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
    scheduler = None
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
        scheduler = ReduceLROnPlateau(monitor=platue_metric, mode='min', factor=0.2, min_delta=0.00001, patience=10)
    elif opt.lr_policy == 'cosine':
        def func():
            schedule = CosineDecay(opt.lr, decay_steps=opt.n_epochs, alpha=0)
            def lambda_rule(epoch, lr):
                return schedule(epoch)
            return lambda_rule
        scheduler = LearningRateScheduler(func())
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


'''
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
'''


