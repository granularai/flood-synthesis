from tensorflow.keras.losses import MeanAbsoluteError, Huber
from loss import GANLoss, ContentLoss
from tensorflow.keras.optimizers import Adam
from models.base_model import BaseModel
from models.networks import define_D, define_G, get_scheduler
from data.mask2image_floodnet_dataset import Mask2ImageFloodnetDataset

import tensorflow as tf
import os


class MetricTest(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(MetricTest, self).__init__(name=name, **kwargs)
        self.value = self.add_weight(name='total_loss', initializer='zeros')
        self.cnt = self.add_weight(name='cnt', initializer='zeros')

    def update_state(self, value):
        self.value.assign_add(tf.reduce_sum(value))
        self.cnt.assign_add(tf.ones(()))

    def result(self):
        return self.value/ self.cnt

        


class Pix2PixModel(BaseModel, tf.keras.models.Model):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True, reduction=None):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt, reduction=tf.keras.losses.Reduction.AUTO):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        tf.keras.models.Model.__init__(self)
        BaseModel.__init__(self, opt)

        self.isTrain = True
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode, reduction=reduction)
            self.criterionL1 = MeanAbsoluteError()
            self.criterionHuber = Huber()
            self.ctriterionContent = ContentLoss(perceptual = False)
            self.criterionPerceptual = ContentLoss(perceptual=True)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = Adam(learning_rate=opt.lr, beta_1=opt.beta1, beta_2 = 0.999)
            self.optimizer_D = Adam(learning_rate=opt.lr, beta_1=opt.beta1, beta_2 = 0.999)
            #self.optimizers.append(self.optimizer_G)
            #self.optimizers.append(self.optimizer_D)
            self.schedulers = get_scheduler(opt)
        
        self.metric_d_real = MetricTest()
        self.metric_d_fake = MetricTest()
        self.metric_d_total = MetricTest()
        self.metric_g_gan = MetricTest()
        # setting 1
        self.metric_g_l1 = MetricTest()
        #setting 2
        self.metric_g_huber = MetricTest()
        self.metric_g_content = MetricTest()
        self.metric_g_perceptual = MetricTest()
        self.metric_g_total = MetricTest()
        self.metric_total = MetricTest()

        #self.print_networks()
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.optimizer_G,
            discriminator_optimizer=self.optimizer_D,
            generator=self.netG,
            discriminator=self.netD
            )
        self.print_networks()

    def call(self, data):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_A = data[0]
        self.real_B = data[1]
        self.fake_B = self.netG(self.real_A)  # G(A)

    def lossNetwork(self):
        pass

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        self.fake_AB = tf.concat([self.real_A, self.fake_B], axis=-1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(self.fake_AB, training=True)[0]
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        self.real_AB = tf.concat([self.real_A, self.real_B], axis=-1)
        pred_real = self.netD(self.real_AB, training=True)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        self.fake_AB = tf.concat([self.real_A, self.fake_B], axis=-1)
        self.real_AB = tf.concat([self.real_A, self.real_B], axis=-1)
        out = self.netD(self.fake_AB, training=False)
        out_ = None
        if self.opt.mode == 'v2':
            out_ = self.netD(self.real_AB, training=False)
        pred_fake = out[0]
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_
        # Second, G(A) = B
        if self.opt.type == 'original':
            self.loss_G_L1 = self.criterionL1(self.real_B, self.fake_B) * self.opt.eta_
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        else:
            self.loss_G_huber = self.criterionHuber(self.real_B, self.fake_B) * self.opt.eta_
            self.loss_G_content = self.ctriterionContent(out_[1:], out[1:]) * self.opt.delta_
            self.loss_G_perceptual = self.criterionPerceptual() * self.opt.gamma_
            self.loss_G = self.loss_G_GAN + self.loss_G_huber + self.loss_G_content + self.loss_G_perceptual


    #@tf.function
    def train_step(self, data):                  
        # compute fake images: G(A)
        
        # update D
        
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            gen_tape.watch(data[0])
            gen_tape.watch(data[1])
            disc_tape.watch(data[0])
            disc_tape.watch(data[1])

            self(data, training=True)
            self.backward_D()                
            self.backward_G()
        disc_gradients = disc_tape.gradient(self.loss_D, self.netD.trainable_variables)
        gen_gradients = gen_tape.gradient(self.loss_G, self.netG.trainable_variables)

        self.optimizer_D.apply_gradients(zip(disc_gradients, self.netD.trainable_variables))

        self.optimizer_G.apply_gradients(zip(gen_gradients, self.netG.trainable_variables))

        self.metric_d_real.update_state(self.loss_D_real)
        self.metric_d_fake.update_state(self.loss_D_fake)
        self.metric_d_total.update_state(self.loss_D)
        self.metric_g_gan.update_state(self.loss_G_GAN)
        if self.opt.mode == 'v1':
            self.metric_g_l1.update_state(self.loss_G_L1)
        else:
            self.metric_g_huber.update_state(self.loss_G_huber)
            self.metric_g_content.update_state(self.loss_G_content)
            self.metric_g_perceptual.update_state(self.loss_G_perceptual)

        self.metric_g_total.update_state(self.loss_G)
        self.metric_total.update_state(self.loss_D+self.loss_G)

        loss_log = {
            'loss_D_fake': self.metric_d_fake.result(),
            'loss_D_real': self.metric_d_real.result(),
            'loss_G_GAN': self.metric_g_gan.result(),
        }

        if self.opt.mode == 'v1':
            loss_log.update({
                'loss_G_L1': self.metric_g_l1.result()
            })
        else:
            loss_log.update({
                'loss_G_huber': self.metric_g_huber.result(),
                'loss_G_content': self.metric_g_content.result(),
                'loss_G_perceptual': self.metric_g_perceptual.result()
            })
        loss_log.update({
            'loss_D_total': self.metric_d_total.result(),
            'loss_G_total': self.metric_g_total.result(),
            'loss_total': self.metric_total.result()
        })
        return loss_log
    
    @property
    def metrics(self):
        out_metrics = [
            self.metric_d_real,
            self.metric_d_fake,
            self.metric_d_total,
            self.metric_g_gan
            ]
        if self.opt.mode == 'v1':
            out_metrics.append(self.metric_g_l1)
        else:
            out_metrics += [
                self.metric_g_huber,
                self.metric_g_content,
                self.metric_g_perceptual
                ]
        out_metrics += [
            self.metric_g_total,
            self.metric_total
            ]
        return out_metrics


def getPix2PixModel(opt):
    train_ds, val_ds = Mask2ImageFloodnetDataset.getTfDataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset = {
        'train': train_ds,
        'val': val_ds
    }
    print(opt.dataset_mode)
    model = Pix2PixModel(opt)              # regular setup: load and print networks; create schedulers
    model.compile()
    print(opt.input_nc,opt.output_nc)

    callbacks = [
        model.schedulers,
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(os.path.join(opt.checkpoints_dir, opt.name), 'logs'),
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            write_steps_per_second=True,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        ) #,
        #visualCallBack(opt, model)
    ]

    return model, dataset, callbacks

