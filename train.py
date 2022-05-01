"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from models.pix2pix_model import getPix2PixModel
from models.segmentation import getSegModel

import time
import os
import tensorflow as tf

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)



if __name__ == '__main__':
    print("parsing args")
    opt = TrainOptions().parse()   # get training options
    
    print('----------------------mode------------------------')    
    if not opt.mode == 'seg':
        model, dataset, callbacks = getPix2PixModel(opt)
    else:
        model, dataset, callbacks = getSegModel(opt)

    model.fit(
        dataset['train'],
        validation_data = dataset['val'],
        verbose=1,
        callbacks = callbacks,
        initial_epoch = opt.epoch_count,
        epochs = opt.n_epochs,
    )
