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
import time
import tensorflow as tf
from options.train_options import TrainOptions
#from data import create_dataset
from data.mask2image_floodnet_dataset import Mask2ImageFloodnetDataset
from models.pix2pix_model import Pix2PixModel
#from models import create_model
from util.visualizer import Visualizer
from torch.nn import DataParallel as DP
import os
import matplotlib.pyplot as plt
import numpy as np

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

class visualCallBack(tf.keras.callbacks.Callback):
    def __init__(self, opt, model):
        super(visualCallBack, self).__init__()
        self.total_iters = 0
        self.epoch_iter = 0
        self.epoch = 0
        self.iter_start_time = 0
        self.opt = opt
        self.batch_size_ = opt.batch_size
        self.visualizer = Visualizer(opt)
        self.model = model
        self.ds_len = Mask2ImageFloodnetDataset.getDataLen(opt)

    def on_train_batch_start(self, batch, logs=None):
        self.iter_start_time = time.time()
    
    def on_train_batch_end(self, batch, logs=None):
        self.total_iters += self.batch_size_
        self.epoch_iter += self.batch_size_
        opt = self.opt

        if self.total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            self.model.compute_visuals()
            self.visualizer.display_current_results(model.get_current_visuals(), self.epoch, save_result)

        #if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
        #    losses = self.model.get_current_losses()
        #    t_comp = (time.time() - self.iter_start_time) / opt.batch_size
        #    self.visualizer.print_current_losses(self.epoch, self.epoch_iter, losses, t_comp, 0.0)
        #    if opt.display_id > 0:
        #        visualizer.plot_current_losses(self.epoch, float(self.epoch_iter) / self.ds_len, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (self.epoch, self.total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            self.model.save_networks(save_suffix)

    def on_epoch_begin(self, epoch, logs = None):
        self.epoch = epoch
        self.epoch_iter = 0

    def on_epoch_end(self, epoch, logs = None):
        opt = self.opt
        if self.epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (self.epoch, self.total_iters))
            self.model.save_networks('latest')
            self.model.save_networks(epoch)



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = Mask2ImageFloodnetDataset.getTfDataset(opt)  # create a dataset given opt.dataset_mode and other options
    print(opt.dataset_mode)
    print('.................')
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    #print('The number of training images = %d' % dataset_size)
    #for sample in dataset:
    #    fig, axes = plt.subplots(nrows=1, ncols=2)
    #    print(np.unique(sample[1][0].numpy(), return_counts=True))
    #    axes[0].imshow(sample[0][0][:,:,1:4])
    #    axes[1].imshow(sample[1][0])
    #    plt.show()
    #fg
    model = Pix2PixModel(opt)              # regular setup: load and print networks; create schedulers
    model.compile()
    print(opt.input_nc,opt.output_nc)
    print('..........')
    #if torch.cuda.device_count() > 1:
    #    print("DataParallel Training!!!!!!!!!!!!!!!!!!")
    #    model = DP(model, device_ids=[i for i in range(torch.cuda.device_count())])
    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    #TODO create visualizer callback
    total_iters = 0                # the total number of training iterations

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

    for data in dataset:
        print(len(data))
        #print(data[0].shape)
        print(data[1].shape)
        break

    model.fit(
        dataset,
        verbose=1,
        callbacks = callbacks,
        initial_epoch = opt.epoch_count,
        epochs = opt.n_epochs + opt.n_epochs_decay + 1,
    )

    '''
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    '''