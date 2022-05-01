
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
        self.ds_len = None #Mask2ImageFloodnetDataset.getDataLen(opt)

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
