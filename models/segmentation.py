from models.networks import define_G
from data.image2mask_floodnet_dataset import Image2MaskFloodnetDataset
from models.networks.network_utils import get_scheduler
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import tensorflow_addons as tfa
import os

def getSegModel(opt):
    model = define_G(
        opt.input_nc, 
        opt.output_nc, 
        opt.ngf, 
        opt.netG, 
        opt.norm,
        not opt.no_dropout, 
        opt.init_type, 
        opt.init_gain, 
        opt.gpu_ids
        )
    
    train_ds, val_ds = Image2MaskFloodnetDataset.getTfDataset(opt)  # crea\]e a dataset given opt.dataset_mode and other options
    dataset = {
        'train': train_ds,
        'val': val_ds
    }
    print(opt.dataset_mode)
    
    scheduler = get_scheduler(opt)

    optimizer = Adam(learning_rate=opt.lr)

    binary_focal_cross_entropy_loss = tf.keras.losses.BinaryFocalCrossentropy(
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=tf.keras.losses.Reduction.AUTO,
        name='binary_focal_crossentropy'
    )

    hamming_loss = tfa.metrics.HammingLoss()

    model.compile(optimizer=optimizer, loss=binary_focal_cross_entropy_loss, metrics=[hamming_loss])

    print(opt.input_nc,opt.output_nc)

    callbacks = [
        scheduler,
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(os.path.join(opt.checkpoints_dir, opt.name), 'logs'),
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            write_steps_per_second=True,
            update_freq='epoch'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.path.join(opt.checkpoints_dir, opt.name), 'ckpt'),
            save_weights_only=True,
            monitor='val_hamming_loss',
            mode='min',
            save_best_only=True
        )
    ]

    return model, dataset, callbacks