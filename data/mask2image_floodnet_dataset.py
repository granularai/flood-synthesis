import os
import glob
import numpy as np
import random
import cv2
import torch
import pandas as pd
import tqdm
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
from tensorflow.keras import layers
from functools import partial

import tensorflow as tf

from data.base_dataset import BaseDataset, get_params, get_transform

from PIL import Image

class Mask2ImageFloodnetDataset(BaseDataset):
    def __init__(self, opt):
        super(Mask2ImageFloodnetDataset, self).__init__(opt)
        data_path = '/mnt/now/houston/processed/train'

        self.imgs = []

        x_dir = data_path+'/x/*'
        y_dir = data_path+'/y/'
        
        imgs = glob.glob(x_dir)
        print(data_path)
        for img in tqdm.tqdm(imgs):
            name = img.split('/')[-1].split('.')[0]
            self.imgs.append((img,y_dir+name+'.npy'))
        print("dataset process finished")
        print(len(self.imgs))
        
        self.input_nc = 26
        opt.input_nc = 26
        self.output_nc = 3
        opt.output_nc = 3
        self.num_classes=26

        self.height, self.width = 512, 512
    
    def __len__(self):
        return len(self.imgs)
    
    def __getG__(self, idx):
        yield self.__getitem__(idx)
    
    def __getGen__(self, idx):
        return tf.data.Dataset.from_generator(
            self.__getG__, 
            output_signature=(
                tf.TensorSpec(shape=(None, None, 26), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
                ),
            args=(idx,)
            )

    def __getitem__(self, idx):
        x,y = self.imgs[idx]
        img_x = np.load(x) #Image.open(x).convert('RGB')
        img_y = np.load(y) #Image.open(y)

        #img_x = img_x.resize((self.width,self.height), resample = Image.BILINEAR)
        #img_y = img_y.resize((self.height,self.width), resample = Image.NEAREST)
        ## split AB image into A and B
        #img_x = tf.keras.utils.img_to_array(img_x)
        #img_y = tf.keras.utils.img_to_array(img_y)
        #print(img_x.shape)
        #print(img_y.shape)
        return img_y, img_x
    
    @tf.function
    def getAugs(self, image_x, image_y):
        if tf.random.uniform(()) > 0.5:
            flip_lr = tf.random.uniform(()) > 0.5
            flip_tb = tf.random.uniform(()) > 0.5
            if flip_lr:
                image_x = tf.image.flip_left_right(image_x)
                image_y = tf.image.flip_left_right(image_y)
            if flip_tb:
                image_x = tf.image.flip_up_down(image_x)
                image_y = tf.image.flip_up_down(image_y)
        return image_x, image_y
        #return tf.one_hot(
        #    tf.cast(image_x, dtype=tf.int32), 
        #    depth=self.num_classes,
        #    axis = -1, 
        #    dtype=tf.float32
        #)[:,:,0,:], image_y
    @tf.function
    def preprocessMask(self, x, y):
        #tf.print(x.shape)
        #tf.print(y.shape)
        house = y*x[:,:,1,None]
        x = tf.concat([x[:,:,:1], house, x[:,:,2:]], axis=-1)
        return x
    
    @classmethod
    def getDataLen(cls, opt):
        ds = cls(opt)
        return (len(ds)//opt.batch_size)*opt.batch_size

    def partition(self):
        #naive
        total_sz = self.__len__()
        shuffled_inds = np.random.permutation(total_sz)
        

    @classmethod
    def getTfDataset(cls, opt):
        ds = cls(opt)
        #resizing = partial(layers.Resizing, height=512, width=512)
        #resize_x = resizing(interpolation='nearest')
        #resize_y = resizing(interpolation='bilinear')
        #normalizing = layers.Normalization(mean=(0.0,0.0,0.0), variance=(255.0,255.0,255.0))

        train_ds = tf.data.Dataset.range(
            len(ds)
        ).shuffle(
            len(ds)*4
        ).interleave(
            lambda i: ds.__getGen__(i),
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache(
        ).map(
            lambda x,y : (x, y/255.0), #normalizing(y)),
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            ds.getAugs,
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            lambda x,y : (
                x, #ds.preprocessMask(x,y),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(
            opt.batch_size,
            drop_remainder=True
        ).prefetch(
            tf.data.AUTOTUNE
        )

        return train_ds, None
