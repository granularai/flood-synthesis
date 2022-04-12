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

from data.image_folder import make_dataset
from PIL import Image

class Mask2ImageFloodnetDataset(BaseDataset):
    def __init__(self, opt):
        super(Mask2ImageFloodnetDataset, self).__init__(opt)
        data_path = '/mnt/now/HADR/data/train'

        self.imgs = []

        x_dir = data_path+'/x/*'
        y_dir = data_path+'/y/'
        
        imgs = glob.glob(x_dir)
        print(data_path)
        for img in tqdm.tqdm(imgs):
            name = img.split('/')[-1].split('.')[0]
            self.imgs.append((img,y_dir+name+'.bmp'))
        print("dataset process finished")
        print(len(self.imgs))
        
        self.input_nc = 25
        opt.input_nc = 25
        self.output_nc = 3
        opt.output_nc = 3
        self.num_classes=23

        self.height, self.width = 512, 512
    
    def __len__(self):
        return len(self.imgs)
    
    def __getG__(self, idx):
        yield self.__getitem__(idx)
    
    def __getGen__(self, idx):
        return tf.data.Dataset.from_generator(
            self.__getG__, 
            output_signature=(
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
                ),
            args=(idx,)
            )

    def __getitem__(self, idx):
        x,y = self.imgs[idx]
        img_x = Image.open(x).convert('RGB')
        img_y = Image.open(y)

        img_x = img_x.resize((self.width,self.height), resample = Image.BILINEAR)
        img_y = img_y.resize((self.height,self.width), resample = Image.NEAREST)
        # split AB image into A and B
        img_x = tf.keras.utils.img_to_array(img_x)
        img_y = tf.keras.utils.img_to_array(img_y)
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
        return tf.one_hot(
            tf.cast(image_x, dtype=tf.int32), 
            depth=self.num_classes,
            axis = -1, 
            dtype=tf.float32
        )[:,:,0,:], image_y
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

    @classmethod
    def getTfDataset(cls, opt):
        ds = cls(opt)
        #resizing = partial(layers.Resizing, height=512, width=512)
        #resize_x = resizing(interpolation='nearest')
        #resize_y = resizing(interpolation='bilinear')
        normalizing = layers.Normalization(mean=(0.0,0.0,0.0), variance=(255.0,255.0,255.0))

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
                ds.preprocessMask(x,y),
                y
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(
            opt.batch_size,
            drop_remainder=True
        ).prefetch(
            tf.data.AUTOTUNE
        )

        return train_ds


class Mask2ImageFloodnetDataset_backup_torch_varsion(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        print("coming here")
        BaseDataset.__init__(self, opt)

        data_path = '/mnt/now/HADR/data/train'

        self.imgs = []

        x_dir = data_path+'/x/*'
        y_dir = data_path+'/y/'
        
        imgs = glob.glob(x_dir)
        print(data_path)
        for img in tqdm.tqdm(imgs):
            name = img.split('/')[-1].split('.')[0]
            self.imgs.append((img,y_dir+name+'.bmp'))
        print("dataset process finished")
        print(len(self.imgs))

        
        self.input_nc = 25
        opt.input_nc = 25
        self.output_nc = 3
        opt.output_nc = 3




        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #self.dirA = os.path.join(opt.dataroot,)
        #self.dirB = os.path.join(opt.dataroot,)
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        #self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        #self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def process(self, mask):
        mask = torch.nn.functional.one_hot(torch.tensor(mask).long(), num_classes=23)
        return mask.numpy()


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        x,y = self.imgs[index]
        img_x = Image.open(x).convert('RGB')
        img_y = Image.open(y)
        # split AB image into A and B
        

        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, x.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #B_transform = get_transform(self.opt, transform_params, grayscale=self.input_nc)


        flip = random.random() > 0.5
        
        transform_img_a = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=float(flip)),
            transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BICUBIC),
            ])
        transform_img_b = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=float(flip)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        transform_img_ = transforms.Compose([
            #transforms.RandomHorizontalFlip(p=float(flip)),
            transforms.Resize((512,512), interpolation=transforms.InterpolationMode.NEAREST),
            
            ])

        img_x = transform_img_a(img_x)
        img_x_ = transform_img_b(img_x)
        img_y = transform_img_(img_y)

        img_y = np.array(img_y)

        #test
        mask = self.process(img_y)
        house = img_x*mask[:,:,1,None] #)[:,:,[2,1,0]]
        img_y = np.concatenate([mask[:,:,0, None], house, mask[:,:,2:]], axis=-1)
        #test

        # if not test uncomment below
        #img_y = img_y[:,:,None] == [i for i in range(25)]

        img_y = img_y.astype(np.float32)

        img_y = torch.tensor(img_y).permute(2,0,1)
        
        return {'A': img_y, 'B': img_x_, 'A_paths': '', 'B_paths': ''}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.imgs)
