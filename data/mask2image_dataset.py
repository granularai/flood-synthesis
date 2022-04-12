import os
import glob
import numpy as np
import random
import cv2
import torch
import tqdm
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class Mask2ImageDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        data_path = '/home/shubham/granular/Dataset/xBD/challange/'
        data = dict(train = data_path+'/train',
            hold = data_path+'/hold',
            tier3 = data_path+'/tier3',
            test = data_path+'/test'
        )

        modes = ['train']

        self.imgs = []

        for mode in modes:
            data_path = data[mode]
            x_dir = data_path+'/patch/*'
            y_dir = data_path+'/target_mask_patch/'
            
            imgs = glob.glob(x_dir)
            for img in tqdm.tqdm(imgs):
                name = img.split('/')[-1].split('.')[0]
                im = np.unique(cv2.imread(y_dir+name+'.png',0))
                if len(im) > 1:
                    self.imgs.append((img,y_dir+name+'.npz',y_dir+name+'.png'))
                del im
        print("dataset process finished")

        
        self.input_nc = 6
        opt.input_nc = 6
        self.output_nc = 3
        opt.output_nc = 3




        #self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        #self.dirA = os.path.join(opt.dataroot,)
        #self.dirB = os.path.join(opt.dataroot,)
        #self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        #self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        #self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        x,y,y_ = self.imgs[index]
        img_x = Image.open(x).convert('RGB')
        img_y_ = Image.open(y_).convert('RGB')
        img_y = np.load(y)['arr_0']
        # split AB image into A and B
        

        # apply the same transform to both A and B
        #transform_params = get_params(self.opt, x.size)
        #A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        #B_transform = get_transform(self.opt, transform_params, grayscale=self.input_nc)

        flip = random.random() > 0.5
        
        transform_img = transforms.Compose([
            transforms.RandomHorizontalFlip(p=float(flip)),
            transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        img_x = transform_img(img_x)
        img_y_ = transform_img(img_y_)

        if flip:
            img_y = img_y[:,::-1,:]
        img_y = cv2.resize(img_y.astype(np.uint8),(512,512),interpolation=cv2.INTER_NEAREST)
        img_y = torch.tensor(img_y).type(torch.float).permute(2,0,1)



        return {'A': img_x, 'B': img_y, 'A_paths': '', 'B_paths': ''}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.imgs)
