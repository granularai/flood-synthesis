import os
import glob
from albumentations.augmentations.transforms import Normalize
import pandas
import torch
import cv2
import albumentations as A
import numpy as np
import pandas as pd
import random


from torch.utils import data
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2


class Track1SegDataset(data.Dataset):
    '''dataset for segmentation subtask Track#1 of FloodNet challange'''
    def __init__(self,root_path,imgs,transforms=None,inference=False):
        super(Track1SegDataset,self).__init__()
        self.root_path = root_path #Train/Labeled
        self.imgs = imgs
        self.transforms = transforms
        self.class_names = [
            'background',
            'building-flooded',
            'building-non-flooded',
            'road-flooded',
            'road-non-flooded',
            'water',
            'tree',
            'vehicle',
            'pool',
            'grass'
        ]
        if not inference:
            self.get_item = self.getTrainItem
        else:
            self.get_item = self.getTestItem

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.get_item(index)

    def getTrainItem(self,index):
        #.print('......'+str(index)+'....'+str(len(self.imgs))+'........')
        image = cv2.cvtColor(cv2.imread(self.imgs[index]),cv2.COLOR_BGR2RGB).astype(np.float32)
        _ = self.imgs[index].split('/')
        type_,img = _[-3],_[-1].split('.')[0]
        y_p = self.root_path+type_+'/mask/'+img+'_lab.png'
        mask = cv2.imread(y_p,0)
        if self.transforms:
            augmented = self.transforms(image=image,mask=mask)
            image,mask = augmented['image'],augmented['mask']
        mask = mask[:,:,None].numpy() == [i for i in range(10)]
        mask = torch.tensor(mask,requires_grad=False).permute(2,0,1)
        return image,mask

    def getTestItem(self,index):
        image = cv2.cvtColor(cv2.imread(self.imgs[index]),cv2.COLOR_BGR2RGB).astype(np.float32)
        _ = self.imgs[index].split('/')
        type_,img = _[-3],_[-1].split('.')[0]
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        _ = np.int32(0)
        return image,np.int32(img)

    @classmethod
    def getTrainVal(cls,root_path):
        imgs_flood = glob.glob(root_path+'Flooded/image/*')
        imgs_no_flood = glob.glob(root_path+'Non-Flooded/image/*')
        sz_flood,sz_no_flood = int(len(imgs_flood)*0.6), int(len(imgs_no_flood)*0.6)
        tr_imgs = imgs_flood[:sz_flood]+imgs_no_flood[:sz_no_flood]
        vl_imgs = imgs_flood[sz_flood:]+imgs_no_flood[sz_no_flood:]
        # start train transforms
        tr_transforms = A.Compose([
            A.Blur(p=0.1,blur_limit=(3,5)),
            A.RandomBrightnessContrast(p=0.1,brightness_limit=(-0.19,0.19),contrast_limit=(-0.19,0.19)),
            A.RandomCrop(p=0.1,height=1500,width=2000),
            A.VerticalFlip(p=0.1),
            A.HorizontalFlip(p=0.1),
            A.Rotate(p=0.1,limit=(-180,180),interpolation=2,border_mode=2),
            A.Resize(always_apply=True,p=1.0,height=512,width=512,interpolation=2),
            #A.Normalize(p=1.0,mean=([106.421936,115.71875,88.01196 ]),std=([106.421936,115.71875,88.01196 ])),
            ToTensorV2()
        ])

        vl_transforms = A.Compose([
            A.Resize(always_apply=True,p=1.0,height=512,width=512,interpolation=2),
            #A.Normalize(p=1.0,mean=([106.421936, 115.71875, 88.01196 ]),std=([106.421936, 115.71875, 88.01196 ])),
            ToTensorV2()
        ])
        # end train transforms
        train_dataset = cls(root_path=root_path,imgs=tr_imgs,transforms=tr_transforms)
        val_dataset = cls(root_path=root_path,imgs=vl_imgs, transforms=vl_transforms)
        return train_dataset, val_dataset
    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img[:, :, ::-1]
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl
