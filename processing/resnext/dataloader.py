'''
dataloader 正统在这里
class dataloader -- train from scratch
class featureloader -- train from features extracted

'''

from torchvision import transforms, datasets
import os
import glob
from random import shuffle,seed
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import scipy.io as scio
import numpy as np
import sys

seed(42) # seed42 is all you need

'''
load from original pics
'''
#TODO:review the code
class myDataSet(torch.utils.data.Dataset):
    #通过传入保存有图片path的list构造dataset，通过提取文件名得到label
    def __init__(self,scale,data_df,shuffled,data_transforms = None):
        label_dic = {
            'H':0,
            'L':1
        }
        img_list = sum(data_df['img_list'],[])  #concat wrapped list into one, constructing full list of img_path_list
        label_list = sum([list(data_df['TMB_H/L'].iloc[i])*data_df[f'{scale}x'].iloc[i] for i in range(data_df.shape[0])],[]) # constrcting full list of label, in same sequence as img_list
        
        self.label_dic = label_dic 
        
        #whether shuffle the data
        #TODO:finish shuffle
        if shuffled == 'shuffled':
            print('shuffled data before training')
            z = list(zip(img_list,label_list))
            shuffle(z)
            img_list[:],label_list[:] = zip(*z)
            self.img_list = img_list
            self.label_list = label_list
        elif shuffled == 'unbalanced':
            self.img_list = img_list
            self.label_list = label_list
        
        self.data_transforms = data_transforms
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label_dic = {
            'H':0,
            'L':1
        }
        item = self.img_list[index]
        label = self.label_list[index]
        img = Image.open(item)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_list[index]))
        label = self.label_dic[label]
        return img, label

def myData(args):
#read grouping infos
    grouping = np.load(f'../../config/data_segmentation_csv/{args.scale}X_grouping.npy',allow_pickle=True).item()
    train_df = grouping['train_list']
    val_df = grouping['val_list']
    
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}

    image_datasets['train'] = myDataSet(args.scale,train_df,args.loader,data_transforms=data_transforms['train'])  #传入train list（paths）
    image_datasets['val'] = myDataSet(args.scale,val_df,args.loader,data_transforms=data_transforms['val'])    #传入val list（paths）

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes


    
# featureloader

class FeatureSet(torch.utils.data.Dataset):
    def __init__(self) -> None:
        pass
    def __len__():
        pass
    def __getitem__(self, index):
        pass

