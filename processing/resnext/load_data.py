from torchvision import transforms, datasets
import os
import glob
from random import shuffle
import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import scipy.io as scio
import numpy as np

grouping = np.load('../../config/data_segmentation_csv/5X_grouping.npy',allow_pickle=True).item()
train = grouping['train_list']
val = grouping['val_list']

def myData(args):
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

    image_datasets['train'] = myDataSet(train,data_transforms=data_transforms['train'])  #传入train list（paths）
    image_datasets['val'] = myDataSet(val,data_transforms=data_transforms['val'])    #传入val list（paths）

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

class myDataSet(torch.utils.data.Dataset):
    #通过传入保存有图片path的list构造dataset，通过提取文件名得到label
    def __init__(self, img_list,data_transforms = None):
        label_dic = {
            'H':0,
            'L':1
        }
        self.img_list = img_list
        self.label_dic = label_dic
        self.data_transforms = data_transforms
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label_dic = {
            'H':0,
            'L':1
        }
        item = self.img_list[index]
        img = Image.open(item)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_list[index]))
        label = label_dic[Path(item).parts[4]]
        return img, label