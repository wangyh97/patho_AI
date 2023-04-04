from torchvision import transforms, datasets
import os
import glob
from random import shuffle,seed
import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import scipy.io as scio
import numpy as np
import sys
sys.path.append('/home/wangyh/uro_biomarker/imbalanced-dataset-sampler/')
from torchsampler import ImbalancedDatasetSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

seed(10)
#---------------------------------------resegmented,unbalanced-------------------------------------------#
# set args.loader = 'unbalanced' to activate
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
    

#-----------------------------------------resegmented,imbalancedsampler implemented----------------------------------------------#
# set args.loader = 'imb' to activate

def imb_Dataloader(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    grouping = np.load(f'../../config/data_segmentation_csv/{args.scale}X_grouping.npy',allow_pickle=True).item()
    train_df = grouping['train_list']
    val_df = grouping['val_list']
    
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
    #image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])

    image_datasets['train'] = imb_DataSet(args.scale,train_df,data_transforms=data_transforms['train'])
    image_datasets['val'] = imb_DataSet(args.scale,val_df,data_transforms=data_transforms['val'])

    # wrap your data and label into Tensor
    dataloders = {'train': torch.utils.data.DataLoader(image_datasets['train'],
                                                 batch_size=args.batch_size,
                                                 sampler = ImbalancedDatasetSampler(image_datasets['train']),
                                                 num_workers=args.num_workers,
                                                 pin_memory=True),
                 'val':torch.utils.data.DataLoader(image_datasets['val'],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

class imb_DataSet(torch.utils.data.Dataset):
    def __init__(self,scale,data_df,data_transforms = None):
        '''
        data_df:[train_df,val_df,test_df],
            obtained from reading files in './config/data_segmentation_csv/{scale}X.csv'. 
            containing infos listed below:
                dir_uuid	TMB_H/L 	5x(patch_count) 	path	img_list
                    img_list: series of list, each correspond to all preprocessed_patch_file_paths
        '''
        label_dic = {
            'H':0,
            'L':1
        }
        self.img_list = sum(data_df['img_list'],[])  #concat wrapped list into one, constructing full list of img_path_list
        self.label_dic = label_dic
        self.label_list = sum([list(data_df['TMB_H/L'].iloc[i])*data_df[f'{scale}x'].iloc[i] for i in range(data_df.shape[0])],[]) # constrcting full list of label, in same sequence as img_list
        self.data_transforms = data_transforms
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        
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
    
    def get_labels(self):
        return [self.label_dic[i] for i in self.label_list]

#-----------------------------------resegmented,patient-wise balance --------------------------------------------#
# set args.loader as over/under to activate
def patient_Dataloader(args):
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

    image_datasets['train'] = patient_DataSet(train_df,args.scale,args.loader,data_transforms=data_transforms['train'])  #传入train list（paths）
    image_datasets['val'] = patient_DataSet(val_df,args.scale,args.loader,data_transforms=data_transforms['val'])    #传入val list（paths）

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 pin_memory=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes

class patient_DataSet(torch.utils.data.Dataset):
    #通过传入保存有图片path的list构造dataset，通过提取文件名得到label
    def __init__(self,data_df,scale,sampling,data_transforms = None):
        label_dic = {
            'H':0,
            'L':1
        }
        
        img_list= sum(data_df['img_list'],[])  #concat wrapped list into one, constructing full list of img_path_list
        label_list = sum([list(data_df['TMB_H/L'].iloc[i])*data_df[f'{scale}x'].iloc[i] for i in range(data_df.shape[0])],[]) # constrcting full list of label, in same sequence as img_list

        self.label_dic = label_dic
        self.data_transforms = data_transforms
        
        #sampling strategy
        if sampling == 'over':
            sampler = RandomOverSampler(random_state=10)
        if sampling == 'under':
            sampler = RandomUnderSampler(random_state=10)
        
        #resample
        img_list = np.array(img_list).reshape(-1,1)
        self.img_list,self.label_list = sampler.fit_resample(img_list,label_list)
        
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label_dic = {
            'H':0,
            'L':1
        }
        img_list = np.squeeze(self.img_list)
        item = img_list[index]
        label = self.label_list[index]
        img = Image.open(item)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_list[index]))
        label = self.label_dic[label]
        return img, label
