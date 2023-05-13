import torch
from torch import nn
from torchvision import models as torch_models
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from skimage import io
from tqdm import tqdm
import pandas as pd
import argparse
from model.resnet_simclr import ResNetSimCLR


class ResNet_extractor(nn.Module):
    def __init__(self, layers=101):
        super().__init__()
        if layers == 18:
            self.resnet = torch_models.resnet18(pretrained=True)
        elif layers == 34:
            self.resnet = torch_models.resnet34(pretrained=True)
        elif layers == 50:
            self.resnet = torch_models.resnet50(pretrained=True)
        elif layers == 101:
            self.resnet = torch_models.resnet101(pretrained=True)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
def main():
    #args
    parser = argparse.ArgumentParser(description='feature extraction using pretrained Resnet18')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device,[0]')
    parser.add_argument('--scale',type=int,help='scale of slides to be extracted')
    parser.add_argument('--layers',type=int,default=18,help='layers of resnet,choose from[18,34,50,101],[18]')
    args = parser.parse_args()
    
    #load data
    data_df = np.load(f'../config/data_segmentation_csv/{args.scale}X_full.npy',allow_pickle=True).item()['full_list']
    
    #basic configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    save_dir = f'../data/pretrained_resnet{args.layers}'
    os.makedirs(save_dir, exist_ok=True)
    
    data_transform = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    batch_size = 4096
    
    N = data_df.shape[0]
    
    #load model
    model = ResNet_extractor(layers=args.layers)
    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=eval(args.gpu))
    model.cuda()
    
    count = 0
    slide_features = {}
    print('start to process')
    
    unextracted = []
    
    with torch.no_grad():
        model = model.eval()
        for i in tqdm(range(N)):
            try:
                paths = data_df['img_list'].iloc[i]
                N_tumor_patch = len(paths)
                feature_list = []
                for batch_idx in range(0, N_tumor_patch, batch_size):
                    end = batch_idx + batch_size if batch_idx+batch_size < N_tumor_patch else N_tumor_patch
                    batch_paths = paths[batch_idx: end]
                    images = []
                    for p in batch_paths:
                        image = Image.open(p).convert('RGB')
                        image_tensor = data_transform(image).unsqueeze(0)
                        images.append(image_tensor)
                    images = torch.cat(images, dim=0) #concat all images into one, every slide generates a image tensor in shape [num_of_extracted_patches,224*224]

                    features = model(images.cuda()) #[num_of_extracted_patches,outdim]
                    if len(features.shape) == 1:
                        features = features.unsqueeze(0)
                    feature_list.append(features.detach().cpu())
                    del features

                feature_list = torch.cat(feature_list, dim=0)
                slide_features[f'index{i}'] = (data_df['dir_uuid'].iloc[i],data_df['TMB_H/L'].iloc[i],feature_list)
            except Exception as e:
                print(f'wrong in slide{i}, error as {e}')
                unextracted.append(i)

        np.save(os.path.join(save_dir,f'{args.scale}X_full_slide_features_PtRes18.npy'),slide_features)
        np.save(os.path.join(save_dir,f'{args.scale}X_full_not_extracted_PtRes18.npy'),unextracted)
        print(f'{len(unextracted)} unextracted,\nindexes are {unextracted},saved in file',end = '\n')
        print('file saved')
        
if __name__ == '__main__':
    main()