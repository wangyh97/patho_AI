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
import ResNet as ResNet


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
    parser = argparse.ArgumentParser(description='The start and end positions in the file list')

    '''general args to be assigned'''

    # gpu index
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')

    # features to be extracted
    parser.add_argument('--scale',type=int,help='used when arg::extractor == res!  scale of slides to be extracted')

    '''extractor type'''

    # major type
    parser.add_argument('--extractor', type=str,choices=['res','saved','simclr','retccl'],help='saved:pretrained simclr on TCGA/cam,simclr:self trained simclr from scratch')

    # subtype of 'saved' extractor -- pretrained simclr
    parser.add_argument('--load',type=str,choices=['TCGA_high','TCGA_low','c16_high'],help='file name under "simclr_feature_extractor/pretrained_embedder" folder')

    # subtype of 'simclr' extractor -- 1.simclr trained from scratch,  2.resnet18 for pretrained extractor
    parser.add_argument('--model', type=str, default='resnet18', help='simclr based resnet,choose from [resnet18,resnet50]')

    # subtype of 'res' extractor -- pretrained resnet
    parser.add_argument('--layers', type=int, default=18, help='layers of resnet,choose from[18,34,50,101],[18]')

    # default args
    parser.add_argument('--outdim',type=int,default=512,help='dim of extracted features, should be assigned in simclr extractor')

    args = parser.parse_args()

    # load tile paths of slides
    data_df = np.load(f'../config/data_segmentation_csv/{args.scale}X_full.npy',allow_pickle=True).item()['full_list']
    
    save_dir = {
        'res':f'../data/pretrained_resnet{args.layers}',
        'sim':'../data/simclr_extracted_feats',
        'saved':f'../data/{args.load}',
        'retccl':'../data/retccl_res50_2048'}
    os.makedirs(save_dir[args.extractor], exist_ok=True)

    data_transform = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    batch_size = 1024
    
    N = data_df.shape[0]

    #load the model
    if args.extractor == 'res':
        model = ResNet_extractor(layers=args.layers) #with pretrained weights
    elif args.extractor == 'simclr':
        state_dict = torch.load(f'simclr_feature_extractor/runs/{args.file_path}/checkpoints/model.pth')
        model = ResNetSimCLR(base_model=args.model,out_dim=args.outdim)
    elif args.extractor == 'retccl': # get features in shape N*2048
        model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        state_dict = torch.load(r'./best_ckpt.pth')
        model.fc = nn.Identity()

    else:
        model_path = {
            'TCGA_high':'simclr_feature_extractor/pretrained_embedder/TCGA/model-high-v1.pth',
            'TCGA_low':'simclr_feature_extractor/pretrained_embedder/TCGA/model-low-v1.pth',
            'c16_high':'simclr_feature_extractor/pretrained_embedder/c16/20X-model-v2.pth',
        }
        state_dict = torch.load(model_path[args.load])
        model = ResNetSimCLR(base_model=args.model,out_dim=args.outdim)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=eval(args.gpu))
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()} #model trained using nn.dataparallel, needs to rename the keys when apply to a new implemented model without using nn.DataParallel
    
    model.load_state_dict(state_dict)
    model = model.cuda()

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
                    images = torch.cat(images, dim=0)

                    features = model(images.cuda())[1]
                    if len(features.shape) == 1:
                        features = features.unsqueeze(0)
                    feature_list.append(features.detach().cpu())
                    del features

                feature_list = torch.cat(feature_list, dim=0)
                slide_features[f'index{i}'] = (data_df['dir_uuid'].iloc[i],data_df['TMB_H/L'].iloc[i],feature_list)
            except Exception as e:
                print(f'wrong in slide{i}, error as {e}')
                unextracted.append(i)

        np.save(os.path.join(save_dir[args.extractor],f'{args.scale}X_full_slide_features.npy'),slide_features)
        np.save(os.path.join(save_dir[args.extractor],f'{args.scale}X_full_not_extracted.npy'),unextracted)
        print(f'{len(unextracted)} unextracted,\nindexes are {unextracted},saved in file',end = '\n')
        print('file saved')

if __name__ == '__main__':
    main()