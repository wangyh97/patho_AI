import os
from PIL import Image
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='convert all imgs into arrays, saved in one file')
parser.add_argument('--chunksize', type=int, help='number of slides to be processed in one cpu')
parser.add_argument('-i','--chunkindex', type=int, help='which chunk to be processed')
# parser.add_argument('-n','--num_workers',type=int,help='number of cpus available')
parser.add_argument('--scale', default=10,type=int, choices=[10,20],help='scale of the images,[10]')
parser.add_argument('--data_path', type=str, help='path to the data')'
args = parser.parse_args()

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    data_df = np.load(f'../config/data_segmentation_csv/{args.scale}X_full.npy',allow_pickle=True).item()['full_list']
    save_dir = f'../data/raw_array/{args.scale}X'
    os.makedirs(save_dir,exist_ok=True)

    N = data_df.shape[0]
    r_lim = min(args.chucksize*args.chunkindex,N)
    slides = slides[args.chunksize*(args.chuckindex-1):r_lim]

    for i in tqdm(range(N)):
        try:
            paths = data_df['img_list'].iloc[i]
        except Exception as e:
            print(f'wrong in slide{i}, error as {e}')