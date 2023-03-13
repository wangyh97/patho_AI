import sys
import os
lib_path = ['/home/wangyh/uro_biomarker/python-wsi-preprocessing/deephistopath/wsi','/home/wangyh/uro_biomarker/python-wsi-preprocessing']
for i in lib_path:
    sys.path.append(i)

openslide_path = {'desktop':'D:/edge下载/openslide-win64-20220811/bin',
                'laptop':'E:/openslide-win64-20171122/bin'}
if hasattr(os,'add_dll_directory'):
    for i in openslide_path.values():
        if Path(i).exists():
            with os.add_dll_directory(Path(i)):
                import openslide
else:
    import openslide
    
import importlib
import glob
import pandas as pd
import numpy as np
import tifffile as tif
from func import basic
from func import visualization
from matplotlib import pyplot as plt
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from deephistopath.wsi.util import Time
import math
import PIL
import filter
import slide as sl
import tiles
import util

def vis_thumb_marks(pen_color_cohort:list,figsize = (20,20),vis = False) -> list:
    #pen_color_cohort is a list of tuple, tuples are indices of positions of thumbnails,eg:orange_pens =[(8,0),(39,3),(56,2),(62,1)]
    #tuples in cohort are under condition that when parameter 'rows' in function 'visualization.ploting' equals 5
    #full list of thumbnails is read from folder generated by running function 'multiprocess_training_slides_to_images'
    figseq = []
    titles = []
    for i,j in pen_color_cohort:
        index = i*5+j
        figseq.append(thumbnails_pic[index])
        titles.append(title[index])
    if vis:
        visualization.ploting(len(pen_color_cohort)//5+1,5,figseq = figseq,title = titles,figsize = figsize)
    else:
        pass
    return titles


def remove_mark_tile(pen_marks,scale:['5X','10X','20X','40X'],tissue_type:['T','nonT']):
    CN_patches = []
    excluded = []  #存放过滤掉的tile path
    included = []  #存放经过过滤保留下来的tile path
    count = 0
    for index in pen_marks:
        uuid = df['dir_uuid'][index]
        CN_patches_path = glob.glob(f'/mnt/wangyh/CN_patches/*/{uuid}/{scale}/{tissue_type}*')
        count += len(CN_patches_path)
        CN_patches.append(CN_patches_path)
    for obj in CN_patches:
        tifs = visualization.path_to_img_array(obj)
        for i,tif in enumerate(tifs):
            green = filter.filter_green_channel(tif)
            green_bd = filter.filter_binary_dilation(green,disk_size = 10,output_type='bool')
            green_be = filter.filter_binary_erosion(green,disk_size = 20,output_type='bool')
            green_berh = filter.filter_remove_small_objects(green_be,min_size = 8000,avoid_overmask=False,output_type='bool')
            if green_berh.sum() >= 10000:
                excluded.append(obj[i])
            else:
                included.append(obj[i])
    ratio = len(excluded) / count
    return excluded,included,ratio

df = pd.read_csv('../config/full.csv')

thumbnails = glob.glob('/mnt/wangyh/svs_thumb_img/*')
imgs = glob.glob('/mnt/wangyh/svs_img/*')

title=[] 
for i in thumbnails:
    title.append(eval(Path(i).stem)) 
thumbnails_pic = []
img_pic = []
for i in thumbnails:
    img = plt.imread(i)
    thumbnails_pic.append(img)

blue_pens = [(1,2),(21,1),(22,2),(24,3),(25,4),(33,0),(37,3),(46,3),(46,4),(49,1),(57,3),(67,3)]
green_pens = [(1,1),(5,0),(8,0),(17,1),(18,4),(39,3),(49,2),(56,2),(57,3),(62,1)]
red_pens = [(8,0),(10,1),(13,1),(39,3),(41,1),(46,3),(56,2),(57,1),(62,1),(63,0)]
orange_pens =[(8,0),(39,3),(56,2),(62,1)]
black_pens = [(14,0),(17,1),(41,1),(46,3),(49,4),(52,1),(55,3),(68,0)]


blue_pen_indeces = vis_thumb_marks(blue_pens)
green_pen_indeces = vis_thumb_marks(green_pens)
red_pen_indeces = vis_thumb_marks(red_pens)
orange_pen_indeces = vis_thumb_marks(orange_pens)
black_pen_indeces = vis_thumb_marks(black_pens)


from itertools import product
results = []
for pm,sc,tt in product([blue_pen_indeces,green_pen_indeces,red_pen_indeces,orange_pen_indeces,black_pen_indeces],['5X','10X','20X','40X'],['T','nonT']):
    print(f'{pm},{sc},{tt} start')
    results.append(remove_mark_tile(pm,sc,tt))
    print(f'{pm},{sc},{tt} done')
print('all done')

result = np.asarray(result)
np.save('~/uro_biomarker/patho_AI/processing/rm_mark_result.npy',result)