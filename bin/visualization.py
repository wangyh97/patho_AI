'''
functions for visualization
'''

import numpy as np
if hasattr(os,'add_dll_directory'):
    for i in openslide_path.values():
        if Path(i).exists():
            with os.add_dll_directory(Path(i)):
                import openslide
else:
    import openslide
from openslide.deepzoom import DeepZoomGenerator
import cv2
from matplotlib import pyplot as plt




def null_or_not(x,pr = False):  #只要有一个元素不为0则不为0
    if type(x) is not np.ndarray:
        x = np.array(x)
    if pr:
        if x.any():
            print('not all 0')
        else:
            print('all 0')
    return x.any()

def show_dzg_info(dzg):
    print(f'level count:{dzg.level_count}')
    print(f'tile arrangement of last 3 level{dzg.level_tiles[-4:-1]}\ndimensions of each tile at last 3 level:{dzg.level_dimensions[-4:-1]}')

def pop_tile(dzg,level,save = False):
    row,col = dzg.level_tiles[level]
    saved = []
    if level<0:
        level = dzg.level_count + level
    for i in range(row):
        for j in range(col):
            if null_or_not(dzg.get_tile(level,(i,j))):
                saved.append((i,j))
                if not save:
                    print((i,j))
    if save:
        return saved

def pixel_255(image,point = False,threshold = False):
    if type(image) is not np.ndarray:
        image = np.array(image)
        if point:
            image[image==0] = 255
        if threshold:
            image[image>threshold] = 255 
        return Image.fromarray(image)
    else:
        if point:
            image[image==0] = 255
        if threshold:
            image[image>threshold] = 255 
        return image

#通过cv显示图片
def imgshow(img_path):
    tiff = cv2.imread(img_path)
    plt.imshow(Image.fromarray(tiff))

#查看tiff文件path、info及图片
def tiff_checker(tiff_path):
    show_info(tiff_path = tiff_path)
    tif.imshow(tif.imread(tiff_path))

#同时展示多张图片
def ploting(rows,cols,figseq,
            figsize=(20,20),fontdict={'size':20},title = []):
    #figseq是一个4维的ndarray
    #rows,cols是展示图片的行/列数
    fig,axes = plt.subplots(rows,cols,figsize=figsize)
    fontdict = fontdict
    if rows != 1:
        for i in range(rows):
            for j in range(cols):
                if title:
                    axes[i,j].set_title(title[i*cols+j])
                else:
                    axes[i,j].set_title(f'{i}_{j}')
                axes[i,j].imshow(figseq[i*cols+j])
    else:
        for j in range(cols):
            if title:
                axes[j].set_title(title[j])
            else:
                axes[j].set_title(f'{j}')
            axes[j].imshow(figseq[j])