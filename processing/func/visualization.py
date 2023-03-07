'''
functions for visualization
'''
openslide_path = {'desktop':'D:/edge下载/openslide-win64-20220811/bin',
                'laptop':'E:/openslide-win64-20171122/bin'}
import numpy as np
import os
from pathlib import Path
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
import tifffile as tif
import extract_patches_nonT as ep
import glob
import time
from PIL import Image
from tqdm import tqdm

#---------------functions applied to wsi level----------------------#

#展示deepzoomgenerator对象的实用信息    
def show_dzg_info(dzg):
    print(f'level count:{dzg.level_count}')
    print(f'tile arrangement of last 3 level{dzg.level_tiles[-4:-1]}\ndimensions of each tile at last 3 level:{dzg.level_dimensions[-4:-1]}')

def show_wsi_with_mark(uuid,show = True,get_pic = False,cmap = 'gray'):
    tick = time.time()
    plt.rcParams['image.cmap'] = cmap
    
    svs_path = glob.glob(f'/mnt/wangyh/TCGA_svs/{uuid}/*.svs')[0]
    xml_path = glob.glob(f'/mnt/wangyh/TCGA_svs/{uuid}/*.xml')[0]
    
    rule =  {"tumor":{"excludes":["artificial","stroma","necrosis"]},
        'stroma':{"excludes":['artificial','necrosis']}}
    
    with openslide.OpenSlide(svs_path) as slide:
        thumbnail = slide.get_thumbnail((512,512))
#         mask_coords,classes = ep.AnnotationParser(xml_path)
        annos = ep.Annotation(slide,xml_path,rule=rule)
        tumor_slide,non_tumor_slide = ep.get_mask_slide(annos)
        T_marked_thumb = tumor_slide.get_thumbnail((512,512)) #masked slides are images range in [0,1],use pixel_255 for better visualization
        non_T_marked_thumb = non_tumor_slide.get_thumbnail((512,512))
        if show:
            ploting(1,3,figseq= [thumbnail,pixel_255(T_marked_thumb,threshold = 0.1),pixel_255(non_T_marked_thumb,threshold = 0.1)],title = ['original','T','nT'])
        if get_pic:
            return thumbnail,T_marked_thumb,non_T_marked_thumb
    print(f'{uuid}---->consuming{time.time()-tick}s')
    
def show_wsis_with_marks(uuid_list:list,figsize):
    t = time.time
    #read to get numbers of extracted patches under different scales
    not_well_marked = np.load('not_well_marked_slides.npy',allow_pickle=True).item()
    
    title = []
    figseq = []
    for uuid in tqdm(uuid_list):
        print(f'start drawing {uuid}')
        thumbnail,T_marked_thumb,non_T_marked_thumb = show_wsi_with_mark(uuid,show = False,get_pic=True)
        t = [str(uuid)[5] + f'--{not_well_marked[uuid]}','T','nonT']
        title.extend(t)
        figseq.extend([thumbnail,T_marked_thumb,non_T_marked_thumb])
    num_of_pics = len(uuid_list)
    ploting(rows = num_of_pics,cols = 3,figseq=figseq,figsize = figsize,title = title)
    t_elapsed = time.time()-t
    print(f'consuming {t_elapsed}s in total')
    
#-----------------------functions manipulating single pics---------------------------------------#
        


def null_or_not(x,pr = False):  #只要有一个元素不为0则不为0
    if type(x) is not np.ndarray:
        x = np.array(x)
    if pr:
        if x.any():
            print('not all 0')
        else:
            print('all 0')
    return x.any()


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
    '''
    notice：
        point and threshold cannot be used in the same time
    args:
        point:if True,convert all 0 pixels into 255,i.e. convert all black pixels into white
        threshold: 
            if true, convert all non-0 pixels into 255,i.e. convert all pixels above threshold into 255,making apparent differentiation with black ones
            threshold should be set as a number > 1
    '''
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

#-------------------------------functions displaying pics & infos------------------------------------#
    
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
            figsize=(20,20),fontdict={'size':20},title = [],cmap = False):
    #figseq是一个4维的ndarray
    #rows,cols是展示图片的行/列数
    fig,axes = plt.subplots(rows,cols,figsize=figsize)
    fontdict = fontdict
    if cmap:
        plt.rcParams['image.cmap'] = cmap
    if rows != 1:
        for i in range(rows):
            for j in range(cols):
                if title:
                    axes[i,j].set_title(title[i*cols+j])
                else:
                    axes[i,j].set_title(f'{i}_{j}')
                try:
                    axes[i,j].imshow(figseq[i*cols+j])
                except:
                    print(f'figseq[{i*cols}_{j}] out of range\n')
    else:
        for j in range(cols):
            if title:
                axes[j].set_title(title[j])
            else:
                axes[j].set_title(f'{j}')
            try:
                axes[j].imshow(figseq[j])
            except:
                print(f'figseq[{i*cols}_{j}] out of range\n')  
    plt.show()

# ----------------------functions manipulating imgs in varies of format --------------------#

# convert paths of imgs into ndarray(if paths contains multiple img files, returns a 4darray, else, 3darray)
def path_to_img_array(paths:list) -> list:
    img_list = []
    for i in paths:
        if Path(i).suffix == 'tiff':
            img_list = tif.TiffSequence(paths).asarray()
        else:
            try:
                img = plt.imread(i)
            except:
                print('unsupported image file type')
            img_list.append(img)
    return img_list
        