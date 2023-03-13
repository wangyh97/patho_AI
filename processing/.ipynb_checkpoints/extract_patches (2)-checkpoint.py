#!/usr/bin/env python
# coding: utf-8

# In[4]:

#!/usr/bin/env python
# coding: utf-8

# In[4]:
from numba import jit,njit

import openslide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import cv2

from lxml import etree

from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import gc
import time
import sys
import getopt

def get_slide(slide_path):
    slide = openslide.OpenSlide(slide_path)
    return slide

#@jit(nopython=True)
def AnnotationParser(path):
    assert Path(path).exists(), "This annotation file does not exist."
    tree = etree.parse(path)
    annotations = tree.xpath("/ASAP_Annotations/Annotations/Annotation")
    annotation_groups = tree.xpath("/ASAP_Annotations/AnnotationGroups/Group")
    classes = [group.attrib["Name"] for group in annotation_groups]
   # @jit(nopython=True)
    def read_mask_coord(cls):
        for annotation in annotations:
            if annotation.attrib["PartOfGroup"] == cls:
                contour = []
                for coord in annotation.xpath("Coordinates/Coordinate"):
                    x = np.float(coord.attrib["X"])
                    y = np.float(coord.attrib["Y"])
                    contour.append([round(float(x)),round(float(y))])
                #mask_coords[cls].extend(contour)
                mask_coords[cls].append(contour)
    #@jit(nopython=True)
    def read_mask_coords(classes):
        for cls in classes:
            read_mask_coord(cls)
        return mask_coords            
    mask_coords = {}
    for cls in classes:
        mask_coords[cls] = []
    mask_coords = read_mask_coords(classes)
    return mask_coords,classes


def Annotation(slide,path,save_path=None,rule=False,save=False):
    #wsi_height = slide.wsi_height
    #wsi_width = slide.wsi_width
    wsi_width,wsi_height = slide.level_dimensions[0]
    masks = {}
    contours = {}
    mask_coords, classes = AnnotationParser(path)
    
    def base_mask(cls,wsi_height,wsi_width):
        masks[cls] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    def base_masks(wsi_height,wsi_width):
        for cls in classes:
            base_mask(cls,wsi_height,wsi_width)
        return masks
    
    def main_masks(classes,mask_coords,masks):
        for cls in classes:
            contours = np.array(mask_coords[cls])
            #contours = mask_coords[cls]
            for contour in contours:
                #print(f"cls:{cls},\ncontour:{contour},\ntype:{type(contour)}")
                masks[cls] = cv2.drawContours(masks[cls],[np.int32(contour)],0,True,thickness=cv2.FILLED)
        return masks
   # def make_masks()
    def export_mask(save_path,cls):
        assert Path(save_path).is_dir()
        cv2.imwrite(str(Path(save_path)/"{}.tiff".format(cls)),masks[cls],(cv2.IMWRITE_PXM_BINARY,1))
    def export_masks(save_path):
        for cls in masks.keys():
            export_mask(save_path,cls)
    def exclude_masks(masks,rule,classes):
        #masks_exclude = masks.copy()
        masks_exclude = masks
        for cls in classes:
            for exclude in rule[cls]["excludes"]:
                if exclude in masks:
                    overlap_area = cv2.bitwise_and(masks[cls],masks[exclude])
                    masks_exclude[cls] = cv2.bitwise_xor(masks[cls],overlap_area)
        #masks = masks_exclude
        return masks_exclude
                    
    masks = base_masks(wsi_height,wsi_width)
    masks = main_masks(classes,mask_coords,masks)
    if rule:
        classes = list(set(classes) & set(rule.keys()))
        masks = exclude_masks(masks,rule,classes)
        #include_masks(rule)
        #exclude_masks(rule)
    if save:
        export_masks(save_path)
    if "artifact" not in classes:
        masks["artifact"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    if "mark" not in classes:
        masks["mark"] = np.zeros((wsi_height,wsi_width),dtype=np.uint8)
    return masks 

def show_thumb_mask(mask,size=512):
    #mask = masks[cls]
    height, width = mask.shape
    scale = max(size / height, size / width)
    mask_resized = cv2.resize(mask, dsize=None, fx=scale, fy=scale)
    mask_scaled = mask_resized * 255
    plt.imshow(mask_scaled)
    return mask_scaled

def get_mask_slide(masks):
    tumor_slide = openslide.ImageSlide(Image.fromarray(masks["stroma"]))
    #mark_slide = openslide.ImageSlide(Image.fromarray(masks["mark"])) ## get tile_masked dont need mark and arti mask
    #arti_slide = openslide.ImageSlide(Image.fromarray(masks["artifact"]))
    return tumor_slide

def get_tiles(slide,tumor_slide,tile_size=512,overlap=False,limit_bounds=False):
    slide_tiles = DeepZoomGenerator(slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    tumor_tiles = DeepZoomGenerator(tumor_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    #mark_tiles = DeepZoomGenerator(mark_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    #arti_tiles = DeepZoomGenerator(arti_slide,tile_size,overlap=overlap,limit_bounds=limit_bounds)
    return slide_tiles,tumor_tiles
#@njit
def remove_arti_and_mask(slide_tile,tumor_tile,mark_tile,arti_tile):
    mark_tile = np.where(mark_tile==0,1,0)
    arti_tile = np.where(arti_tile==0,1,0)
    #assert slide_tile.shape
    #slide_tile = np.array(slide_tile)
    #tumor_tile = np.array(tumor_tile)
    x = slide_tile.shape
    if not x == tumor_tile.shape:
        tumor_tile = tumor_tile[:x[0],:x[1],:]
    if not mark_tile.shape == x:
        mark_tile = mark_tile[:x[0],:x[1],:]
    if not arti_tile.shape == x:
        arti_tile = arti_tile[:x[0],:x[1],:]
    tile = np.multiply(np.multiply(slide_tile,mark_tile),arti_tile)
    #if tile[np.where(tile==np.array([0,0,0]))].shape!=(0,):
        #tile[np.where(tile==np.array([0,0,0]))]= fill
    #tile[np.where(tile==np.array([0,0,0]))] = fill # fill blank may cause color torsion
    tile_masked= np.multiply(tile,tumor_tile)
    #tile = Image.fromarray(np.uint8(tile))
    #assert tile.size==(512,512),f"wrong shape:{tile.size}"
    return tile,tile_masked

def get_tile_masked(slide_tile,tumor_tile): ####version_update: To save tile_masked, use this function
    x = slide_tile.shape
    if not x == tumor_tile.shape:
        tumor_tile = tumor_tile[:x[0],:x[1],:]
    tile_masked = np.multiply(slide_tile,tumor_tile)
    return tile_masked

def filtered(tile):
    tolerance = np.array([230,230,230])
    #tile_1 = tile.copy()
    tile[np.all(tile>tolerance,axis=2)]=0
    percent = ((tile != np.array([0,0,0])).astype(float).sum(axis=2)!=0).sum()/(tile.shape[0]**2)
    return percent
def filtered_cv(img):
    #tolerance = np.array([230,230,230])
    #tile_1 = tile.copy()
    tile = np.copy(img).astype(np.uint8)
    gray = cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,_ = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    tile[np.all(tile>ret,axis=2)] = 0
    percent = ((tile != np.array([0,0,0])).astype(float).sum(axis=2)!=0).sum()/(tile.shape[0]**2)
    return percent
def filtered_same(img):
    percent = ((img[:,:,0]==img[:,:,1]).astype(float) *(img[:,:,0]==img[:,:,2]).astype(float)).sum()/(img.shape[0]**2)
    return percent
def filter_blood(img):
    ## lower mask(0-10)
    img_hsv = cv.cvtColor(img,cv.COLOR_RGB2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1
    percent = ((mask != 0)).sum()/mask.shape[0]**2
    return percent
#@jit(nopython=True)
def extract_patches(levels,scales):
    
    for i,level in enumerate(levels):
        
        print(f'processing ---level {scales[i]}')
        print(tile_path)
        tiledir = Path(tile_path)/str(scales[i])
        #print(f"tile_dir creating--{tiledir}")
        
        if not Path(tiledir).exists():
            os.makedirs(tiledir)
           # print("tile_dir created")
        assert slide_tiles.level_tiles[level] == mark_tiles.level_tiles[level]
        cols,rows = slide_tiles.level_tiles[level]
        for row in range(rows):
            for col in range(cols):
                tilename = os.path.join(tiledir,'%d_%d.%s'%(col,row,"tiff"))
               # print("tile_name creating")
                if not Path(tilename).exists():
                    slide_tile = np.array(slide_tiles.get_tile(level,(col,row)))
                    tumor_tile = np.array(tumor_tiles.get_tile(level,(col,row)))
                    #mark_tile = np.array(mark_tiles.get_tile(level,(col,row)))
                    #arti_tile = np.array(arti_tiles.get_tile(level,(col,row)))
                    #print("tiles are processing")
                    #tile,tile_masked = remove_arti_and_mask(slide_tile,tumor_tile,mark_tile,arti_tile)
                    tile_masked = get_tile_masked(slide_tile,tumor_tile)
                   # tile_masked = np.multiply(slide_tile,mark_tile)
                    percent_1 = filtered_cv(tile_masked)
                    percent_2 = filtered_same(tile_masked)
                    percent_3 = filter_blood(tile_masked)
                    if all((percent_1 >= 0.4,percent_2 <= 0.15,percent_3 <= 0.15)):
                       # Image.fromarray(np.uint8(tile)).save(tilename)
                        Image.fromarray(np.uint8(tile_masked)).save(tilename)
                        #print("saving tile")
                    else:
                        pass
        print("Done!")
    print("All levels processed!!")
    
os.chdir("/GPUFS/sysu_jhluo_1")



INDEX= 0

n = 5

# TCGA 159
# SYSUCC 337
# TCGA 470

argv = sys.argv[1:]
try:
    opts,args = getopt.getopt(argv,"n:i:x:")
except:
    print("Error")
for opt,arg in opts:
    if opt in ['-n']:
        n = int(arg)
    elif opt in ['-i']:
        i = int(arg)
    elif opt in ['-x']:
        INDEX = int(arg)


TILE_SIZE = 512

sysu_path = "Pathology/SYSUCC_cases/SYSU-CancerCenter"
zsyy_path = "Pathology/ZSYY"
pufh_path = "Pathology/PUFH"
tcga_path = "Pathology/TCGA_cases"
all_paths = [zsyy_path,pufh_path,sysu_path,tcga_path]
all_patch_path = [f"Pathology/ZSYY_cases_Patch_{TILE_SIZE}",f"Pathology/PUFH_cases_Patch_{TILE_SIZE}",f"Pathology/SYSUCC_cases_Patch_{TILE_SIZE}",f"Pathology/TCGA_cases_Patch_{TILE_SIZE}"]


OVERLAP =0
LIMIT = False
rule = {"stroma":{"excludes":["blood","artifact","mark"]}}
scales = ['5X','10X','20X','40X']
#slide_source = "Pathology/ZSYYCASES/zsyy-cases-new11-5"
#patch_path = f"Pathology/ZSYY_cases_Patch_{TILE_SIZE}"
slide_source = all_paths[INDEX]
patch_path = all_patch_path[INDEX]
svs_paths = list(Path(slide_source).rglob("*.svs"))+list(Path(slide_source).rglob("*.tif"))
#slide_paths = [Path(slide).name for slide in glob.glob(f"{patch_path}/*/*/*") if not len(os.listdir(slide))==4] #

# In[7]:


len(svs_paths)

# # i=??

# In[ ]:




# In[5]:



number = len(svs_paths)

if n*i < number:
    svs_paths = svs_paths[n*(i-1):n*i]
if n*i >= number:
    svs_paths = svs_paths[n*(i-1):]


# In[7]:


extracted_case = []
un_extracted_case = []
for i,svs in enumerate(svs_paths):
    start = time.time()
    totol_num = len(svs_paths)
    print(f"processing  {i+1}/{totol_num}:------{svs}")
    xml_path = str(Path(svs).with_suffix(".xml"))
    case_name = Path(svs).parent.name
    case_path = Path(patch_path)/Path(svs).parent.parent.name/case_name
    tile_path = Path(case_path)/Path(svs).stem.split(".")[0]
    slide = get_slide(str(svs))
    try:
        masks = Annotation(slide,path=str(xml_path))
        print(f"masks groups includes :{list(masks.keys())}")
        tumor_slide = get_mask_slide(masks)
        slide_tiles,tumor_tiles = get_tiles(slide,tumor_slide,tile_size=TILE_SIZE,overlap=OVERLAP,limit_bounds=LIMIT)
        del slide
        del masks
        del tumor_slide
        gc.collect()
        level_count = slide_tiles.level_count
        #fill = int(np.array(slide_tiles.get_tile(level_count-1,(0,0))).mean())
        levels=[level_count-4,level_count-3,level_count-2,level_count-1]
        #print(f"fill_blank_value:{fill}")

        try:
            extract_patches(levels,scales)
            extracted_case.append(svs)
        except Exception as e:
            un_extracted_case.append(svs)
            print("something is wrong when extracting")
            print("ERROR!",e)
            continue
    except Exception as e:
        print("something is wrong when parsing")
        print("ERROR!",e)
        continue
    end = time.time()
    print(f"Time consumed : {(end-start)/60} min")
    print(f"******{len(un_extracted_case)}/{len(svs_paths)} remain unextract******")


