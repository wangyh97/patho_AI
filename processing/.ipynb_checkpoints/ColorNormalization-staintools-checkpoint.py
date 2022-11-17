#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path

import tensorflow as tf
from pathlib import Path
from PIL import Image
import os
import glob
import random
import sys
import getopt
from sklearn.model_selection import train_test_split,ShuffleSplit,StratifiedShuffleSplit

import cupy as cp
import tifffile as tif
import cv2
import gc
from tqdm import tqdm
import staintools 

#!/usr/bin/env python
# coding: utf-8
os.chdir("/GPUFS/sysu_jhluo_1/")
argv = sys.argv[1:]
opts,args = getopt.getopt(argv,'c:l:i:n:')
for opt,arg in opts:
    if opt in ['-c']:
        center = arg # ["SYSUCC","SYSUFAH"]
    elif opt in ['-l']:
        LEVEL= arg # [20X/40X]
    elif opt in ["-i"]:
        INDEX=int(arg) # for indexing
    elif opt in ["-n"]:
        n = int(arg)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{INDEX-1}"


class HENormalizer:
    def fit(self, target):
        pass

    def normalize(self, I, **kwargs):
        raise Exception('Abstract method')

"""
Inspired by torchstain :
Source code adapted from: https://github.com/schaugf/HEnorm_python;
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class Normalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = cp.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        self.maxCRef = cp.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -cp.log((I.astype(cp.float32)+1)/Io)

        # remove transparent pixels
        ODhat = OD[~cp.any(OD < beta, axis=1)]

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        #project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:,1:3])

        phi = cp.arctan2(That[:,1],That[:,0])

        minPhi = cp.percentile(phi, alpha)
        maxPhi = cp.percentile(phi, 100-alpha)

        vMin = eigvecs[:,1:3].dot(cp.array([(cp.cos(minPhi), cp.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(cp.array([(cp.cos(maxPhi), cp.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        if vMin[0] > vMax[0]:
            HE = cp.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = cp.array((vMax[:,0], vMin[:,0])).T

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = cp.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = cp.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1,3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = cp.linalg.eigh(cp.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = cp.array([cp.percentile(C[0,:], 99), cp.percentile(C[1,:],99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        I = cp.asarray(I)
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    def normalize(self, I, Io=240, alpha=1, beta=0.15):
        ''' Normalize staining appearence of H&E stained images
        Example use:
            see test.py
        Input:
            I: RGB input image
            Io: (optional) transmitted light intensity
        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image
        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
       # I = cp.asarray(I)
        batch,h, w, c = I.shape
        I = I.reshape((-1,3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = cp.divide(maxC, self.maxCRef)
        C2 = cp.divide(C, maxC[:, cp.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = cp.multiply(Io, cp.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 255
        Inorm = cp.reshape(Inorm.T, (batch,h, w, c)).astype(cp.uint8)



        return Inorm



        


classes_to_index = {'nonprogress':0,'progress':1}

cases = glob.glob(f"grey/{center}_tiles/*")

cn_path = f"{center}_tiles_CN"

cases_select = cases[n*(INDEX-1):n*INDEX]

#mempool = cp.get_default_memory_pool()
#pinned_mempool = cp.get_default_pinned_memory_pool() CP_template_new/*/*.tiff
#templates = tif.imread(glob.glob("grey/CN_template/*.tiff")[8])
template_40 = tif.imread(glob.glob("grey/CP_template_new/40X/*.tiff")[0])
template_20 = tif.imread(glob.glob("grey/CP_template_new/20X/*.tiff")[0])
normalizer_20 = staintools.StainNormalizer(method="macenko")
normalizer_40 = staintools.StainNormalizer(method="macenko")
normalizer_20.fit(staintools.LuminosityStandardizer.standardize(template_20))
normalizer_40.fit(staintools.LuminosityStandardizer.standardize(template_40))
unnorm_tiles = []
print(len(cases_select))
for case in tqdm(cases_select):
    tiles = list(Path(case).rglob("*.tiff"))
    print(len(tiles))
    try:
        for tile in tiles:
            try:
                tile_name = str(tile).replace(f"{center}_tiles",cn_path)
                if not Path(tile_name).exists():
                    if not Path(tile_name).parent.exists():
                        Path(tile_name).parent.mkdir(parents=True)
                    tile_img = tif.imread(str(tile))
                    #if tile_img.shape[0] != 512 or tile_img.shape[1] != 512:
                        #tile_img = Image.fromarray(np.uint8(tile_img)).resize((512,512),Image.ANTIALIAS)
                        #tile_img = np.asarray(tile_img)
                    #tile_name = str(tile).replace
                   # tile_img = tif.imread(str(tile))
                    #print(tile_img)

                    #tile_img = tile_img.reshape(1,512,512,3)
                    #imgs = cp.asarray(tile_img,dtype=cp.float64)
                    #norm_imgs= cp.asnumpy(normalizer.normalize(I=imgs))
                   # norm_imgs = norm_imgs.reshape(512,512,3)
                    tile_img = staintools.LuminosityStandardizer.standardize(tile_img)
                    if "20X" in str(tile):
                        norm_img = normalizer_20.transform(tile_img)
                    else:
                        norm_img = normalizer_40.transform(tile_img)
                    tif.imsave(tile_name,norm_img)
                   # Image.fromarray(np.uint8(norm_imgs)).save(tile_name)
            except Exception as e:
                    print(e)
                    print(tile)
                    unnorm_tiles.append(tile)
            #norm_imgs = None
            #imgs = None
            #mempool.free_all_blocks() 
            #pinned_mempool.free_all_blocks()
            gc.collect() 
     
    except Exception as e:
        print(e)
        print(case) 
        continue
np.save(f"grey/unnorm_tiles_{center}_{n}_{INDEX}.npy",np.asarray(unnorm_tiles))