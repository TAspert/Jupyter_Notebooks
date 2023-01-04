# Import required libraries
import glob
import os
from pathlib import Path
import tkinter
import math

from dataclasses import dataclass

from tqdm.notebook import trange, tqdm

import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
#import ptitprince as pt # if not installed, run: "conda install -c conda-forge ptitprince"

import numpy as np
from tkinter.filedialog import askdirectory, askopenfilename
import pandas as pd
from skimage import io, measure, filters, morphology, exposure,color, draw, segmentation
import cv2 as cv
from PIL import Image #openCV sucks at importing tiff, we use pillow
from cellpose import models, utils, io as iocp

# %%
p='C:/Users/Silence/Desktop/ALD/Basile/wt_g6pd_calcein-pi_221222-20230103T185802Z-001/wt_g6pd_calcein-pi_221222'


image_paths=glob.glob(p+'/*.tiff')
gpuvalvalue='True'
diamvalue=30

#%% First, deal with tritc channel
cc=0
alive_dead=[]
for image_path in image_paths:
# image_paths_tritc=list();
# for s in image_paths:
#     if 'tritc' in s:
#         image_paths_tritc.append(s)
#image_paths_tritc=[s for s in image_paths if 'tritc' in s]
    if 'gfp' in image_path:
        #% Deal with gfp channel
        im_name=os.path.basename(image_path).split('gfp')[0]
        
        # Load the cellpose model
        #model = models.CellposeModel(gpu=gpuvalvalue, pretrained_model='cyto2')
        model = models.Cellpose(gpu=gpuvalvalue, model_type='cyto2')
        channels=[0,0]
        
        #image_paths_gfp=[s for s in image_paths if 'gfp' in s]
        # Segment the image using cellpose
        img = Image.open(image_path) #scikit gives some weird warning with tiff
        img=np.array(img,dtype=np.uint16)
        #img=io.imread(image_path) 
        img=exposure.rescale_intensity(img, out_range='uint8')
        mask, _,_,_ = model.eval(img,channels=channels,diameter=diamvalue)

        boundaries=segmentation.find_boundaries(mask)
        
        #img_norm=exposure.rescale_intensity(img)
        overlay=color.gray2rgb(img)
        overlay[boundaries,0]=0
        overlay[boundaries,1]=np.max(img)
        overlay[boundaries,2]=0
        
        io.imsave(os.path.join(p, im_name+'_gfp_masked.png'), overlay)
        
        # iocp.save_masks(img, 
        #           mask, 
        #           flow, 
        #           image_path, 
        #           channels=channels,
        #           png=False, # save masks as PNGs and save example image
        #           tif=False, # save masks as TIFFs
        #           save_txt=False, # save txt outlines for ImageJ
        #           save_flows=False, # save flows as TIFFs
        #           save_outlines=False, # save outlines as TIFFs. Useful to use with ImageJ
        #           )
    
        # mask2=utils.remove_edge_masks(mask)
        # outline=utils.masks_to_outlines(mask2)
        #perimeters=utils.get_mask_perimeters(mask2)
        regions = measure.regionprops(mask)
        num_alive=len(regions)
        
    elif 'tritc' in image_path:
        im_name=os.path.basename(image_path).split('tritc')[0]
        img = Image.open(image_path)
        img=np.array(img,dtype=np.uint16)
        #img=io.imread(image_path)
        img=exposure.rescale_intensity(img, out_range='uint8')
        # npix=img.size
        # kmin=npix-round(satur[1]*npix)
        # idx = np.argpartition(np.ravel(img), kmin)
        # minv=max(np.ravel(img)[idx[:kmin]])
                 
        # kmax=round(satur[0]*npix)
        # idx = np.argpartition(np.ravel(img), -kmax)
        # maxv=min(np.ravel(img)[idx[-kmax:]])
        minv,maxv=np.uint8(np.percentile(img, (0.01, 99.99)))
                 
                         
        # Otsu's thresholding after Gaussian filtering
        # blur = cv.GaussianBlur(img,(5,5),0)
        # ret,th = cv.threshold(blur,50000,65535,cv.THRESH_BINARY+cv.THRESH_OTSU)
        thresh = filters.threshold_otsu(img)
        binary = img > thresh*2
        binary=morphology.remove_small_objects(binary, min_size=16)
        binary_labeled=measure.label(binary)
        regions = measure.regionprops(binary_labeled)
        
        num_dead=len(regions)
        # plt.subplot(1,2,1)
        # plt.imshow(img,'gray',norm='linear',extent=[0, 500, 0, 500],vmin=minv, vmax=maxv)
        # plt.subplot(1,2,2)
        # plt.imshow(binary,'gray',norm='linear',extent=[0, 500, 0, 500],vmin=0, vmax=1)
        # plt.show()
        
        boundaries=segmentation.find_boundaries(binary)
        img_norm=exposure.rescale_intensity(img, in_range=(minv, maxv))
        overlay=color.gray2rgb(img_norm)
        overlay[boundaries,0]=np.max(img_norm)
        overlay[boundaries,1]=0
        overlay[boundaries,2]=0
        #plt.imshow(img_norm,'gray')
        # plt.imshow((overlay/65535*255).astype('uint8'))
        # plt.show()
        io.imsave(os.path.join(p, im_name+'_tritc_masked.png'), overlay)
        
        alive_dead+=[[num_alive,num_dead,im_name]]
        df = pd.DataFrame([alive_dead[cc]], columns=['Number_alive', 'Number_dead', 'Source_Images'])
        df.to_excel(os.path.join(p, im_name + '.xlsx'))
        cc=cc+1
        
df_all = pd.DataFrame(alive_dead, columns=['Number_alive', 'Number_dead', 'Source_Images'])
df_all.to_excel(os.path.join(p, 'data_all.xlsx'))