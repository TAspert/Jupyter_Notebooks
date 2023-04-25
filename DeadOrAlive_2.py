# Import required libraries
import glob
import os
from pathlib import Path
import tkinter
import math

from dataclasses import dataclass

import matplotlib, matplotlib.pyplot as plt
import seaborn as sns
#import ptitprince as pt # if not installed, run: "conda install -c conda-forge ptitprince"

import numpy as np
import pandas as pd
from skimage import io, measure, filters, morphology, exposure,color, draw, segmentation
import cv2 as cv
from PIL import Image as PILI #openCV sucks at importing tiff, we use pillow
from cellpose import models, utils, io as iocp

# %%
p=r"C:\Users\Silence\Desktop\ALD\Basile\helene v-2"


image_paths=glob.glob(p+'/*_dapi.tiff')
#image_paths = [image_path for image_path in image_paths if '_masked' not in image_path]
gpuvalvalue='True'
diamvalue=30

#%% First, deal with tritc channel
cc=0
alive_dead=[]
for image_path in image_paths:
        image_path=Path(image_path)
# image_paths_tritc=list();
# for s in image_paths:
#     if 'tritc' in s:
#         image_paths_tritc.append(s)
#image_paths_tritc=[s for s in image_paths if 'tritc' in s]

        #% Deal with gfp channel
        im_name=image_path.stem.split('dapi')[0]
        
        # Load the cellpose model
        #model = models.CellposeModel(gpu=gpuvalvalue, pretrained_model='cyto2')
        model = models.Cellpose(gpu=gpuvalvalue, model_type='cyto2')
        channels=[0,0]
        
        #image_paths_gfp=[s for s in image_paths if 'gfp' in s]
        # Segment the image using cellpose
        img = PILI.open(image_path) #scikit gives some weird warning with tiff
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
        
        io.imsave(os.path.join(p, im_name+'_dapi_masked.png'), overlay)
        

        regions = measure.regionprops(mask)
        num_alive=len(regions)
        
        image_path_tritc=str(image_path).replace('dapi','tritc')
        img = PILI.open(image_path_tritc)
        img=np.array(img,dtype=np.uint16)
        #img=io.imread(image_path)
        img=exposure.rescale_intensity(img, out_range='uint8')
        
        fluo_mean=np.empty(mask.max())
        area=np.empty(mask.max())
        for i in np.arange(1,mask.max()+1):
            maski=mask==i
            maski=maski.astype(int)
            area[i]=np.sum(maski)
            fluo_mean[i]=np.sum(maski*img)/cellsize[i]
        
        data=pd.DataFrame({
            'name': im_name,
            'fluo': fluo,
            'area': area,
            })
        
        data.to_excel(im_name + '_data.xlsx')
        data.to_pickle(im_name + '_data.pkl')
        
        #% Merge all the data from the position
        data_paths =glob.glob(subfolder + "/*data.pkl")
        
        plt.hist(fluo,bins=45)
        plt.show()
        
        # data_pos=pd.DataFrame({
        # 'time': [],
        # 'diameter': [],
        # 'section_area': [],
        # 'circularity': [],
        # 'source_image': [],
        # 'pixel_size': []
        # })
        # # diameters_pos=pd.DataFrame([])
        # for data_path in data_paths:
        # data = pd.read_pickle(data_path)
        # # diameters=np.load(data_path)
        
        # pos_name=os.path.basename(os.path.normpath(subfolder))
        
        # data_pos=pd.concat([data_pos,data],ignore_index=True)
        # # diameters_pos=np.append(diameters_all,diameters)
        
        # main_folder=str(Path(subfolder).parent.absolute())
        # data_pos.to_excel(os.path.join(main_folder, pos_name+'_data_pos.xlsx'))
        # data_pos.to_pickle(os.path.join(main_folder, pos_name+'_data_pos.pkl'))
    
    
            
plt.imshow(img)            
plt.hist(fluo,bins=45)
plt.show()
        