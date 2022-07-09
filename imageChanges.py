from PIL import Image
import PIL

import os
from os import listdir
import numpy as np


# get the path/directory

j = 0
folder_dir = "/home/mohanadhas/Documents/TestSeg/DeepLabv3FineTuning/Pellets/Images/"
for images in os.listdir(folder_dir):
    path = folder_dir +images

    '''
    a = images.split('.png')
    print(a)
    if(len(a) != 2):
        out = a[0]+a[1]+'.png'
        os.rename(path,folder_dir+out)
        print(out)
    print()
    '''
    #b = a[1].split('.png_')[-1]
    #if(b.split('.')[0]!=''):
        #out = a[0]+'_'+b.split('.')[0]+'-label.png'
    for i in range(1, 360):
        #if(i  != 90 and i != 180 and i != 270):
        im = Image.open(path)
        out = im.rotate(i)
        base_name = images.split('.')[0]+'_rotated_'+str(i)+'.png'
        out.save(base_name)
    print(j, path)
        
    #Rotate 90
    #out = im.rotate(90)
    #Image
    #base_name = images.split('.')[0] +'_rotated_90.png'
    #Mask
    #base_name = images.split('-label')[0] +'_rotated_90-label.png'
    #out.save(base_name)

    #Rotate 180
    #out = im.rotate(180)
    #Image
    #base_name = images.split('.')[0] +'_rotated_180.png'
    #Mask
    #base_name = images.split('-label')[0] +'_rotated_180-label.png'
    #out.save(base_name)

    #Rotate270
    #out = im.rotate(270)
    #Image
    #base_name = images.split('.')[0] +'_rotated_270.png'
    #Mask
    #base_name = images.split('-label')[0] +'_rotated_270-label.png'
    #out.save(base_name)

    #Flip Vertical
    #out = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    #Image
    #base_name = images.split('.')[0] +'_mirrored_vertical.png'
    #Mask
    #base_name = images.split('-label')[0] +'_mirrored_vertical-label.png'
    #out.save(base_name)

    #Flip Horizontal
    #out = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    #Image
    #base_name = images.split('.')[0] +'_mirrored_horizontal.png'
    #Mask
    #base_name = images.split('-label')[0] +'_mirrored_horizontal-label.png'
    #out.save(base_name)

    #TO DO
    #Flip y=x
    #out = 
    #Image
    #base_name = images.split('.')[0] +'_mirrored_negative.png'
    #Mask
    #base_name = images.split('-label')[0] +'_mirrored_negative-label.png'
    #out.save(base_name)

    #TO DO
    #Flip y=-x
    #out = 
    #Image
    #base_name = images.split('.')[0] +'_mirrored_positive.png'
    #Mask
    #base_name = images.split('-label')[0] +'_mirrored_positive-label.png'
    #out.save(base_name)

    

