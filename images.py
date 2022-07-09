from PIL import Image
import PIL

import os
from os import listdir
import numpy as np


# get the path/directory

i = 0
folder_dir = "/home/mohanadhas/Documents/TestSeg/DeepLabv3FineTuning/Pellets/Augmentations/Images/"
for images in os.listdir(folder_dir):
    path = folder_dir +images

    a = images.split('.png')
    print(a)
    if(len(a) != 2):
        out = a[0]+a[1]+'.png'
        os.rename(path,folder_dir+out)
        print(out)
    print()
    #b = a[1].split('.png_')[-1]
    #if(b.split('.')[0]!=''):
        #out = a[0]+'_'+b.split('.')[0]+'-label.png'
        

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


    #Mask Correction
    '''
    print(i, path)
    i += 1
    im = Image.open(path)

    picture = Image.open(path)
    width, height = picture.size
    #im = PIL.Image.new(mode = "RGB", size=(width, height))
    width, height = im.size
    color = []
    for x in range (width):
        for y in range (height):
            color = []
            current_color = im.getpixel( (x,y) )[0]
            if(current_color == 0 or current_color == 3):
                new_color = (0, 0, 0)
                #new_color = 0
            elif current_color == 1:
                new_color = (1, 1, 1)
                #new_color = 100
            elif current_color == 2:
                new_color = (2, 2, 2)
                #new_color = 200
            im.putpixel( (x,y), new_color)
    im.save(path)
    '''

'''
            color.append(current_color)
            print(current_color)
    colors = np.array(color)
    print(np.unique(colors))
    '''

    
