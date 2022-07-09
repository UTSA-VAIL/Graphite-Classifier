from PIL import Image
import PIL

import os
from os import listdir
import numpy as np


print("Starting mask transformations to: ", os.getcwd())
# get the path/directory
j = 0
folder_dir = "/home/mohanadhas/Documents/TestSeg/DeepLabv3FineTuning/Pellets/Masks/"
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
    
    '''
    for i in range(1, 360):
        #if(i  != 90 and i != 180 and i != 270):
        im = Image.open(path)
        out = im.rotate(i)
        base_name = images.split('-label')[0]+'_rotated_'+str(i)+'-label.png'
        out.save(base_name)
    j+=1
    print(j, path)
    '''
    
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
    print(j, path)
    j += 1
    im = Image.open(path)

    picture = Image.open(path)
    width, height = picture.size
    #colors = picture.convert('RGB').getcolors()

    im = PIL.Image.new(mode = "RGB", size=(width, height))
    color = []
    for x in range (width):
        for y in range (height):
            color = []
            current_color = picture.getpixel( (x,y) )
            #print(current_color)
            if(current_color == (0, 0, 0 ) ):
                new_color = (1, 1, 1)
                #new_color = 1
            elif current_color == (1, 1, 1):
                new_color = (2, 2, 2)
                #new_color = 2
            elif current_color == (2, 2, 2):
                #print("Hello")
                new_color = (3, 3, 3)
                #new_color = 3
            elif current_color == (3, 3, 3):
                new_color = (3, 3, 3)
                print("ERROROROROROROROR")
                #new_color = 4
            #if(current_color != (0,0,0)):
            #print(current_color)
            #color.append(current_color)
            im.putpixel( (x,y), new_color)
    #colors = np.array(color)
    #print(np.unique(colors))
    #print(colors)
        #print("Color:", current_color)
    im.save(path)
    '''

