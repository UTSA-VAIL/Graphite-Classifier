import glob
import os
import shutil
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import math
from image_slicer import slice
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', default=None, type=int)
parser.add_argument('--data_path', default='/data/graphite', type=str)
args = parser.parse_args()

base_path = args.data_path

image_source_path = f'{base_path}/dataset/Labeled/Images/'
mask_source_path = f'{base_path}/dataset/Labeled/Masks/'
ul_source_path = f'{base_path}/dataset/Unlabeled/'
processed_path = f'{base_path}/processed'
processed_l_path = f'{processed_path}/Labeled'
processed_ul_path = f'{processed_path}/Unlabeled/'
processed_image_path = f'{processed_l_path}/Images/'
processed_mask_path = f'{processed_l_path}/Masks/'
tiles_path = f'{processed_path}/Tiles/'
tiles_l_path = f'{tiles_path}/Labeled/'
tiles_ul_path = f'{tiles_path}/Unlabeled/'
tiles_image_path = f'{tiles_path}/Labeled/Images/'
tiles_mask_path = f'{tiles_path}/Labeled/Masks/'
tiles_split_mask_path = f'{tiles_path}/Labeled/Split_Mask/'
tiles_split_mask_all_path = f'{tiles_split_mask_path}/all/'
temp_path = f'{base_path}/temp/'

colors_list = [
    (0, 0, 0),       #black
    (255, 0, 0),    #red
    (0, 0, 255),    #blue
    (255, 0, 255),  #purple
]

# Tile parameters
wTile = 384
hTile = 384


def is_similar(pixel_a, pixel_b):
    return math.sqrt(sum((pixel_b[i]-pixel_a[i])**2 for i in range(3)))

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size > size:
            #points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points



get_cropped_images = True
get_deblurred_masks = True
get_tiles = True
get_class_split = True

if get_cropped_images:
    print('\nBegin cropping...')
    if os.path.exists(processed_path):
        shutil.rmtree(processed_path)
    os.mkdir(processed_path)
    os.mkdir(processed_l_path)
    os.mkdir(processed_ul_path)
    os.mkdir(processed_image_path)
    os.mkdir(processed_mask_path)

    #crop unlabeled images
    images = glob.glob(ul_source_path + "*")
    for image in images:
        image_stem = image.split('/')[-1]
        im = Image.open(image)
        w, h = im.size
        new_h = h * 0.94
        cropped_im = im.crop((0, 0, w, new_h))
        image_name = image_stem.replace('.tiff', '.png').replace('.tif', '.png')
        new_path = processed_ul_path + image_name
        print(new_path)
        cropped_im.save(new_path)

    #crop images
    images = glob.glob(image_source_path + "*")
    for image in images:
        image_stem = image.split('/')[-1]
        im = Image.open(image)
        w, h = im.size
        new_h = h * 0.94
        cropped_im = im.crop((0, 0, w, new_h))
        image_name = image_stem.replace('.tiff', '.png').replace('.tif', '.png')
        new_path = processed_image_path + image_name
        print(new_path)
        cropped_im.save(new_path)

    #crop masks
    masks = glob.glob(mask_source_path + "*")
    for mask in masks:
        mask_stem = mask.split('/')[-1]
        im = Image.open(mask_source_path + mask_stem.replace('.tiff', '.tif'))
        w, h = im.size
        new_h = h * 0.94
        cropped_im = im.crop((0, 0, w, new_h))
        mask_name = mask_stem.replace('.tiff', '.png').replace('.tif', '.png')
        new_path = processed_mask_path + mask_name
        print(new_path)
        cropped_im.save(new_path)


if get_deblurred_masks:
    print('\nBegin deblurring...')
    masks = glob.glob(processed_mask_path + '*')

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)

    for mask_name in masks:
        #mask_name = masks[m]
        base_mask_name = mask_name.split('/')[-1].replace('.tiff', '.png').replace('.tif', '.png')
        print(mask_name)

        mask_im = Image.open(mask_name)
        mask_im.save(temp_path + base_mask_name)
        img = Image.open(temp_path + base_mask_name)

        w, h = img.size
        pixels = img.load()

        for x in range(w):
            for y in range(h):
                color_rankings = [
                    is_similar(pixels[x, y], colors_list[0]),
                    is_similar(pixels[x, y], colors_list[1]),
                    is_similar(pixels[x, y], colors_list[2]),
                    is_similar(pixels[x, y], colors_list[3])
                ]
                min_rank = min(color_rankings)
                min_index = color_rankings.index(min_rank)
                pixels[x, y] = colors_list[min_index]

        for x in range(w):
            for y in range(h):   
                if x > 0 and x < w-1 and y > 0 and y < h-1:
                    neighbor_pixels = [
                        pixels[x-1, y-1],   
                        pixels[x, y-1], 
                        pixels[x+1, y-1],
                        pixels[x-1, y],    
                        pixels[x+1, y], 
                        pixels[x-1, y+1],   
                        pixels[x, y+1], 
                        pixels[x+1, y+1], 
                    ]

                    if pixels[x, y] not in neighbor_pixels:
                        pixels[x, y] = max(set(neighbor_pixels), key = neighbor_pixels.count)

        img.save(temp_path + base_mask_name)


if get_tiles:
    print('\nBegin tile creation...')
    if os.path.exists(tiles_path):
        shutil.rmtree(tiles_path)
    os.mkdir(tiles_path)
    os.mkdir(tiles_l_path)
    os.mkdir(tiles_image_path)
    os.mkdir(tiles_mask_path)
    os.mkdir(tiles_ul_path)

    images = glob.glob(processed_image_path + '*')
    masks = glob.glob(temp_path + '*')
    ul_images = glob.glob(processed_ul_path + '*')


    for type_idx, i_type in enumerate([ul_images, images, masks]):
        type_path = tiles_ul_path
        if type_idx == 1:
            type_path = tiles_image_path
        elif type_idx == 2:
            type_path = tiles_mask_path
        for image_file in i_type:
            print(image_file)
            img = Image.open(image_file)
            i_name = image_file.split('/')[-1].replace('.tiff', '.png').replace('.tif', '.png')
            w, h = img.size
            X_points = start_points(w, wTile, 0.0)
            Y_points = start_points(h, hTile, 0.0)
            print(w, X_points)
            print(h, Y_points)

            count = 0

            for i in Y_points:
                for j in X_points:
                    cropped_im = img.crop((j, i, j+wTile, i+hTile))
                    cropped_im.save(type_path + i_name.replace('_C', f'_{count:02}'))
                    count += 1



if get_class_split:
    print('\nBegin binary class splitting...')
    if os.path.exists(tiles_split_mask_path):
        shutil.rmtree(tiles_split_mask_path)
    os.mkdir(tiles_split_mask_path)
    os.mkdir(tiles_split_mask_all_path)

    masks = glob.glob(tiles_mask_path + '*')
    if not args.num_classes:
        num_classes = 4
    else:
        num_classes = args.num_classes

    uniques = []
    for mask in masks:
        img = Image.open(mask).convert('L')
        img_arr = np.array(img)
        im_uniques = np.unique(img_arr)
        for u in im_uniques:
            if u not in uniques:
                uniques.append(u)
    uniques.sort()
    print(uniques)

    for mask in masks:
        img = Image.open(mask).convert('L')
        img_name = mask.split('/')[-1]
        img_arr = np.array(img)
        all_arr = np.copy(img_arr)
        for c, color in enumerate(uniques):
            if c == 0:
                continue
            class_dir_path = tiles_split_mask_path + f'{c}/'
            if not os.path.exists(class_dir_path):
                os.mkdir(class_dir_path)
            one_class_array = np.copy(img_arr)
            one_class_array[one_class_array != color] = 254
            one_class_array[one_class_array == color] = 255
            one_class_array[one_class_array == 254] = 0
            all_arr[all_arr == color] = c
            all_arr[all_arr == 3] = 2
            one_class_img = Image.fromarray(one_class_array)
            one_class_img.save(class_dir_path + img_name)
        #print(np.unique(all_arr))
        all_class_img = Image.fromarray(all_arr)
        all_class_img.save(tiles_split_mask_all_path + img_name)





 

