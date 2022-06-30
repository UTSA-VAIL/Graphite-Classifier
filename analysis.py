

from typing import OrderedDict
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import pandas as pd
import argparse
from PIL import Image, ImageOps
from helper import per_class_mIoU
import numpy as np
import glob
from tqdm import tqdm
import pathlib
import os
import shutil
import csv


# palette = np.array([[  255,   255,  255],   # white
#                     [0, 0, 0],         # black
#                     [255,   0,   0],   # red
#                     [  0,   0, 255],   # blue
#                     [255,   0, 255],   # purple
#                     ])


palette = np.array([
                    [0, 0, 0],         # black
                    [255,   0,   0],   # red
                    [  0,   0, 255],   # blue
                    [255,   0, 255],   # purple
                    ])



def create_test_image(img, mask, output, save_path):
    print(f'Generating Test image: {save_path}')
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.imshow(img[0,...].transpose(1,2,0))
    plt.title('Image')
    plt.axis('off')
    plt.subplot(132)
    colorized_mask = palette[mask]
    plt.imshow(colorized_mask)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(133)
    colorized_output = palette[output]
    plt.imshow(colorized_output)
    plt.title('Segmentation Output')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()





def test_analysis_multi(model, metrics, data_dir, exp_dir, test_imgs, c):
    if not model:
        print('No model: loading from file')
        model = torch.load(f'./{exp_dir}/weights_{c}.pt')
    model.eval()

    class_path = f"./{exp_dir}/{c}"

    print('\nRunning Tests...')
    print(f'Viewable tests saved to: "{class_path}/"\n')
    mious = []


    for image in test_imgs:
        image = pathlib.Path(image).stem
        base_name = image.split('/')[-1]
        #CT-Org
        #nums = base_name.strip('images-').strip('.png')
        #ino, kno = nums.split('_')
        #ino = int(ino)
        #kno = int(kno)
        # Read  a sample image and mask from the data-set
        #img = cv2.imread(f'{data_dir}/Images/images-{ino:03d}_{kno:03d}.png').transpose(2,0,1).reshape(1,3,512,512)
        #mask = Image.open(f'{data_dir}/Masks/{c}/labels-{ino:03d}_{kno:03d}.png').convert("L")
        img = cv2.imread(f'{data_dir}/Images/{base_name}.png').transpose(2,0,1).reshape(1,3,512,512)
        mask = Image.open(f'{data_dir}/Masks/{c}/{base_name}-label.png').convert("L")
        mask = (np.asarray(mask)/255).astype(int).reshape(1,1,512,512)
        with torch.no_grad():
            output = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
        if isinstance(output, OrderedDict):
            output = output['out']

        y_true = torch.from_numpy(mask).type(torch.int32).cpu()#.numpy()
        y_pred = output.data.cpu()#.numpy()
        miou = metrics(y_pred, y_true)
        mious.append(miou)
        print(f"'{image.replace('images', 'test')}.png' - Test IoU: {miou:.4f}")

    avg_miou = sum(mious) / len(mious)
    msg = f'Avg Test mIoU: {avg_miou:.4f}'
    print(f'{msg}\n')
    with open(os.path.join(class_path, f'log_{c}.csv'), 'a', newline='') as file:
        file.write(msg)

    return avg_miou


def test_analysis_multi_all(models, metrics, data_dir, exp_dir, test_imgs, threshold):

    all_path = f"./{exp_dir}/all"
    if os.path.exists(all_path):
        shutil.rmtree(all_path)
    os.mkdir(all_path)

    mious = []
    for image in test_imgs:
        image = pathlib.Path(image).stem
        base_name = image.split('/')[-1]
        #nums = base_name.strip('images-').strip('.png')
        #ino, kno = nums.split('_')
        #ino = int(ino)
        #kno = int(kno)
        img = cv2.imread(f'{data_dir}/Images/{base_name}.png').transpose(2,0,1).reshape(1,3,512,512)

        outputs = {}
        max_array = None
        max_array_assigned = False
        for key in models.keys():
            mask = Image.open(f'{data_dir}/Masks/{key}/{base_name}-label.png').convert("L")
            mask = (np.asarray(mask)/255).astype(int).reshape(1,1,512,512)
            model = models[key]
            with torch.no_grad():
                a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
            if isinstance(a, OrderedDict):
                a = a['out']
            model_output = (a.cpu().detach().numpy()[0][0] > threshold) * key

            class_path = f"./{exp_dir}/{key}"
            save_path = f'{class_path}/test_{base_name}.png'
            mask = mask * key
            create_test_image(img, mask[0][0], model_output, save_path)


            outputs[key] = model_output
            if not max_array_assigned:
                max_array = model_output
                max_array_assigned = True
            else:
                max_array = np.maximum(max_array, model_output)


        mask_path = f'{data_dir}/Masks/all/{base_name}-label.png'
        mask = Image.open(mask_path).convert("L")
        mask = (np.asarray(mask)).astype(int).reshape(1,1,512,512)

        y_true = torch.from_numpy(mask).type(torch.int32).cpu()#.numpy()
        y_pred = torch.from_numpy(max_array).type(torch.int32).cpu()
        miou = metrics(y_pred, y_true)
        mious.append(miou)
        print(f"'{image.replace('images', 'test')}.png' - Test IoU: {miou:.4f}")
    

        output = np.reshape(max_array,(512,512))
        save_path = f'{all_path}/test_{base_name}.png'
        #save_path = f'{all_path}/test_{nums}.png'

        create_test_image(img, mask[0][0], output, save_path)
        print()
    print(f'Avg Test mIoU : {sum(mious) / len(mious):.4f}')






def test_analysis_multi_all_nonbinary(model, metrics, data_dir, exp_dir, test_imgs, threshold):

    all_path = f"./{exp_dir}/all"
    if os.path.exists(all_path):
        shutil.rmtree(all_path)
    os.mkdir(all_path)

    ious = []
    for image in test_imgs:
        image = pathlib.Path(image).stem
        base_name = image.split('/')[-1]
        #nums = base_name.strip('images-').strip('.png')
        #ino, kno = nums.split('_')
        #ino = int(ino)
        #kno = int(kno)
        img = cv2.imread(f'{data_dir}/Images/{base_name}.png').transpose(2,0,1).reshape(1,3,512,512)

        mask = Image.open(f'{data_dir}/Masks/{base_name}-label.png').convert("L")
        mask = (np.asarray(mask)).astype(int).reshape(1,1,512,512)
        with torch.no_grad():
            a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)
        if isinstance(a, OrderedDict):
            a = a['out']
        model_output = torch.argmax(a.cpu(), dim=1).numpy().squeeze()

        class_path = f"./{exp_dir}/all"
        save_path = f'{class_path}/test_{base_name}.png'
        create_test_image(img, mask[0][0], model_output, save_path)



        y_true = torch.from_numpy(mask).cpu()#.numpy()
        y_pred = a.cpu()
        iou = metrics(y_pred, y_true)
        ious.append(iou)
        print(f"iou: {round(iou.item(), 4)}")
        #print(torch.argmax(y_pred, dim=1))
        #print(y_true)
        #print(f"'{image.replace('images', 'test')}.png' - Test IoU: {miou:.4f}")
    

        #output = np.reshape(max_array,(512,512))
        #save_path = f'{all_path}/test_{base_name}.png'
        #save_path = f'{all_path}/test_{nums}.png'

        #create_test_image(img, mask[0][0], output, save_path)
        #print()
    #print(f'Avg Test mIoU : {sum(mious) / len(mious):.4f}')
    miou = torch.mean(torch.stack(ious), dim=0)
    print(f"\nMean Intersection Over Union of Test Samples: {round(float(miou), 4)}")












