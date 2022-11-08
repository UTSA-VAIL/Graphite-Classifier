

from simplejson import OrderedDict
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
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
from torchvision import transforms




palette = np.array([
                    [0, 0, 0],         # black
                    [  0,   0, 255],   # blue
                    [255,   0,   0],   # red
                    [255,   0, 255],   # purple
                    ])



def create_test_image(img, labels, save_path):
    print(f'Generating Test image: {save_path}')
    plt.figure(figsize=(10,10))
    subplot_val = 131
    plt.subplot(subplot_val)
    plt.imshow(np.asarray(img))
    plt.title('Image')
    plt.axis('off')
    plt.subplots_adjust(left=0.01,
                    bottom=0.01, 
                    right=1.0, 
                    top=1.0, 
                    wspace=0.01, 
                    hspace=0.01)
    for label in labels:
        subplot_val += 1
        plt.subplot(subplot_val)
        colorized_label = palette[label[0]]
        plt.imshow(colorized_label)
        plt.title(label[1])
        plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    plt.close()







def test_analysis(model, metrics, data_dir, exp_dir, test_imgs, size):
    all_path = f"{exp_dir}/all"
    if os.path.exists(all_path):
        shutil.rmtree(all_path)
    os.mkdir(all_path)
    data_transforms = transforms.Compose([transforms.ToTensor()])
    ious = []
    for image in test_imgs:
        image = pathlib.Path(image).stem
        base_name = image.split('/')[-1]
        image = Image.open(f'{data_dir}/Labeled/Images/{base_name}.png').convert("RGB")
        mask = Image.open(f'{data_dir}/Labeled/Split_Mask/all/{base_name}-label.png').convert("L")

        img = data_transforms(image).cuda()
        img = torch.unsqueeze(img, 0)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long().cuda()

        with torch.no_grad():
            output = model(img)
        if isinstance(output, OrderedDict):
            output = output['out']   

        #generate test image display
        class_all_path = f"./{exp_dir}/all"
        save_path = f'{class_all_path}/test_{base_name}.png'
        mask_print = (np.asarray(mask.cpu()), 'Ground Truth')
        output_print = (torch.squeeze(torch.argmax(output.cpu(), dim=1), 0), 'Prediction')
        prints = [mask_print, output_print]
        create_test_image(image, prints, save_path)

        y_pred = output
        y_true = mask
        iou = metrics(y_pred, y_true)
        ious.append(iou)
        print(f"iou: {round(iou.item(), 4)}")

    miou = torch.mean(torch.stack(ious), dim=0)
    msg = f"\nMean Intersection Over Union of Test Samples: {round(float(miou), 4)}"
    print(msg)
    with open(os.path.join(exp_dir, f'log.csv'), 'a', newline='') as logfile:
        logfile.write(msg)















