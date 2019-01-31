'''
Convert a stack of images to numpy array
'''
import os
import numpy as np
from PIL import Image
import argparse
import cv2
import glob

parser = argparse.ArgumentParser(description="Segmentation using tensorflow")

parser.add_argument('-m','--mode', type=str, default='rgb', help='rgb or seg')
parser.add_argument('-sf','--split_file', type=str, default=None, help='a text file indicate training or testing images')
parser.add_argument('-i','--img_dir', type=str, help='directory to one image or a folder contains images')
parser.add_argument('-o','--out_dir', type=str, required=True, help='directory to the output numpy file')
parser.add_argument('-d','--dim_order', type=str, default='nhwc', help='order of the dimentions of image, nhwc or nwhc or ncwh or nchw')

args = parser.parse_args()
split_file = args.split_file
img_dir = args.img_dir
out_dir = args.out_dir
mode = args.mode

BAD_IMAGES = ['/media/DATA/UnrealLandingDataset/AirSimCollectData/label/513-0.png',
              '/media/DATA/UnrealLandingDataset/AirSimCollectData/label/3112-0.png',
              '/media/DATA/UnrealLandingDataset/AirSimCollectData/label/3792-0.png',
              '/media/DATA/UnrealLandingDataset/AirSimCollectData/label/3965-0.png']

if split_file is not None:
    with open(split_file, 'r') as f:
        all_image_files = f.read().splitlines()
        all_images = []
        for i, image_file in enumerate(all_image_files):
            if mode == 'rgb':
                try:
                    image = np.array(Image.open(image_file).convert('RGB'))
                    all_images.append(image)
                except:
                    print(image_file)
                    continue
            elif mode == 'seg':
                if image_file in BAD_IMAGES:
                    print(image_file)
                    continue
                image = np.array(Image.open(image_file))
                all_images.append(image)

elif os.path.isdir(img_dir):
        all_image_files = sorted(glob.glob(os.path.join(img_dir, '*')))
        all_images = []
        for image_file in all_image_files:
            try:
                image = np.array(Image.open(image_file).convert('RGB'))
                all_images.append(image)
            except:
                print(image_file)
                continue
            # img = Image.open(image_file).convert('RGB')
            # image = np.asarray(img)
            # img.close()

else:
    try:
        image = np.asarray(Image.open(image_file).convert('RGB'))
    except:
        raise NameError("indicated image doesn't exit!")


all_images = np.array(all_images)
print(all_images.shape)
np.save(out_dir, all_images)
