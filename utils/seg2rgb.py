import numpy as np
import glob
from PIL import Image


ROOT = '/media/DATA/UnrealLandingDataset/AirSimCollectData/'
OUT_DIR = ROOT + '/results/'

def get_seg2rgb_map(cmap_file, number_of_classes):
    f = open(cmap_file, 'r')
    all_rows = f.read().splitlines()
    seg2rgb_map = {}
    for row in all_rows:
        seg = row.split('\t')[0]
        rgb = list(map(lambda x: int(x)/255, row.split('\t')[1][1:-1].split(',')))
        rgb.append(1)
        seg2rgb_map[int(seg)] = rgb

    cmap_list = list(seg2rgb_map.values())[:number_of_classes]
    return cmap_list, seg2rgb_map

def seg2rgb(seg, seg2rgb_map):
    '''
    :Params::
        seg: integer image with size (W,H)
        seg2rgb_map: a dictionary mappign interger classes to RGB values
    :Return::
        seg_rgb: an RGB image showing segmentation, (W,H,3)
    '''
    seg_rgb = np.array(seg.shape[0], seg.shape[1], 3)
    all_classes = np.unique(seg)
    for i in all_classes:
        seg_rgb[np.where(seg == i)] = seg2rgb_map[i]
        
    return seg_rgb

def save_pred_results(segmentation, output_dir, cmap_list):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('unreal roof cmap', cmap_list, number_of_classes)
    # define the bins and normalize
    bounds = np.linspace(0,number_of_classes,number_of_classes+1)
    norm = mpl.colors.BoundaryNorm(bounds, number_of_classes)


    fig, (ax1, ax2) = plt.subplots(figsize=(16, 8), nrows=2, ncols=1)

    ax2.imshow(segmentation,cmap=cmap, norm=norm)
    