import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import tifffile as tiff
from PIL import Image
from glob import glob
from utility_modules.utils import *
from tqdm import tqdm
import cv2
import tifffile as tiff
import os
import h5py
import imageio
import pandas as pd
import re
from torchvision import transforms
import shutil
import torchvision.transforms.functional as tf
import imageio
from utility_modules.datasets import SyntheticDataset, RealDataset, EPFLDataset


GP = {
    'hl60': '/data/vadineanus/data_label_error_study/hl60_tiff/',
    'granulocytes': '/data/vadineanus/data_label_error_study/granulocytes_tiff/',
    'epfl_train': '/data/vadineanus/data_label_debiasing/epfl/train/',
    'epfl_test': '/data/vadineanus/data_label_debiasing/epfl/test/',
    'inclusion_hl60': '/data/vadineanus/data_label_debiasing/inclusion_data_hl60/',
    'inclusion_granulocytes': '/data/vadineanus/data_label_debiasing/inclusion_data_granulocytes/',
    'inclusion_epithelial_train': '/data/vadineanus/data_label_debiasing/inclusion_real_data_train/',
    'inclusion_epithelial_test': '/data/vadineanus/data_label_debiasing/inclusion_real_data_test/',
    'models': '/data/vadineanus/data_label_debiasing/models/'
}


def get_dice(predict, target, return_dice=True):
    tp = torch.sum(predict * target)
    fp = torch.sum(predict) - tp
    fn = torch.sum(target) - tp

    if return_dice:
        return 2 * tp / (2 * tp + fp + fn)
    else:
        return tp, fp, fn


def center_transform(image, mask):
#     image = Image.fromarray(image)
#     mask = Image.fromarray(mask)

#     image = tf.center_crop(image, output_size=(512, 512))
#     mask = tf.center_crop(mask, output_size=(512, 512))

    return np.array(image) / np.max(image), np.array(mask)


def transform_real(image, mask, output_size=(256, 256)):

#     image = Image.fromarray(image)
#     mask = Image.fromarray(mask)

#     # Random horizontal flipping
#     if random.random() > 0.5:
#         image = tf.hflip(image)
#         mask = tf.hflip(mask)

#     # Random vertical flipping
#     if random.random() > 0.5:
#         image = tf.vflip(image)
#         mask = tf.vflip(mask)

    return np.array(image) / np.max(image), np.array(mask)


def load_debiasing_model(drop_p_unclean, incl_p_unclean, max_iter_unclean, drop_p, incl_p, max_iter, rep, n_volumes, dataset='hl60', debiasing_type='debiasing_input',
                         cuda_no=0, total_slices=0, model_name='unet'):
    if 'input' in debiasing_type:
        n_channels = 2
        n_classes = 2
    elif 'full' in debiasing_type:
        n_channels = 1
        n_classes = 2

    if 'epithelial' in dataset:
        n_channels += 2    
    
    if total_slices > 0:
        total_slices = f'_total_slices_{total_slices}'
    else:
        total_slices = ''

    debiasing_model = get_model(n_channels=n_channels, n_classes=n_classes, type=model_name)
    debiasing_model.load_state_dict(torch.load(GP['models'] +
                                               f'{model_name}_dice2_{dataset}_{debiasing_type}_slice_'
                                               f'{drop_p_unclean}_{incl_p_unclean}_{max_iter_unclean}_'
                                               f'v{n_volumes}_'
                                               f'{drop_p}_{incl_p}_{max_iter}_'
                                               f'{rep}{total_slices}_best.pt',
                                               map_location=get_device(cuda_no=cuda_no)))
    debiasing_model = debiasing_model.to(get_device(cuda_no=cuda_no))
#     debiasing_model.eval()
    
    return debiasing_model


columns = ['dataset', 'debiasing_mode', 'omission_hq', 'inclusion_hq', 'qmax_hq',
           'omission_lq', 'inclusion_lq', 'qmax_lq',
           'n_volumes', 'rep', 'dice_biased', 'dice_debiased', 'dice_diff']
rows = []

# Reading from bash file
mode = sys.argv[1]
err_type = sys.argv[2]
n_volumes = int(sys.argv[3])
cuda_no = int(sys.argv[4])
total_slices = int(sys.argv[5])
dataset = str(sys.argv[6])
model_name = str(sys.argv[7])

if 'epithelial' in dataset:
    ds_extension = 'epithelial'
elif 'hl60'in dataset:
    ds_extension = 'hl60'
elif 'granulocytes' in dataset:
    ds_extension = 'granulocytes'
elif 'epfl' in dataset:
    ds_extension = 'epfl'

if err_type == 'omission':
    setup_list = [(0.1, 0.0, 0), (0.2, 0.0, 0), (0.3, 0.0, 0)]
    drop_p = 0.7
    incl_p = 0.0
    max_iter = 0
elif err_type == 'inclusion':
    setup_list = [(0.0, 0.1, 0), (0.0, 0.2, 0), (0.0, 0.3, 0)]
    drop_p = 0.0
    incl_p = 0.7
    max_iter = 0
elif err_type == 'bias':
    setup_list = [(0.0, 0.0, 2), (0.0, 0.0, 4), (0.0, 0.0, 6)]
    drop_p = 0.0
    incl_p = 0.0
    max_iter = 6


for setup in setup_list:
    drop_p_unclean, incl_p_unclean, max_iter_unclean = setup

    for rep in [0, 1, 2, 3, 4]:

        try:
            debiasing_model = load_debiasing_model(drop_p_unclean=drop_p_unclean, incl_p_unclean=incl_p_unclean, 
                                                   max_iter_unclean=max_iter_unclean,
                                                   drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                                   rep=rep, n_volumes=n_volumes, cuda_no=cuda_no,
                                                   debiasing_type=mode,
                                                   dataset=ds_extension, total_slices=total_slices, model_name=model_name)
        except Exception as e:
            print(e)
            print("Missing model!", flush=True)
            continue


        start_time = time.time()
        print(f'{mode}, omission_model={drop_p}, inclusion_model={incl_p}, qmax_model={max_iter}, '
              f'omission_data={drop_p}, inclusion_data={incl_p}, qmax_data={max_iter}, '
              f'n_volumes={n_volumes}, rep={rep}', flush=True)

        if 'hl60' in dataset or 'granulocytes' in dataset: 
            data_gt = SyntheticDataset(root=GP[dataset],
                                       mode='normal', train=True, val=False, train_test_split=1.0,
                                       seed=rep, file_range=(25, 30), binary_mask=True, drop_p=0,
                                       incl_p=0, max_iter=0, transform=center_transform, debiasing_model=None,
                                       debiasing_device=None,
                                       bias_assumption='slice', constant_perturbations=True)

            data_biased = SyntheticDataset(root=GP[dataset],
                                           mode='normal', train=True, val=False, train_test_split=1.0,
                                           seed=rep, file_range=(25, 30), binary_mask=True, drop_p=drop_p,
                                           incl_p=incl_p, max_iter=max_iter,
                                           transform=center_transform, debiasing_model=None, debiasing_device=None,
                                           bias_assumption='slice', constant_perturbations=True)

        
        elif 'epfl' in dataset:
            data_gt = EPFLDataset(root=GP['epfl_test'],
                                     mode='normal', train=True, val=False, train_test_split=1.0,
                                     seed=rep, transform=transform_real, bias_assumption='slice',
                                     drop_p=0, incl_p=0, max_iter=0,
                                     debiasing_model=None, debiasing_device=None, constant_perturbations=True)
            
            data_biased = EPFLDataset(root=GP['epfl_test'],
                                     mode='normal', train=True, val=False, train_test_split=1.0,
                                     seed=rep, transform=transform_real, bias_assumption='slice',
                                     drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                     debiasing_model=None, debiasing_device=None, constant_perturbations=True)
        

        tp_deb, fp_deb, fn_deb = 0, 0, 0
        tp_b, fp_b, fn_b = 0, 0, 0
        
        for i in range(len(data_gt)):
            img, lab_gt = data_gt[i]
            _, lab_biased = data_biased[i]

            if 'input' in mode:
                if len(img.shape) == 2:
                    img = img.unsqueeze(dim=0)
                input4debiasing = torch.zeros((img.shape[0] + 1, img.shape[1], img.shape[2]))
                input4debiasing[:img.shape[0], :, :] = img
                input4debiasing[img.shape[0], :, :] = lab_biased
            input4debiasing = input4debiasing.unsqueeze(dim=0).to(get_device(cuda_no))
            
            _, lab_debiased = torch.max(debiasing_model(input4debiasing).squeeze().detach(), dim=0)

            tp_deb_, fp_deb_, fn_deb_ = get_dice(lab_debiased.detach().cpu(), lab_gt, return_dice=False)
            tp_b_, fp_b_, fn_b_ = get_dice(lab_biased.detach().cpu(), lab_gt, return_dice=False)

            tp_deb += tp_deb_
            fp_deb += fp_deb_
            fn_deb += fn_deb_

            tp_b += tp_b_
            fp_b += fp_b_
            fn_b += fn_b_

        dice_deb = 2 * tp_deb / (2 * tp_deb + fp_deb + fn_deb)
        dice_b = 2 * tp_b / (2 * tp_b + fp_b + fn_b)

        row = [dataset, mode,
               drop_p_unclean, incl_p_unclean, max_iter_unclean,
               drop_p, incl_p, max_iter,
               n_volumes, rep, dice_b.item(), dice_deb.item(), dice_deb.item() - dice_b.item()]
        rows.append(row)

        print(f'It took: {np.round(time.time() - start_time, 4)} s', flush=True)

df = pd.DataFrame(rows, columns=columns)

if total_slices > 0:
    df.to_csv(f'results_unclean_{model_name}_{ds_extension}_{err_type}_{mode}_vols_{n_volumes}_total_slices_{total_slices}.csv')
else:
    df.to_csv(f'results_unclean_{model_name}_{ds_extension}_{err_type}_{mode}_vols_{n_volumes}.csv')
    