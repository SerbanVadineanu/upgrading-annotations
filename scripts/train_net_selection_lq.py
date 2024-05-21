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
import math
from datetime import datetime

torch.autograd.set_detect_anomaly(False)    # Speedup
torch.autograd.profiler.profile(False)      # Speedup
torch.autograd.profiler.emit_nvtx(False)    # Speedup
torch.set_flush_denormal(True)              # Added this to try to fix training slowing down. It apporximates very small numbers to 0


class SyntheticDatasetSelectionLQ(torch.utils.data.Dataset):
    @staticmethod
    def get_filenames(root, tiff_volumes_names):
        img_list = []
        lab_list = []
        for tiff_num in tiff_volumes_names:
            crt_img_files = glob(root + f'image-final_{tiff_num}_*.tiff')
            crt_label_files = glob(root + f'image-labels_{tiff_num}_*.tiff')

            crt_img_files = sorted(crt_img_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            crt_label_files = sorted(crt_label_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            img_list.extend(crt_img_files)
            lab_list.extend(crt_label_files)

        return img_list, lab_list

    def __init__(self, root, mode, train=True, val=False, train_test_split=0.7,
                 seed=42, file_range=(0, 3), binary_mask=True, drop_p=0,
                 incl_p=0, max_iter=0, transform=None, debiasing_model=None, debiasing_device=None,
                 bias_assumption='slice', constant_perturbations=False, slices_per_volume=0, total_slices=0,
                 drop_p_unclean_debiasing=0, incl_p_unclean_debiasing=0, max_iter_unclean_debiasing=0, hq_percentage=0):

        assert file_range[0] >= 0 and file_range[1] <= 30
        assert 0 <= drop_p < 1

        if 'unclean' in mode:
            assert drop_p_unclean_debiasing is not None or incl_p_unclean_debiasing is not None or max_iter_unclean_debiasing is not None
            self.drop_p_unclean_debiasing = drop_p_unclean_debiasing
            self.incl_p_unclean_debiasing = incl_p_unclean_debiasing
            self.max_iter_unclean_debiasing = max_iter_unclean_debiasing

        super().__init__()
        self.train = train
        self.val = val
        self.binary_mask = binary_mask
        self.drop_p = drop_p
        self.incl_p = incl_p
        self.morph_op = max_iter > 0
        # if 'debiasing' then the data set provides the bias as label else it acts as a usual segmentation data set
        self.mode = mode
        self.kernel = np.ones((3, 3), np.uint8)
        self.transform = transform
        self.max_iter = max_iter
        self.debiasing_model = debiasing_model  # Already trained debiasing model
        self.debiasing_device = debiasing_device  # The cuda device on which we have the model
        self.file_range = file_range
        self.bias_assumption = bias_assumption    # Can be 'cell', 'volume', 'slice'
        # This variable is true if we choose to add data set-wise perturbations that we can replicate
        # As in the MIDL 2022 paper
        # Otherwise the perturbations are chosen whenever an image is read
        self.constant_perturbations = constant_perturbations
        # Save kept indices as an attribute
        self.hq_indices = []
        # Variable used to know where is the cutoff point between the main and included labels
        self.inclusion_point = 0    # 0 is default, it changes if we perform inclusions
        # Initialize a couple of variables that we will later use
        self.dropped_labels = []
        self.included_labels = []
        self.operations_list = []
        self.iterations_list = []
        self.dropped_labels_unclean_debiasing = []
        self.included_labels_unclean_debiasing = []
        self.operations_list_unclean_debiasing = []
        self.iterations_list_unclean_debiasing = []
        # If we have a data set with two cell types this indicates the index after which the other cell type begins
        self.omission_labels_offset = 0
        self.seed = seed
        # Debugging var
        self.LQ_labels = []

        tiff_volumes_names = ['00' + f"{x:02d}" for x in range(*file_range)]
        self.img_filenames, self.lab_filenames = SyntheticDatasetSelectionLQ.get_filenames(root, tiff_volumes_names)
        
        # We select fewer slices from each volume
        if slices_per_volume > 0 or total_slices > 0:
            kept_indices = []
            volumes = []    # List of containing the slices for each selected volume
            for vol in range(*file_range):
                volume_slices = list(filter(lambda x: int(x[1].split('_')[-2]) == vol, enumerate(self.img_filenames)))
                # Trim volume to 75 slices (yes, hardcoded, yes)
                no_slices = len(volume_slices)
                excess_slices = no_slices - 75    # Hardcoded, yes, BAD
                volume_slices = volume_slices[excess_slices // 2:]
                volume_slices = volume_slices[:-(excess_slices - excess_slices // 2)]

                volumes.append(volume_slices)

            if total_slices == 0:
                total_slices = slices_per_volume * len(volumes)
            volume_size = len(volumes[0])
            volume_iterations = math.ceil(total_slices / volume_size)
            remainder_slices = total_slices % volume_size

            volumes_indices = list(np.arange(len(volumes)))
            random.seed(seed)
            random.shuffle(volumes_indices)
            
            for it in range(volume_iterations):
                if it > 0:
                    volumes_indices = volumes_indices[1:] + volumes_indices[:1]    # cyclic permutation

                # On the last iteration, we take the middle slice from equally-sized volume blocks
                if it == volume_iterations - 1:
                    local_indices = np.round(volume_size / (remainder_slices * 2) +
                                             np.arange(remainder_slices) * volume_size / remainder_slices).astype(int)
                # On all the other iterations, we take all slices
                else:
                    local_indices = np.arange(volume_size)

                for i, local_index in enumerate(local_indices):
                    volume_index = volumes_indices[i % len(volumes_indices)]
                    global_index = volumes[volume_index][local_index][0]
                    kept_indices.append(global_index)
            
            # We only keep the slices that we have sampled
            self.img_filenames = list(np.array(self.img_filenames)[kept_indices])
            self.lab_filenames = list(np.array(self.lab_filenames)[kept_indices])
                                    
            # Let's select the slices for which we create a HQ version
            np.random.seed(seed)
            self.hq_indices = np.random.choice(range(0, len(self.img_filenames)), int(np.ceil(len(self.img_filenames) * hq_percentage)), replace=False).astype(int)
                        
            if 'debiasing' in self.mode:
                # We only keep the slices selected to have HQ annotations
                self.img_filenames = list(np.array(self.img_filenames)[self.hq_indices])
                self.lab_filenames = list(np.array(self.lab_filenames)[self.hq_indices])
                # The hq indices will simply be now all the indices of the selected filenames
                self.hq_indices = np.arange(len(self.img_filenames))
        
        # Check if we drop cell labels
        if 'hl60_tiff' in root:
            self.n_labels = 20
        elif 'granulocytes_tiff' in root:
            self.n_labels = 15
        

        # Check if we use inclusion
        self.n_labels_extra_ds = 0    # We initialize the variable regardless (needed for below)
        if incl_p != 0 or 'inclusion' in root:
            np.random.seed(seed)

            if 'hl60' in root:
                self.n_labels = 10
                self.n_extra_labels = 8
                self.inclusion_labels_offset = 10    # granulocytes are numbered after hl60
                self.omission_labels_offset = 0      # hl60 are the first cells    
                self.inclusion_point = 10
            elif 'granulocytes' in root:
                self.n_labels = 8
                self.n_extra_labels = 10
                self.inclusion_labels_offset = 0    # hl60 are the first numbered labels
                self.omission_labels_offset = 10    # granulocytes are numbered after hl60
                self.inclusion_point = -10

            # If we do constant perturbations then we select the cell labels to be included at init time
            # Otherwise we keep the list empty and randomly select cell labels when we read a slice
            if self.constant_perturbations:
                self.included_labels = self.inclusion_labels_offset + np.random.choice(range(1, self.n_extra_labels + 1),
                                                             int(np.ceil(incl_p * self.n_extra_labels)), replace=False)
                # If we assume unclean initial data we need to make a larger inclusion set
                if 'unclean' in self.mode:
                    self.included_labels_unclean_debiasing = self.inclusion_labels_offset + np.random.choice(np.setdiff1d(np.arange(1, self.n_extra_labels + 1), self.included_labels),
                                                             int(np.ceil(incl_p_unclean_debiasing * (self.n_extra_labels - len(self.included_labels)))), replace=False)
                    self.included_labels_unclean_debiasing = np.union1d(self.included_labels, self.included_labels_unclean_debiasing)
            
        if drop_p != 0:
            np.random.seed(seed)
            # If we do constant perturbations then we select the cell labels to be dropped at init time
            # Otherwise we keep the list empty and randomly select cell labels when we read a slice
            if self.constant_perturbations:
                # Requires knowledge about the number of cells from the mask
                self.dropped_labels = self.omission_labels_offset + np.random.choice(range(1, self.n_labels+1),
                                                       int(drop_p * self.n_labels), replace=False)
                # If we assume unclean initial data we need to make a larger omission set
                if 'unclean' in self.mode:
                    self.dropped_labels_unclean_debiasing = np.random.choice(np.setdiff1d(np.arange(1, self.n_labels+1), self.dropped_labels), int(drop_p_unclean_debiasing * (self.n_labels - len(self.dropped_labels))), replace=False)
                    self.dropped_labels_unclean_debiasing = np.union1d(self.dropped_labels, self.dropped_labels_unclean_debiasing)

        # If we do constant perturbations then we select the morphological operations and iterations at init time
        # Otherwise we keep the lists empty and randomly select operations/iterations when we read a slice
        if max_iter > 0:
            if self.constant_perturbations:
                def get_operations_iterations_per_slice():
                    if self.bias_assumption == 'cell':
                        # Choose operations and iterations for all 30 volumes
                        operations_list = [np.random.choice(['d', 'e'], self.n_labels + self.n_labels_extra_ds)
                                                for _ in range(*file_range)]
                        iterations_list = [np.random.choice(range(1, max_iter + 1), self.n_labels + self.n_labels_extra_ds)
                                                for _ in range(*file_range)]
                    elif 'slice' in self.bias_assumption:
                        slices_list = []
                        operations_list = []
                        iterations_list = []
                        for tiff_volume in tiff_volumes_names:
                            vol_filenames, _ = SyntheticDatasetSelectionLQ.get_filenames(root, [tiff_volume])
                            volume_operations = []
                            volume_iterations = []
                            volume_slices = []
                            for vf in vol_filenames:
                                slice_no = int(vf.split('/')[-1].split('_')[2].split('.')[0])
                                # If we assume a bias per both slices and cells
                                if 'cell' in self.bias_assumption:
                                    operations_per_slice = np.random.choice(['d', 'e'], self.n_labels + self.n_labels_extra_ds)
                                    iterations_per_slice = np.random.choice(range(1, max_iter + 1),
                                                                            self.n_labels + self.n_labels_extra_ds)
                                else:
                                    operations_per_slice = np.random.choice(['d', 'e'])
                                    iterations_per_slice = np.random.choice(range(1, max_iter + 1))

                                volume_operations.append(operations_per_slice)
                                volume_iterations.append(iterations_per_slice)
                                volume_slices.append(slice_no)
                            operations_list.append(volume_operations)
                            iterations_list.append(volume_iterations)
                            slices_list.append(volume_slices)
                            
                        return operations_list, iterations_list, slices_list
                            
                np.random.seed(seed)
                self.operations_list, self.iterations_list, self.slices_list = get_operations_iterations_per_slice()
                if 'unclean' in self.mode:
                    self.operations_list_unclean_debiasing, self.iterations_list_unclean_debiasing, _ = get_operations_iterations_per_slice()

        dataset_size = len(self.img_filenames)

        if train:
            train_size = int(train_test_split * dataset_size)

            np.random.seed(seed + 1) # Otherwise we have the same selection as for the hq indices
            self.train_indices = np.random.choice(dataset_size, size=train_size, replace=False)
            self.val_indices = np.setdiff1d(np.arange(dataset_size), self.train_indices)

    def __getitem__(self, index):   
        # Debugging variable
        self.LQ_labels = []
        
        if torch.is_tensor(index):
            index = index.tolist()

        if self.train:
            if self.val:
                index = self.val_indices[index]
            else:
                index = self.train_indices[index]

        img_filename = self.img_filenames[index]
        lab_filename = self.lab_filenames[index]

        img = tiff.imread(img_filename)
        label = tiff.imread(lab_filename).astype(np.uint8)

        # We apply transforms if we have them
        if self.transform:
            img, label = self.transform(img, label)
        else:
            img = img / np.max(img)  # Scale the image

        volume_no = int(img_filename.split('/')[-1].split('_')[1])
        slice_no = int(img_filename.split('/')[-1].split('_')[2].split('.')[0])
        
        # Constant perturbations
        # We either prepare for segmentation with debiased model or we train for debiasing with unclean data or both
        if self.constant_perturbations:
            dropped_labels = self.dropped_labels
            included_labels = self.included_labels
            operations_list = self.operations_list
            iterations_list = self.iterations_list

            # We do 5 perturbation and majority vote the HQ label
            LQ_labels = []
            
            np.random.seed(self.seed)
            if len(dropped_labels) > 0:
                if index in self.hq_indices:
                    pass
                else:
                    # Normal perturbation to generate a LQ annotation
                    label = perform_omission(mask=label, omitted_labels=dropped_labels)
                
            if len(included_labels) > 0:
                if index in self.hq_indices:
                    pass
                else:
                    label = perform_inclusion(mask=label, inclusion_point=self.inclusion_point,
                                              included_labels=included_labels)
            # If we have an inclusion data set we need to keep only the main cell labels in case of no inclusion
            elif self.inclusion_point != 0:
                if self.inclusion_point > 0:
                    label[np.where(label > self.inclusion_point)] = 0
                else:
                    label[np.where(label <= abs(self.inclusion_point))] = 0
            if len(operations_list) > 0:
                if index in self.hq_indices:
                    pass
                else:
                    # We only assume a per-slice bias
                    adjusted_volume_index = volume_no - self.file_range[0]
                    slices = self.slices_list[adjusted_volume_index]
                    slice_index = slices.index(slice_no)
                    operations = operations_list[adjusted_volume_index][slice_index]
                    iterations = iterations_list[adjusted_volume_index][slice_index]

                    label = perform_bias(label, operation_iteration_list=list(zip([operations], [iterations])))
                

        # At the very end we do debiasing
        # Normal mode is the regular segmentation task
        if 'debiasing' in self.mode:
            np.random.seed(int(datetime.now().timestamp()))
            # We first keep track of the label we modify so that
            # we can chain our errors instead of applying them in isolation
            modified_label = label.copy()
            # Change the perturbations according to the chosen mode
            # if we assume a clean training data then the perturbations for debiasing are specified
            # in the main attributres,
            # otherwise we use the 'unclean' attributes to specify the perturbation we train for
            drop_p = self.drop_p_unclean_debiasing if 'unclean' in self.mode else self.drop_p
            if 'unclean' in self.mode:
                incl_p = self.incl_p_unclean_debiasing
                # We remove the already included masks from the copy, so we only do inclusion with the other masks
                if incl_p is not None and incl_p > 0:
                    for i_l in self.included_labels:
                        label_copy[label_copy == i_l] = 0
                    modified_label = label_copy
            else:
                incl_p = self.incl_p

            max_iter = self.max_iter_unclean_debiasing if 'unclean' in self.mode else self.max_iter

            # Omission
            if drop_p != 0:
                dropped_labels = self.omission_labels_offset + np.random.choice(range(1, self.n_labels+1),
                                                       int(drop_p * self.n_labels), replace=False)
                
                modified_label = perform_omission(modified_label, omitted_labels=dropped_labels)
            # Inclusion
            if incl_p != 0:
                modified_label = perform_inclusion(modified_label, inclusion_point=self.inclusion_point,
                                                   inclusion_rate=self.incl_p)
                if 'unclean' not in self.mode:
                    # We clean the label from unwanted additional cells
                    if self.inclusion_point > 0:
                        label[np.where(label > self.inclusion_point)] = 0
                    else:
                        label[np.where(label <= abs(self.inclusion_point))] = 0
                else:
                    # We now get the full inclusion
                    modified_label = label + modified_label
            # If we have an inclusion data set we need to keep only the main cell labels in case of no inclusion
            elif self.inclusion_point != 0:
                if self.inclusion_point > 0:
                    modified_label[np.where(modified_label > self.inclusion_point)] = 0
                    label[np.where(label > self.inclusion_point)] = 0
                else:
                    modified_label[np.where(modified_label <= abs(self.inclusion_point))] = 0
                    label[np.where(label <= abs(self.inclusion_point))] = 0
            # Bias
            if max_iter != 0:
                modified_label = perform_bias(modified_label, qmax=self.max_iter,
                                              bias_per_cell=self.bias_assumption == 'cell')

            if 'input' in self.mode:
                if len(modified_label.shape) > 2:
                    input_info = np.sum(modified_label, axis=0)
                else:
                    input_info = modified_label
                input_info[np.where(input_info > 1)] = 1
                img = np.stack([img, input_info], axis=0)
            elif 'full'in self.mode:
                label = modified_label
        
        if len(label.shape) > 2:
            label = np.sum(label, axis=0)
            
        # We clean the additional cells from the labels
        if 'debiasing' not in self.mode and self.incl_p == 0:
            if index in self.hq_indices:
                if self.inclusion_point:
                    if self.inclusion_point > 0:
                        label[np.where(label > self.inclusion_point)] = 0
                    else:
                        label[np.where(label <= abs(self.inclusion_point))] = 0
    
        if self.binary_mask:
            label[np.where(label > 1)] = 1

        img = torch.from_numpy(img.astype(float)).float()
        label = torch.from_numpy(label.astype(np.uint8)).long()

        if len(img.shape) == 2:
            img = img.unsqueeze(dim=0)

        if 'segmentation' in self.mode:
            if 'debiased' in self.mode and index not in self.hq_indices:
                assert self.debiasing_model is not None
                assert self.debiasing_device is not None
                if 'input' in self.mode:
                    input4debiasing = torch.stack([img.squeeze(), label])
                elif 'full' in self.mode:
                    input4debiasing = img.detach().clone()
                input4debiasing = input4debiasing.unsqueeze(dim=0).to(self.debiasing_device)
                debiasing_pred = self.debiasing_model(input4debiasing)
                _, debiasing_lab = torch.max(debiasing_pred.squeeze().detach(), dim=0)
                
                label = debiasing_lab.detach().cpu().long()

                
        # Debugging
        self.LQ_labels = LQ_labels
        return img, label

    def __len__(self):
        if self.train:
            if self.val:
                return len(self.val_indices)
            else:
                return len(self.train_indices)
        else:
            return len(self.img_filenames)




GP = {
    'hl60': '/data/vadineanus/data_label_error_study/hl60_tiff/',
    'granulocytes': '/data/vadineanus/data_label_error_study/granulocytes_tiff/',
    'epithelial_train': '/data/vadineanus/data_label_debiasing/train_real_padded/',
    'epithelial_test': '/data/vadineanus/data_label_debiasing/test_real_padded/',
    'epfl_train': '/data/vadineanus/data_label_debiasing/epfl/train/',
    'epfl_test': '/data/vadineanus/data_label_debiasing/epfl/test/',
    'inclusion_hl60': '/data/vadineanus/data_label_debiasing/inclusion_data_hl60/',
    'inclusion_granulocytes': '/data/vadineanus/data_label_debiasing/inclusion_data_granulocytes/',
    'inclusion_epithelial': '/data/vadineanus/data_label_debiasing/inclusion_real_data/',
    'models': '/data/vadineanus/data_label_debiasing/models/',
}


def transform(image, mask):
#     image = Image.fromarray(image)
#     mask = Image.fromarray(mask)
#     # Random crop
#     i, j, h, w = transforms.RandomCrop.get_params(
#         image, output_size=(512, 512))
#     image = tf.crop(image, i, j, h, w)
#     mask = tf.crop(mask, i, j, h, w)

    return np.array(image) / np.max(image), np.array(mask)


def transform_real(image, mask, output_size=(256, 256)):

    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)

    return np.array(image) / np.max(image), np.array(mask)


model_name = sys.argv[1]
batch_size = int(sys.argv[2])
crt = sys.argv[3]
cuda_no = int(sys.argv[4])
reps = str(sys.argv[5])

drop_p = float(sys.argv[6])
incl_p = float(sys.argv[7])
max_iter = int(sys.argv[8])

mode = str(sys.argv[9])
assumption = str(sys.argv[10])

n_volumes = int(sys.argv[11])
ds = str(sys.argv[12])

slices_per_volume = int(sys.argv[13])
total_slices = int(sys.argv[14])

if len(sys.argv) == 16:
    hq_percentage = float(sys.argv[15])
else:
    hq_percentage = 0
    
if len(sys.argv) == 18:
    drop_p_debiasing = float(sys.argv[15])
    incl_p_debiasing = float(sys.argv[16])
    max_iter_debiasing = int(sys.argv[17])
else:
    drop_p_debiasing = 0.0
    incl_p_debiasing = 0.0
    max_iter_debiasing = 0

if 'epithelial' in ds:
    ds_extension = 'epithelial'
elif 'hl60'in ds:
    ds_extension = 'hl60'
elif 'granulocytes' in ds:
    ds_extension = 'granulocytes'
elif 'epfl' in ds:
    ds_extension = 'epfl'

# We need to make a distinction between individual data sets and combined data sets if we do omision or bias
if 'inclusion' in ds and (drop_p != 0 or max_iter != 0):
    ds_extension = 'combined_' + ds_extension

if mode != 'segmentation_clean': 
    if mode == 'segmentation_unclean':
        extension = crt + f'_{ds_extension}_{mode}_{assumption}_{drop_p}_{incl_p}_{max_iter}'
    elif 'unclean' in mode:
        extension = crt + f'_{ds_extension}_{mode}_{assumption}_{drop_p}_{incl_p}_{max_iter}_v{n_volumes}_{drop_p_debiasing}_{incl_p_debiasing}_{max_iter_debiasing}'
    else:
        extension = crt + f'_{ds_extension}_{mode}_{assumption}_{drop_p}_{incl_p}_{max_iter}_v{n_volumes}'
else:
    extension = crt + f'_{ds}_{mode}_{assumption}_v{n_volumes}'

if 'input' in mode:
    n_channels = 2
    n_classes = 2
    mode_extension = 'input'
else:
    n_channels = 1
    n_classes = 2
    if 'full' in mode:
        mode_extension = 'full'

if 'epithelial' in ds:
    n_channels += 2

reps = list(map(int, reps.split(',')))

constant_perturbations = False
if 'segmentation' in mode:
    n_channels_deb = n_channels
    n_classes_deb = n_classes

    if (drop_p > 0 or incl_p > 0 or max_iter > 0) and '_clean' not in mode:
        constant_perturbations = True
elif 'unclean' in mode:
    constant_perturbations = True

# Setup description
print(f'Data: {ds}')
print(f'Omission: {drop_p}')
print(f'Inclusion: {incl_p}')
print(f'qmax: {max_iter}')

if drop_p_debiasing is not None or incl_p_debiasing is not None or max_iter_debiasing is not None:
    print(f'Omission debiasing: {drop_p_debiasing}')
    print(f'Inclusion debiasing: {incl_p_debiasing}')
    print(f'qmax debiasing: {max_iter_debiasing}')

print('\n')

extension = 'controlled_selection' + extension

for i in reps:
    crt_mode = mode
    print(f'Rep: {i}\n', flush=True)

    debiasing_model = None
    debiasing_device = None
    if 'debiased' in mode:
        debiasing_model = get_model(n_channels=n_channels_deb, n_classes=n_classes_deb, type='unet')
        n_channels = 1 if 'epithelial' not in ds else 3
        n_classes = 2
        debiasing_device = get_device(cuda_no)
        if 'unclean' in mode:
            path_to_load = f'unet_dice2_{ds_extension}_debiasing_{mode_extension}_unclean_slice_'\
                           f'{drop_p}_{incl_p}_{max_iter}_'\
                           f'v{n_volumes}_'\
                           f'{drop_p_debiasing}_{incl_p_debiasing}_{max_iter_debiasing}_'\
                           f'{i}_total_slices_{total_slices}_best.pt'
        else:
            path_to_load = f'unet_controlled_selectiondice2_{ds_extension}_debiasing_{mode_extension}_slice_'\
                           f'{drop_p}_{incl_p}_{max_iter}_'\
                           f'v{n_volumes}_{i}_total_slices_{total_slices}_hq_percentage_{hq_percentage}_best.pt'
            
        debiasing_model.load_state_dict(torch.load(GP['models'] + path_to_load, map_location=debiasing_device))
        debiasing_model = debiasing_model.to(debiasing_device)
    
    crt_extension = extension + '_' + str(i)
    
    if slices_per_volume > 0:
        crt_extension = crt_extension + f'_slices_per_volume_{slices_per_volume}'
    elif total_slices > 0:
        crt_extension = crt_extension + f'_total_slices_{total_slices}'

    crt_extension = crt_extension + f'_hq_percentage_{hq_percentage}'
     

            
    if ds in ['hl60', 'granulocytes', 'inclusion_hl60', 'inclusion_granulocytes']:
        train_data = SyntheticDatasetSelectionLQ(root=GP[ds],
                                      mode=crt_mode, train=True, val=False, file_range=(0, n_volumes), train_test_split=0.8,
                                      seed=i, transform=transform, bias_assumption=assumption,
                                      drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                      debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                                      slices_per_volume=slices_per_volume, total_slices=total_slices,
                                      constant_perturbations=True,
                                      drop_p_unclean_debiasing=drop_p_debiasing, 
                                      incl_p_unclean_debiasing=incl_p_debiasing,
                                      max_iter_unclean_debiasing=max_iter_debiasing,
                                      hq_percentage=hq_percentage)

        # Validation data should not vary
#         if 'debiased' in mode:
#             crt_mode = 'segmentation'
        val_data = SyntheticDatasetSelectionLQ(root=GP[ds],
                                    mode=crt_mode, train=True, val=True, file_range=(0, n_volumes),
                                    train_test_split=0.8,
                                    seed=i, transform=transform, bias_assumption=assumption,
                                    drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                    debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                                    slices_per_volume=slices_per_volume, total_slices=total_slices,
                                    constant_perturbations=True,
                                    drop_p_unclean_debiasing=drop_p_debiasing,
                                    incl_p_unclean_debiasing=incl_p_debiasing,
                                    max_iter_unclean_debiasing=max_iter_debiasing,
                                    hq_percentage=hq_percentage)

        
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
 

    model = get_model(n_channels=n_channels, n_classes=n_classes, type=model_name)
    model = model.to(get_device(cuda_no))

    torch.backends.cudnn.benchmark = True    # Speedup
    train(model=model, train_loader=train_loader,
          val_loader=val_loader, lr=0.001, epochs=None,
          save_extension=crt_extension, cuda_no=cuda_no, criterion=crt, save_path=GP['models'])
