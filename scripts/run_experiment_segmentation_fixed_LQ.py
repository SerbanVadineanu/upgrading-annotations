import sys
from torchvision import transforms
from utility_modules.utils import train, get_model, get_device
from PIL import Image
import torchvision.transforms.functional as tf
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
from utility_modules.datasets import SyntheticDataset

torch.autograd.set_detect_anomaly(False)    # Speedup
torch.autograd.profiler.profile(False)      # Speedup
torch.autograd.profiler.emit_nvtx(False)    # Speedup

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
#         if 'debiasing' not in self.mode and self.incl_p == 0:
        if False:
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
    'epfl_test_improved': '/data/vadineanus/data_label_debiasing/epfl/improved_test/',
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
    if mask.shape[0] < output_size[0] or mask.shape[1] < output_size[1]:
        image_new = np.zeros((max(mask.shape[0], output_size[0]), max(mask.shape[1], output_size[1]), 3)).astype(
            np.uint8)
        mask_new = np.zeros((max(mask.shape[0], output_size[0]), max(mask.shape[1], output_size[1]))).astype(np.uint8)
        idx_start_x = max(0, output_size[0] - mask.shape[0])
        idx_start_y = max(0, output_size[1] - mask.shape[1])

        image_new[idx_start_x:, idx_start_y:, :] = image
        mask_new[idx_start_x:, idx_start_y:] = mask

        image = image_new
        mask = mask_new

    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=output_size)
    image = tf.crop(image, i, j, h, w)
    mask = tf.crop(mask, i, j, h, w)

    return np.array(image) / np.max(image), np.array(mask)


def transform_real_test(image, mask, output_size=(256, 256)):

    return np.array(image) / np.max(image), np.array(mask)


def get_dice(predict, target, return_dice=True):
    tp = torch.sum(predict * target)
    fp = torch.sum(predict) - tp
    fn = torch.sum(target) - tp

    if return_dice:
        return 2 * tp / (2 * tp + fp + fn)
    else:
        return tp, fp, fn


def center_transform(image, mask):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    image = tf.center_crop(image, output_size=(512, 512))
    mask = tf.center_crop(mask, output_size=(512, 512))

    return np.array(image) / np.max(image), np.array(mask)


def load_segmentation_model(drop_p, incl_p, max_iter, rep, n_volumes, dataset='hl60', segmentation_type='segmentation_clean',
                            cuda_no=0, total_slices=0, loss='dice2',
                            drop_p_debiasing=0, incl_p_debiasing=0, max_iter_debiasing=0, hq_percentage=0.0):
    n_channels = 1 if 'epithelial' not in dataset else 3
    n_classes = 2
    
    if 'inclusion' in dataset and 'clean' not in segmentation_type:
        dataset = dataset.split('_')[1]
        if drop_p != 0 or max_iter != 0:
            dataset = 'combined_' + dataset

    if '_clean' in segmentation_type:
        setup_extension = ''
    else:
        setup_extension = f'{drop_p}_{incl_p}_{max_iter}_'
        
    if 'unclean_debiased' in segmentation_type:
        debiasing_setup_extension = f'{drop_p_debiasing}_{incl_p_debiasing}_{max_iter_debiasing}_'
    else:
        debiasing_setup_extension = ''
    
    if total_slices > 0:
        extra_extension = f'_total_slices_{total_slices}'
    else:
        extra_extension = ''
        
    # For this specific experiment we will always have hq_percentage
    extra_extension = extra_extension + f'_hq_percentage_{hq_percentage}'
    
    if segmentation_type == 'segmentation_unclean':
        volume_extension = ''
    else:
        volume_extension = f'v{n_volumes}_'

        
    path = GP['models'] + f'unet_{loss}_{dataset}_{segmentation_type}_slice_' \
                                       f'{setup_extension}' \
                                       f'{volume_extension}{debiasing_setup_extension}{rep}{extra_extension}_best.pt'
                          
    segmentation_model = get_model(n_channels=n_channels, n_classes=n_classes, type='unet')
    segmentation_model.load_state_dict(torch.load(GP['models'] +
                                       f'unet_{loss}_{dataset}_{segmentation_type}_slice_'
                                       f'{setup_extension}'
                                       f'{volume_extension}{debiasing_setup_extension}{rep}{extra_extension}_best.pt',
                                       map_location=get_device(cuda_no=cuda_no)))

    segmentation_model = segmentation_model.to(get_device(cuda_no=cuda_no))

    return segmentation_model



drop_p = float(sys.argv[1])
incl_p = float(sys.argv[2])
max_iter = int(sys.argv[3])
dataset = sys.argv[4]

cuda_no = int(sys.argv[5])

n_volumes = 10

columns = ['dataset', 'omission', 'inclusion', 'qmax', 'total_slices', 'hq_percentage', 'rep', 'dice']

rows = []

# for total_slices in [10, 25, 50, 100, 250, 500]:
for total_slices in [10, 25, 50, 100]:
    for hq_percentage in [0.0, 0.1, 0.2, 0.3]:
#     for hq_percentage in [1.0]:
        if 0.1 <= hq_percentage < 1.0:    # Model trained only on HQ slices
            mode = 'segmentation_debiased_input'
            if total_slices == 10:
                continue
        elif hq_percentage == 1.0:
            mode = 'segmentation'
            if total_slices > 100:
                continue
        else: 
            mode = 'segmentation'
        
        if hq_percentage == 1.0:
            crt_drop_p = 0.0
            crt_incl_p = 0.0
            crt_max_iter = 0
        else:
            crt_drop_p = drop_p
            crt_incl_p = incl_p
            crt_max_iter = max_iter
            
        for rep in [0, 1, 2, 3, 4]:
    #     for rep in [0]:


            segmentation_model = load_segmentation_model(drop_p=crt_drop_p, incl_p=crt_incl_p, max_iter=crt_max_iter,
                                                         rep=rep, n_volumes=n_volumes, cuda_no=cuda_no,
                                                         segmentation_type=mode,
                                                         dataset=dataset, total_slices=total_slices, 
                                                         loss='controlled_selectiondice2', hq_percentage=hq_percentage)

            start_time = time.time()
            print(rep)

            data_gt = SyntheticDataset(root=GP[dataset],
                                       mode='normal', train=True, val=False, train_test_split=1.0,
                                       seed=rep, file_range=(25, 30), binary_mask=True, drop_p=0,
                                       incl_p=0, max_iter=0, transform=transform, debiasing_model=None,
                                       debiasing_device=None,
                                       bias_assumption='slice', constant_perturbations=True)


            tp, fp, fn = 0, 0, 0

            for i in range(len(data_gt)):
                img, lab_gt = data_gt[i]


                _, pred = torch.max(segmentation_model(img.unsqueeze(dim=0).to(get_device(cuda_no=cuda_no))).squeeze().detach(), dim=0)

                tp_, fp_, fn_ = get_dice(pred.detach().cpu(), lab_gt, return_dice=False)

                tp += tp_
                fp += fp_
                fn += fn_

            dice = 2 * tp / (2 * tp + fp + fn)

            row = [dataset, drop_p, incl_p, max_iter, total_slices, hq_percentage, rep, dice.item()]
            rows.append(row)
            
            print(row, flush=True)
            print(f'It took: {np.round(time.time() - start_time, 3)} s', flush=True)

df = pd.DataFrame(rows, columns=columns)

df.to_csv(f'results_segmentation_fixed_LQ_unet_{dataset}_{drop_p}_{incl_p}_{max_iter}.csv')