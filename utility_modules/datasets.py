import torch
import numpy as np
import tifffile as tiff
from glob import glob
from utility_modules.utils import perform_omission, perform_inclusion, perform_bias
import math
import random
import pickle
import matplotlib.pyplot as plt
import imageio
import scipy.io as sio


class SyntheticDataset(torch.utils.data.Dataset):
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
                 drop_p_unclean_debiasing=0, incl_p_unclean_debiasing=0, max_iter_unclean_debiasing=0, total_slices_biased=0):

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
        self.clean_labels_indices = []
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
        # If we also sample the biased samples
        self.biased_labels_indices = []

        tiff_volumes_names = ['00' + f"{x:02d}" for x in range(*file_range)]
        self.img_filenames, self.lab_filenames = SyntheticDataset.get_filenames(root, tiff_volumes_names)

        # We select fewer slices from each volume
        if slices_per_volume > 0 or total_slices > 0 or total_slices_biased > 0:
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

                    
            # Save kept indices as an attribute
            self.clean_labels_indices = kept_indices  
            
            # The case where we want to also sample the biased slices
            if total_slices_biased > 0:
                slices_per_volume_biased = total_slices_biased // len(volumes)
                rem_slices_per_volume_biased = total_slices_biased % len(volumes)

                kept_indices_biased = []

                if slices_per_volume_biased > 0:
                    for v in volumes:
                        global_vol_indices = [vi[0] for vi in v]
                        # We make sure that we do not select a clean label slice
                        indices_to_choose = np.setdiff1d(global_vol_indices, self.clean_labels_indices)
                        # We add the randomly chosen indices to our list
                        kept_indices_biased += list(np.random.choice(indices_to_choose, slices_per_volume_biased, replace=False))

                if rem_slices_per_volume_biased > 0:
                    # For the remaining slices we randomly select from all volumes
                    global_indices = set([vi[0] for vi in v for v in volumes])
                    # We make sure that we do not select a clean label slice
                    indices_to_choose = np.setdiff1d(global_indices, self.clean_labels_indices)
                    # We make sure that we do not select previously selected inidices either
                    indices_to_choose = np.setdiff1d(indices_to_choose, kept_indices_biased)
                    # We randomly select the remaining slices
                    kept_indices_biased += list(np.random.choice(indices_to_choose, rem_slices_per_volume_biased, replace=False))

                self.biased_labels_indices = kept_indices_biased
            
            if 'debiasing' in self.mode or self.mode == 'segmentation_clean' or self.mode == 'segmentation_unclean':
                self.img_filenames = list(np.array(self.img_filenames)[kept_indices])
                self.lab_filenames = list(np.array(self.lab_filenames)[kept_indices])
            # If we want to do segmentation on biased labels with a fixed number of samples
            elif total_slices_biased > 0:
                self.img_filenames = list(np.array(self.img_filenames)[kept_indices]) + list(np.array(self.img_filenames)[kept_indices_biased])
                self.lab_filenames = list(np.array(self.lab_filenames)[kept_indices]) + list(np.array(self.lab_filenames)[kept_indices_biased])
        
                # Since we are not taking all the samples, we will not have the same indices as the global ones that we saved
                if len(kept_indices) > 0:
                    # The clean labels indices are always first
                    self.clean_labels_indices = np.arange(len(kept_indices))
        
        
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
                            vol_filenames, _ = SyntheticDataset.get_filenames(root, [tiff_volume])
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

            np.random.seed(seed)
            self.train_indices = np.random.choice(dataset_size, size=train_size, replace=False)
            self.val_indices = np.setdiff1d(np.arange(dataset_size), self.train_indices)

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        # If the index is among the images we keep intact we skip perturbations
        skip_perturbation = index in self.clean_labels_indices or self.mode == 'segmentation_clean' or self.mode == 'segmentation_unclean'

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
            # If we do not have perfect labels in the well-annotated set
            if 'unclean' in self.mode:
                # If the labels belong to the quickly annotated set
                if not skip_perturbation:
                    # If we train for debiasing then the well-annotated set errors come from the main variables
                    if 'debiasing' in self.mode:
                        dropped_labels = self.dropped_labels
                        included_labels = self.included_labels
                        operations_list = self.operations_list
                        iterations_list = self.iterations_list
                    # If we train for segmentation then the errors are for the quickly-annotated set here
                    elif 'segmentation' in self.mode:
                        dropped_labels = self.dropped_labels_unclean_debiasing
                        included_labels = self.included_labels_unclean_debiasing
                        operations_list = self.operations_list_unclean_debiasing
                        iterations_list = self.iterations_list_unclean_debiasing
                # If the labels belong to the better annotated set, but we assume errors in this set
                else:
                    dropped_labels = self.dropped_labels
                    included_labels = self.included_labels
                    operations_list = self.operations_list
                    iterations_list = self.iterations_list
            # If we have perfect labels for the well-annotated set
            else:
                if not skip_perturbation:
                    dropped_labels = self.dropped_labels
                    included_labels = self.included_labels
                    operations_list = self.operations_list
                    iterations_list = self.iterations_list
                else:
                    dropped_labels = []
                    included_labels = []
                    operations_list = []
                    iterations_list = []

            if len(dropped_labels) > 0:
                label = perform_omission(mask=label, omitted_labels=dropped_labels)
                
            if len(included_labels) > 0:
                if 'unclean' in self.mode:
                    label_copy = label.copy()    # We copy the label with all masks to perform debiasing inclusion
                label = perform_inclusion(mask=label, inclusion_point=self.inclusion_point,
                                          included_labels=included_labels)
            # If we have an inclusion data set we need to keep only the main cell labels in case of no inclusion
            elif self.inclusion_point != 0:
                if self.inclusion_point > 0:
                    label[np.where(label > self.inclusion_point)] = 0
                else:
                    label[np.where(label <= abs(self.inclusion_point))] = 0
            if len(operations_list) > 0:
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
                                                   inclusion_rate=incl_p)
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
                modified_label = perform_bias(modified_label, qmax=max_iter,
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
            
        
        # Remove the additional cells if we did not perform inclusion        
        if (skip_perturbation or self.incl_p == 0) and self.inclusion_point != 0:
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
            
        if 'segmentation' in self.mode and not skip_perturbation:
            if 'debiased' in self.mode:
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


class EPFLDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, train=True, val=False, train_test_split=0.7,
                 seed=42, binary_mask=True,
                 drop_p=0, incl_p=0, max_iter=0, transform=None, debiasing_model=None, debiasing_device=None,
                 bias_assumption='slice', constant_perturbations=False, total_slices=0,
                 drop_p_unclean_debiasing=0, incl_p_unclean_debiasing=0, max_iter_unclean_debiasing=0):

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
        self.seed = seed
        self.transform = transform
        self.max_iter = max_iter
        self.train_test_split = train_test_split
        self.debiasing_model = debiasing_model  # Already trained debiasing model
        self.debiasing_device = debiasing_device  # The cuda device on which we have the model
        self.bias_assumption = bias_assumption  # Can be 'cell', 'volume', 'slice'
        # This variable is true if we choose to add data set-wise perturbations that we can replicate
        # As in the MIDL 2022 paper
        # Otherwise the perturbations are chosen whenever an image is read
        self.constant_perturbations = constant_perturbations
        # We keep the path to the data
        self.root = root
        # We need the total number of slices as an attribute
        self.total_slices = total_slices
        
        if 'unclean' in mode:
            assert drop_p_unclean_debiasing is not None or incl_p_unclean_debiasing is not None or max_iter_unclean_debiasing is not None
            self.drop_p_unclean_debiasing = drop_p_unclean_debiasing
            self.incl_p_unclean_debiasing = incl_p_unclean_debiasing
            self.max_iter_unclean_debiasing = max_iter_unclean_debiasing

        if 'train' in root:
            file_pref = 'training'
        elif 'test' in root:
            file_pref = 'testing'

        # Read all images into memory
        self.images = tiff.imread(root + f'{file_pref}.tif')
        self.labels = tiff.imread(root + f'{file_pref}_groundtruth_labeled.tif')

        if self.incl_p > 0:
            self.inclusion_labels = tiff.imread(root + f'inclusion_epfl.tiff')

        self.clean_indices = []
        if total_slices > 0:
            # Smarter sampling
            volume_size = len(self.images)
            reminder = total_slices % volume_size
            indices = np.round(volume_size / (2 * reminder) + np.arange(total_slices) * volume_size / reminder).astype(int)
            if 'debiasing' in self.mode or 'segmentation_clean' in self.mode:
                # We only keep the images corresponding to the (not so) carefully-annotated images
                self.images = self.images[indices]
                self.labels = self.labels[indices]
            
            self.clean_indices = indices
                
        if train_test_split:
            dataset_size = len(self.images)
            train_size = int(train_test_split * dataset_size)

            np.random.seed(seed)
            self.train_indices = np.random.choice(dataset_size, size=train_size, replace=False)
            self.val_indices = np.setdiff1d(np.arange(dataset_size), self.train_indices)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.train:
            if self.val:
                index = self.val_indices[index]
            else:
                index = self.train_indices[index]

        # Take the images from memory
        img, label = self.images[index].copy(), self.labels[index].copy()
        
        # We apply transforms if we have them
        if self.transform:
            img, label = self.transform(img, label)
        else:
            img = img / np.max(img)  # Scale the image

        # If the index is among the images we keep intact we skip perturbations
        skip_perturbation = index in self.clean_indices or self.mode == 'segmentation_clean' or self.mode == 'segmentation_unclean'

        # To avoid the case when the seed is set because we do unclean HQ labels we create an extra seed before setting the 
        # "unclean" seed
        extra_seed = random.randrange(2**32 - 1)
        
        # Constant perturbations
        # We either prepare for segmentation with debiased model or we train for debiasing with unclean data or both
        if self.constant_perturbations:
            # If we do not have perfect labels in the well-annotated set
            if 'unclean' in self.mode:
                # If the labels belong to the quickly annotated set
                if not skip_perturbation:
                    # If we train for debiasing then the well-annotated set errors come from the main variables
                    if 'debiasing' in self.mode:
                        drop_p = self.drop_p
                        incl_p = self.incl_p
                        max_iter = self.max_iter
                    # If we train for segmentation then the errors are for the quickly-annotated set here
                    elif 'segmentation' in self.mode:
                        drop_p = self.drop_p_unclean_debiasing
                        incl_p = self.incl_p_unclean_debiasing
                        max_iter = self.max_iter_unclean_debiasing
                # If the labels belong to the better annotated set, but we assume errors in this set
                else:
                    drop_p = self.drop_p
                    incl_p = self.incl_p
                    max_iter = self.max_iter
            # If we have perfect labels for the well-annotated set
            else:
                if not skip_perturbation:
                    drop_p = self.drop_p
                    incl_p = self.incl_p
                    max_iter = self.max_iter
                else:
                    drop_p = 0
                    incl_p = 0
                    max_iter = 0
                
            if drop_p > 0:
                cell_labels = np.unique(label)
                label = perform_omission(label, omission_rate=drop_p, seed=self.seed, cell_labels=cell_labels)
            if incl_p > 0:
                inclusion_label = self.inclusion_labels[index].copy()
                # Perform omission on included label
                cell_labels = np.unique(inclusion_label)
                inclusion_label = perform_omission(inclusion_label, omission_rate=1-incl_p,
                                                   seed=self.seed, cell_labels=cell_labels)
                label = label + inclusion_label
            if max_iter > 0:
                label = perform_bias(label, qmax=max_iter, seed=self.seed)

        if 'debiasing' in self.mode:
            # We first keep track of the label we modify so that
            # we can chain our errors instead of applying them in isolation
            modified_label = label.copy()
            # Make sure to utilize the right variable in case we have unclean HQ data
            drop_p = self.drop_p if 'unclean' not in self.mode else self.drop_p_unclean_debiasing
            incl_p = self.incl_p if 'unclean' not in self.mode else self.incl_p_unclean_debiasing
            max_iter = self.max_iter if 'unclean' not in self.mode else self.max_iter_unclean_debiasing
                        
            # We extend the functionality to other types of errors
            # Omission
            if drop_p != 0:
                modified_label = perform_omission(modified_label, omission_rate=drop_p, seed=extra_seed)
            # Inclusion
            if incl_p != 0:
                
                if 'unclean' in self.mode:
                    inclusion_label_full = self.inclusion_labels[index].copy()
                    # We make sure that we do not include the cells we have already included
                    inclusion_label = inclusion_label_full - inclusion_label
                else:
                    inclusion_label = self.inclusion_labels[index].copy()

                inclusion_label = perform_omission(inclusion_label, omission_rate=1-incl_p, seed=extra_seed)
                modified_label = modified_label + inclusion_label
            # Bias
            if max_iter > 0:
                modified_label = perform_bias(modified_label, qmax=max_iter,
                                              bias_per_cell=self.bias_assumption == 'cell', seed=extra_seed)

            if 'input' in self.mode:
                if len(modified_label.shape) > 2:
                    input_info = np.sum(modified_label, axis=0)
                else:
                    input_info = modified_label
                input_info[np.where(input_info > 1)] = 1
                img = np.stack([img, input_info], axis=0)
            elif 'full' in self.mode:
                label = modified_label

        if 'testing' in self.mode:
            copy_label = label.copy()

        if len(label.shape) > 2:
            label = np.sum(label, axis=0)
        label[np.where(label > 1)] = 1

        img = torch.from_numpy(img.astype(float)).float()
        label = torch.from_numpy(label.astype(np.uint8)).long()

        if len(img.shape) == 2:
            img = img.unsqueeze(dim=0)

        if 'segmentation' in self.mode and not skip_perturbation:
            if 'debiased' in self.mode:
                assert self.debiasing_model is not None
                assert self.debiasing_device is not None
                if 'input' in self.mode:
                    input4debiasing = torch.zeros((img.shape[0] + 1, img.shape[1], img.shape[2]))
                    input4debiasing[:img.shape[0], :, :] = img
                    input4debiasing[img.shape[0], :, :] = label
                elif 'full' in self.mode:
                    input4debiasing = img.detach().clone()
                input4debiasing = input4debiasing.unsqueeze(dim=0).to(self.debiasing_device)
                debiasing_pred = self.debiasing_model(input4debiasing)
                _, debiasing_lab = torch.max(debiasing_pred.squeeze().detach(), dim=0)
                label = debiasing_lab.detach().cpu().long()

        if 'testing' in self.mode:
            copy_label[np.where(copy_label > 1)] = 1
            label = (label, torch.from_numpy(copy_label.astype(np.uint8)).long())

        return img, label

    def __len__(self):
        if self.train:
            if self.val:
                return len(self.val_indices)
            else:
                return len(self.train_indices)
        else:
            return len(self.patches)


class LizardDataset(torch.utils.data.Dataset):
    @staticmethod
    def get_mask(filename, cls=True, ids=True):
        label = sio.loadmat(filename)

        if cls:
            if ids:
                return label['inst_map'], label['class'], label['id']
            else:
                return label['inst_map'], label['class']
        else:
            return label['inst_map']
    
    def __init__(self, root, path_full_labels, mode, train=True, val=False, train_test_split=0.7,
                 seed=42, binary_mask=True,
                 drop_p=0, incl_p=0, max_iter=0, transform=None, debiasing_model=None, debiasing_device=None,
                 bias_assumption='slice', constant_perturbations=False, total_slices=0,
                 drop_p_unclean_debiasing=0, incl_p_unclean_debiasing=0, max_iter_unclean_debiasing=0):

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
        self.seed = seed
        self.transform = transform
        self.max_iter = max_iter
        self.train_test_split = train_test_split
        self.debiasing_model = debiasing_model  # Already trained debiasing model
        self.debiasing_device = debiasing_device  # The cuda device on which we have the model
        self.bias_assumption = bias_assumption  # Can be 'cell', 'volume', 'slice'
        # This variable is true if we choose to add data set-wise perturbations that we can replicate
        # As in the MIDL 2022 paper
        # Otherwise the perturbations are chosen whenever an image is read
        self.constant_perturbations = constant_perturbations
        # We keep the path to the data
        self.root = root
        # We need the total number of slices as an attribute
        self.total_slices = total_slices
        self.path_full_labels = path_full_labels
            
        if 'unclean' in mode:
            assert drop_p_unclean_debiasing is not None or incl_p_unclean_debiasing is not None or max_iter_unclean_debiasing is not None
            self.drop_p_unclean_debiasing = drop_p_unclean_debiasing
            self.incl_p_unclean_debiasing = incl_p_unclean_debiasing
            self.max_iter_unclean_debiasing = max_iter_unclean_debiasing

        if 'train' in root:
            file_pref = 'training'
        elif 'test' in root:
            file_pref = 'testing'

        # Get filenames
        self.img_filenames = np.array(sorted(glob(root + f'*-final.png'), key=lambda x: x.split('/')[-1].split('-')[0]))
        self.label_filenames = np.array(sorted(glob(root + f'*-label.tiff'), key=lambda x: x.split('/')[-1].split('-')[0]))
        
        self.clean_indices = []
        if total_slices > 0:
            np.random.seed(self.seed)
            indices = np.random.choice(len(self.img_filenames), size=total_slices, replace=False)
            self.img_filenames = self.img_filenames[indices]
            self.label_filenames = self.label_filenames[indices]
            
            self.clean_indices = indices
                
        if train_test_split:
            dataset_size = len(self.img_filenames)
            train_size = int(train_test_split * dataset_size)

            np.random.seed(seed)
            self.train_indices = np.random.choice(dataset_size, size=train_size, replace=False)
            self.val_indices = np.setdiff1d(np.arange(dataset_size), self.train_indices)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.train:
            if self.val:
                index = self.val_indices[index]
            else:
                index = self.train_indices[index]

        # Read the image and the label patches
        img = imageio.imread(self.img_filenames[index])
        label = imageio.imread(self.label_filenames[index])
        
        # We apply transforms if we have them
        if self.transform:
            img, label = self.transform(img, label)
        else:
            img = img / np.max(img)  # Scale the image
        img = np.transpose(img, (2, 0, 1))

        
        # We also need to read the label metadata 
        # Put together the label filename
        file_id = '_'.join(self.label_filenames[index].split('/')[-1].split('_')[:2])
        _, cls, ids = LizardDataset.get_mask(self.path_full_labels + f'{file_id}.mat')
        # Separate the ids
        full_target_cells = ids[np.where(cls == 2)]
        full_other_cells = ids[np.where(cls != 2)]
        patch_target_cells = np.intersect1d(np.unique(label), np.array(full_target_cells))
        patch_other_cells = np.intersect1d(np.unique(label), np.array(full_other_cells))
        # Separate the label into label with target cells only
        # and inclusion label only with other cells
        inclusion_label = label.copy()
        inclusion_label[np.where(np.isin(inclusion_label, patch_target_cells))] = 0
        # We also need to keep a clean copy of the inclusion label for the case when we 
        # include twice like for the unclean HQ data
        inclusion_label_full = inclusion_label.copy()
        label[np.where(np.isin(label, patch_other_cells))] = 0
        
        
        # If the index is among the images we keep intact we skip perturbations
        skip_perturbation = index in self.clean_indices or self.mode == 'segmentation_clean' or self.mode == 'segmentation_unclean'

        # To avoid the case when the seed is set because we do unclean HQ labels we create an extra seed before setting the 
        # "unclean" seed
        extra_seed = random.randrange(2**32 - 1)
        
        # Constant perturbations
        # We either prepare for segmentation with debiased model or we train for debiasing with unclean data or both
        if self.constant_perturbations:
            # If we do not have perfect labels in the well-annotated set
            if 'unclean' in self.mode:
                # If the labels belong to the quickly annotated set
                if not skip_perturbation:
                    # If we train for debiasing then the well-annotated set errors come from the main variables
                    if 'debiasing' in self.mode:
                        drop_p = self.drop_p
                        incl_p = self.incl_p
                        max_iter = self.max_iter
                    # If we train for segmentation then the errors are for the quickly-annotated set here
                    elif 'segmentation' in self.mode:
                        drop_p = self.drop_p_unclean_debiasing
                        incl_p = self.incl_p_unclean_debiasing
                        max_iter = self.max_iter_unclean_debiasing
                # If the labels belong to the better annotated set, but we assume errors in this set
                else:
                    drop_p = self.drop_p
                    incl_p = self.incl_p
                    max_iter = self.max_iter
            # If we have perfect labels for the well-annotated set
            else:
                if not skip_perturbation:
                    drop_p = self.drop_p
                    incl_p = self.incl_p
                    max_iter = self.max_iter
                else:
                    drop_p = 0
                    incl_p = 0
                    max_iter = 0
                
            if drop_p > 0:
                cell_labels = patch_target_cells
                label = perform_omission(label, omission_rate=drop_p, seed=self.seed, cell_labels=cell_labels)
            if incl_p > 0:
                cell_labels = patch_other_cells
                inclusion_label = perform_omission(inclusion_label, omission_rate=1-incl_p,
                                                   seed=self.seed, cell_labels=cell_labels)
                label = label + inclusion_label
            if max_iter > 0:
                label = perform_bias(label, qmax=max_iter, seed=self.seed, kernel_size=2)

        if 'debiasing' in self.mode:
            # We first keep track of the label we modify so that
            # we can chain our errors instead of applying them in isolation
            modified_label = label.copy()
            # Make sure to utilize the right variable in case we have unclean HQ data
            drop_p = self.drop_p if 'unclean' not in self.mode else self.drop_p_unclean_debiasing
            incl_p = self.incl_p if 'unclean' not in self.mode else self.incl_p_unclean_debiasing
            max_iter = self.max_iter if 'unclean' not in self.mode else self.max_iter_unclean_debiasing
                        
            # We extend the functionality to other types of errors
            # Omission
            if drop_p != 0:
                modified_label = perform_omission(modified_label, omission_rate=drop_p, seed=extra_seed)
            # Inclusion
            if incl_p != 0:
                
                if 'unclean' in self.mode:
                    # We make sure that we do not include the cells we have already included
                    inclusion_label = inclusion_label_full - inclusion_label

                inclusion_label = perform_omission(inclusion_label, omission_rate=1-incl_p, seed=extra_seed)
                modified_label = modified_label + inclusion_label
            # Bias
            if max_iter > 0:
                modified_label = perform_bias(modified_label, qmax=max_iter,
                                              bias_per_cell=self.bias_assumption == 'cell', seed=extra_seed, kernel_size=2)
                
            if 'input' in self.mode:
                if len(modified_label.shape) > 2:
                    input_info = np.sum(modified_label, axis=0)
                else:
                    input_info = modified_label
                input_info[np.where(input_info > 1)] = 1
                input_info = np.expand_dims(input_info, axis=0)
                img = np.vstack([img, input_info])
            elif 'full' in self.mode:
                label = modified_label

        if 'testing' in self.mode:
            copy_label = label.copy()

        if len(label.shape) > 2:
            label = np.sum(label, axis=0)
        label[np.where(label > 1)] = 1
        
        img = torch.from_numpy(img.astype(float)).float()
        label = torch.from_numpy(label.astype(np.uint8)).long()

        if len(img.shape) == 2:
            img = img.unsqueeze(dim=0)

        if 'segmentation' in self.mode and not skip_perturbation:
            if 'debiased' in self.mode:
                assert self.debiasing_model is not None
                assert self.debiasing_device is not None
                if 'input' in self.mode:
                    input4debiasing = torch.zeros((img.shape[0] + 1, img.shape[1], img.shape[2]))
                    input4debiasing[:img.shape[0], :, :] = img
                    input4debiasing[img.shape[0], :, :] = label
                elif 'full' in self.mode:
                    input4debiasing = img.detach().clone()
                input4debiasing = input4debiasing.unsqueeze(dim=0).to(self.debiasing_device)
                debiasing_pred = self.debiasing_model(input4debiasing)
                _, debiasing_lab = torch.max(debiasing_pred.squeeze().detach(), dim=0)
                label = debiasing_lab.detach().cpu().long()

        if 'testing' in self.mode:
            copy_label[np.where(copy_label > 1)] = 1
            label = (label, torch.from_numpy(copy_label.astype(np.uint8)).long())

        return img, label

    def __len__(self):
        if self.train:
            if self.val:
                return len(self.val_indices)
            else:
                return len(self.train_indices)
        else:
            return len(self.patches)
