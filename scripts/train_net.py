import sys
from torchvision import transforms
from torch.utils.data import DataLoader
from utility_modules.utils import train, get_model, get_device
from PIL import Image
import torchvision.transforms.functional as tf
from utility_modules.datasets import SyntheticDataset, RealDataset, EPFLDataset
import torch
import numpy as np
import random

torch.autograd.set_detect_anomaly(False)    # Speedup
torch.autograd.profiler.profile(False)      # Speedup
torch.autograd.profiler.emit_nvtx(False)    # Speedup
torch.set_flush_denormal(True)              # Added this to try to fix training slowing down. It apporximates very small numbers to 0

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
    total_slices_biased = int(sys.argv[15])
else:
    total_slices_biased = 0
    
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
            path_to_load = f'unet_dice2_{ds_extension}_debiasing_{mode_extension}_slice_'\
                           f'{drop_p}_{incl_p}_{max_iter}_'\
                           f'v{n_volumes}_{i}_total_slices_{total_slices}_best.pt'
            
        debiasing_model.load_state_dict(torch.load(GP['models'] + path_to_load, map_location=debiasing_device))
        debiasing_model = debiasing_model.to(debiasing_device)
    
    crt_extension = extension + '_' + str(i)
    
    if slices_per_volume > 0:
        crt_extension = crt_extension + f'_slices_per_volume_{slices_per_volume}'
    elif total_slices > 0:
        crt_extension = crt_extension + f'_total_slices_{total_slices}'
    if total_slices_biased > 0:
        crt_extension = crt_extension + f'_biased_slices_{total_slices_biased}'
     

    if ds in ['hl60', 'granulocytes', 'inclusion_hl60', 'inclusion_granulocytes']:
        train_data = SyntheticDataset(root=GP[ds],
                                      mode=crt_mode, train=True, val=False, file_range=(0, n_volumes), train_test_split=0.8,
                                      seed=i, transform=transform, bias_assumption=assumption,
                                      drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                      debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                                      slices_per_volume=slices_per_volume, total_slices=total_slices,
                                      constant_perturbations=constant_perturbations,
                                      drop_p_unclean_debiasing=drop_p_debiasing, 
                                      incl_p_unclean_debiasing=incl_p_debiasing,
                                      max_iter_unclean_debiasing=max_iter_debiasing,
                                      total_slices_biased=total_slices_biased)

        # Validation data should not vary
        if 'debiased' in mode:
            crt_mode = 'segmentation'
        val_data = SyntheticDataset(root=GP[ds],
                                    mode=crt_mode, train=True, val=True, file_range=(0, n_volumes),
                                    train_test_split=0.8,
                                    seed=i, transform=transform, bias_assumption=assumption,
                                    drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                    debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                                    slices_per_volume=slices_per_volume, total_slices=total_slices,
                                    constant_perturbations=constant_perturbations,
                                    drop_p_unclean_debiasing=drop_p_debiasing,
                                    incl_p_unclean_debiasing=incl_p_debiasing,
                                    max_iter_unclean_debiasing=max_iter_debiasing,
                                    total_slices_biased=total_slices_biased)
    
    elif 'epfl' in ds:
        train_data = EPFLDataset(root=GP[ds],
                                 mode=crt_mode, train=True, val=False, train_test_split=0.8,
                                 seed=i, transform=transform_real, bias_assumption=assumption,
                                 drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                 debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                                 total_slices=total_slices, constant_perturbations=constant_perturbations,
                                 drop_p_unclean_debiasing=drop_p_debiasing,
                                 incl_p_unclean_debiasing=incl_p_debiasing,
                                 max_iter_unclean_debiasing=max_iter_debiasing)

        # Validation data should not vary
        if 'debiased' in mode:
            crt_mode = 'segmentation'
        val_data = EPFLDataset(root=GP[ds],
                               mode=crt_mode, train=True, val=True, train_test_split=0.8,
                               seed=i, transform=None, bias_assumption=assumption,
                               drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                               debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                               total_slices=total_slices, constant_perturbations=constant_perturbations,
                               drop_p_unclean_debiasing=drop_p_debiasing,
                               incl_p_unclean_debiasing=incl_p_debiasing,
                               max_iter_unclean_debiasing=max_iter_debiasing)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
 

    model = get_model(n_channels=n_channels, n_classes=n_classes, type=model_name)
    model = model.to(get_device(cuda_no))

    torch.backends.cudnn.benchmark = True    # Speedup
    train(model=model, train_loader=train_loader,
          val_loader=val_loader, lr=0.001, epochs=None,
          save_extension=crt_extension, cuda_no=cuda_no, criterion=crt, save_path=GP['models'])
