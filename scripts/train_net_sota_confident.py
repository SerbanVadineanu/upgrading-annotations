import sys
from torchvision import transforms
from torch.utils.data import DataLoader
from utility_modules.utils import train, get_model, get_device
from PIL import Image
import torchvision.transforms.functional as tf
from utility_modules.datasets import RealDataset, EPFLDataset, SyntheticDataset
import torch
import numpy as np
import random
import torch.nn as nn
import torch
import numpy as np
import random
from utility_modules.network_architectures.unet import UNet, UNetLabelPass
from utility_modules.network_architectures.segnet import SegNet
from utility_modules.losses import CrossEntropy2d, DiceLoss, DiceLoss2
import msd_pytorch as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
import cv2
import itertools
from skimage.util import map_array
import torch
import numpy as np
import tifffile as tiff
from glob import glob
from utility_modules.utils import perform_omission, perform_inclusion, perform_bias
import math
import random
import pickle
import matplotlib.pyplot as plt

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


def load_segmentation_model(drop_p, incl_p, max_iter, rep, n_volumes, dataset='hl60', segmentation_type='segmentation_clean',
                            cuda_no=0, total_slices=0, loss='dice2',
                            drop_p_debiasing=0, incl_p_debiasing=0, max_iter_debiasing=0):
    n_channels = 1 if 'epithelial' not in dataset else 3
    n_classes = 2
    
    if 'inclusion' in dataset:
        dataset = dataset.split('_')[1]

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



def confident_label_generation(output, label, confidence=0.8):
    # Threshold calculation
    t_bg = np.float64(0)
    t_fg = np.float64(0)

    t_bg = torch.mean(output[0][label == 0].detach())
    t_fg = torch.mean(output[1][label == 1].detach())
    
    # Confidence matrix
    C = np.zeros((output.shape[0], output.shape[0]))
    p = [t_bg, t_fg]
    
    label = label.cpu()
        
    for i in range(len(C)):
        for j in range(len(C[0])):
            C[i, j] = len(np.where((label == i) & (output[j].cpu() >= p[j].cpu()))[0])

    # Adjusted confidence matrix
    for i in range(len(C)):
        for j in range(len(C[0])):
            C[i, j] = C[i, j] / sum(C[i, :]) * len(np.where(label == i)[0])       
    
    C = np.nan_to_num(C)
    
    # Confidence distribution matrix
    Q = C / np.sum(C)
    
    # Total pixel count
    n = label.shape[-2] * label.shape[-1]
            
    # Corrected labels foreground
    i, j = 0, 1

    n_low_confidence = int(n * Q[i, j])
    lab_err_foreground = np.zeros((label.shape[-2], label.shape[-1]))

    # Select the lowest confidence pixels
    indices = np.where(label == i)
    sorted_low_conf_indices = torch.argsort(output[i][indices])[:n_low_confidence].cpu()
    low_conf_indices = (indices[0][sorted_low_conf_indices], indices[1][sorted_low_conf_indices])

    lab_err_foreground[low_conf_indices] = confidence
            
    
    # Corrected labels background
    i, j = 1, 0

    n_low_confidence = int(n * Q[i, j])
    lab_err_background = np.zeros((label.shape[-2], label.shape[-1]))

    # Select the lowest confidence pixels
    indices = np.where(label == i)
    sorted_low_conf_indices = torch.argsort(output[i][indices])[:n_low_confidence].cpu()
    low_conf_indices = (indices[0][sorted_low_conf_indices], indices[1][sorted_low_conf_indices])

    lab_err_background[low_conf_indices] = confidence
    
    
    confident_label = label - lab_err_background + lab_err_foreground
    confident_label[confident_label < 0] = 0
    
    return confident_label


# BIG assumption: only one image per batch!!
# The implementation does not work otherwise
def train(model, train_loader, val_loader, teacher_model, epochs=50, lr=0.01,
          optimizer='adam', criterion='crossentropy', save_extension='',
          cuda_no=0, seed=42, save_path=''):
    # Set the seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = get_device(cuda_no)
    print(device, flush=True)

    train_history_loss = []
    val_history_loss = []

    if isinstance(model, SegNet):
        model_name = 'segnet'
    elif isinstance(model, UNet):
        model_name = 'unet'
    elif isinstance(model, UNetLabelPass):
        model_name = 'unet_label_pass'
    else:
        model_name = 'msd'
        model.set_normalization(train_loader)

    print(f'Training {model_name}\n', flush=True)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    if criterion == 'crossentropy':
        criterion = CrossEntropy2d()
#         criterion = nn.CrossEntropyLoss()
    elif criterion == 'dice':
        criterion = DiceLoss(smooth=0)
    elif criterion == 'dice2':
        criterion = DiceLoss2()

    min_val_loss = math.inf

    # If the number of epochs is not specified
    # We count to infinity
    if epochs is not None:
        iterator = range(epochs)
    else:
        iterator = itertools.count(start=0, step=1)

    for i in iterator:
        start = time.time()
        train_crt_loss = 0.0
        val_crt_loss = 0.0
        train_dice_loss = 0.0

        for step in ['train', 'val']:
            if step == 'train':
                loader = train_loader
            else:
                if val_loader is None:
                    break
                else:
                    loader = val_loader
                    
            for batch, labels in loader:
                batch = batch.to(device)
                labels = labels.to(device)

                # Here we process the label for the student model
                with torch.no_grad():
                    teacher_output = nn.Softmax2d()(teacher_model(batch)).squeeze()
                 
                confident_label = confident_label_generation(teacher_output.squeeze(dim=0), labels.squeeze(dim=0), confidence=0.8)
                confident_label = confident_label.unsqueeze(dim=0)
                confident_label = confident_label.to(device)
            
                output = model(batch)
                loss = criterion(output, confident_label)

                if step == 'train':
                    # optimizer.zero_grad()
                    # Should speed things up a bit
                    for param in model.parameters():    # Speedup
                        param.grad = None    # Speedup

                    loss.backward()
                    optimizer.step()

                    train_crt_loss += loss.item()
                    train_dice_loss += 0
                else:
                    val_crt_loss += loss.item()

        print('Epoch {:} took {:.4f} seconds'.format(i, time.time() - start), flush=True)
        t_l = train_crt_loss / len(train_loader)
        v_l = val_crt_loss / len(val_loader)
        print('Train loss: {:.4f}'.format(t_l), flush=True)
        print('Valid loss: {:.4f}'.format(v_l), flush=True)

        if i == 0:
            # We initialize a counter for lack of improvement in validation score
            no_improvement_count = 0
        elif v_l < min_val_loss:
            # Improvement? Then we reset the count
            no_improvement_count = 0
            min_val_loss = v_l
            torch.save(model.state_dict(), f'{save_path}{model_name}_{save_extension}_best.pt')
        else:
            # If we do not have a fixed number of epochs
            # and there is no improvement for 10 consecutive epochs
            if no_improvement_count == 10 and epochs is None:
                print('End of training!', flush=True)
                return
            # We increase the number of epochs with no improvement
            no_improvement_count += 1



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

if len(sys.argv) > 15:
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

extension = 'SOTA_confident_' + extension 

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
                                      max_iter_unclean_debiasing=max_iter_debiasing)

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
                                    max_iter_unclean_debiasing=max_iter_debiasing)
    
    elif 'epfl' in ds:
        train_data = EPFLDataset(root=GP[ds],
                                 mode=crt_mode, train=True, val=False, train_test_split=0.8,
                                 seed=i, transform=transform_real, bias_assumption=assumption,
                                 drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                 debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                                 total_slices=total_slices, constant_perturbations=constant_perturbations)

        # Validation data should not vary
        if 'debiased' in mode:
            crt_mode = 'segmentation'
        val_data = EPFLDataset(root=GP[ds],
                               mode=crt_mode, train=True, val=True, train_test_split=0.8,
                               seed=i, transform=None, bias_assumption=assumption,
                               drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                               debiasing_model=debiasing_model, debiasing_device=debiasing_device,
                               total_slices=total_slices, constant_perturbations=constant_perturbations)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size)
 

    teacher_model = load_segmentation_model(drop_p=drop_p, incl_p=incl_p, max_iter=max_iter,
                                             rep=i, n_volumes=n_volumes, cuda_no=cuda_no,
                                             segmentation_type='segmentation',
                                             dataset=ds, total_slices=total_slices, 
                                             loss='dice2')


    model = get_model(n_channels=n_channels, n_classes=n_classes, type=model_name)
    model = model.to(get_device(cuda_no))

    torch.backends.cudnn.benchmark = True    # Speedup
    train(model=model, train_loader=train_loader,
          val_loader=val_loader, lr=1e-4, epochs=None,
          save_extension=crt_extension, cuda_no=cuda_no, criterion=crt, save_path=GP['models'],
          teacher_model=teacher_model)
