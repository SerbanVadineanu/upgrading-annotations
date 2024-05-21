import torch
import numpy as np
import random
from utility_modules.network_architectures.unet import UNet, UNetLabelPass, GeneralUNet
from utility_modules.network_architectures.segnet import SegNet
from utility_modules.losses import CrossEntropy2d, DiceLoss, DiceLoss2, SoftDiceLoss2, DiceLossSigmoid, DiceLossCombined
import msd_pytorch as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
import cv2
import itertools
from skimage.util import map_array
import torchvision


def get_device(cuda_no=0):
    return torch.device(f'cuda:{cuda_no}' if (torch.cuda.is_available() and cuda_no is not None) else 'cpu')


def train(model, train_loader, val_loader, epochs=50, lr=0.01,
          optimizer='adam', criterion_name='crossentropy', save_extension='',
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
    elif isinstance(model, GeneralUNet):
        model_name = 'unet'
    elif isinstance(model, UNetLabelPass):
        model_name = 'unet_label_pass'
    elif isinstance(model, torchvision.models.VGG):
        model_name = 'vgg'
    elif isinstance(model, torchvision.models.ResNet):
        model_name = 'resnet'
    else:
        model_name = 'msd'
        model.set_normalization(train_loader)

    print(f'Training {model_name}\n', flush=True)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    if criterion_name == 'crossentropy':
        criterion = CrossEntropy2d()
#         criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'dice':
        criterion = DiceLoss()
    elif criterion_name == 'dice2':
        criterion = DiceLoss2()
    elif criterion_name == 'soft_dice2':
        criterion = SoftDiceLoss2()
    elif criterion_name == 'dice_combined':
        criterion = DiceLossCombined()
    elif criterion_name == 'mse':
        criterion = torch.nn.MSELoss()
    elif criterion_name == 'crossentropy1d':
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]).float(), ignore_index=-1).to(get_device(0))
    elif criterion_name == 'dice_sigmoid':
        criterion = DiceLossSigmoid()

    min_val_loss = 1e5

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
                
                # Extra step for MSD
                if model_name == 'msd':
                    model.set_input(batch)
#                     model.set_target(labels)

                output = model(batch)
                loss = criterion(output, labels)

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

        if min_val_loss == 0:
            return
        
        print('Improvement: ', np.round((min_val_loss - v_l) / min_val_loss, 3), flush=True)
        
        if i == 0:
            # We initialize a counter for lack of improvement in validation score
            no_improvement_count = 0
        elif (min_val_loss - v_l) / min_val_loss >= 0.001:
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


def get_model(n_channels, n_classes, type='segnet'):
    if type == 'segnet':
        model = SegNet(n_channels=n_channels, n_classes=n_classes)
    elif type == 'unet':
        model = UNet(n_channels=n_channels, n_classes=n_classes)
    elif type == 'unet_label_pass':
        model = UNetLabelPass(n_channels=n_channels, n_classes=n_classes)
    elif type == 'msd':
        depth = 50
        width = 1
        dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        model = mp.MSDSegmentationModel(n_channels, n_classes, depth, width,
                                        dilations=dilations)

    return model


def net_prediction(model, img, cuda_no=0):
    with torch.no_grad():
        device = get_device(cuda_no)
            
        img = img.view(1, *img.shape)
            
        if isinstance(model, mp.MSDSegmentationModel):
            pred = model.net(img.to(device))
        else:
            model = model.to(device)
            pred = model(img.to(device))

    return pred


def show_prediction(model, img, cuda_no=0, ax=None, title=''):
    prediction = net_prediction(model, img, cuda_no)
    _, pred_img = torch.max(prediction.detach(), dim=1)
    pred_img = pred_img.cpu().numpy().squeeze()
    
    show_label(pred_img, ax=ax, title=title)
        

def show_label(lab, ax=None, title=''):
    img_to_show = np.zeros((3, *lab.shape))
    img_to_show[0][np.where(lab == 1)] = 1
    img_to_show[1][np.where(lab == 2)] = 1
    
    if ax is None:
        plt.imshow(np.transpose(img_to_show, (1, 2, 0)))
        plt.title(title)
    else:
        ax.imshow(np.transpose(img_to_show, (1, 2, 0)))
        ax.set_title(title)


def get_acc_metrics(predict, target):
    '''
    We get here TP, FP, FN. We also assume binary classification.
    '''
#     predict = torch.nn.Softmax2d()(predict)
    predict = torch.argmax(predict, dim=1)
    target = target.to(predict.get_device()).view(*predict.shape)

    tp = torch.sum(predict * target)
    fp = torch.sum(predict) - tp
    fn = torch.sum(target) - tp
                
    return tp, fp, fn


def get_performance(model, data, cuda_no=0):
    device = get_device(cuda_no)

    tp = 0
    fp = 0
    fn = 0
    
    for datum in tqdm(data):
        img, label = datum
        pred = net_prediction(model, img, cuda_no=cuda_no)
       
        tp_, fp_, fn_ = get_acc_metrics(pred.detach(), label.to(device), 2)

        tp += tp_.item()
        fp += fp_.item()
        fn += fn_.item()

    return tp, fp, fn


def init_plots():
    # Edit the font, font size, and axes width

    global colors, markers

    mpl.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 17

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # Generate 2 colors from the 'tab10' colormap
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ('o', 'v', 's', '*', '+')


# Returns a list of binary masks corresponding to each cell
def expand_labels(mask, n_cells):
    new_mask = np.zeros((n_cells, *mask.shape))

    obj_ids, counts = np.unique(mask, return_counts=True)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of binary masks
    mask = mask == obj_ids[:, None, None]

    mask = mask.astype(np.uint8)

    for i, obj_id in enumerate(obj_ids):
        new_mask[obj_id - 1] = mask[i].astype(np.uint8)

    return new_mask.astype(np.uint8)


def perform_omission(mask, omission_rate=0.0, omitted_labels=None, seed=None, inplace=False, cell_labels=None):
    """
    Performs the omission perturbation on the cell mask.
    :param mask: A 2D image or a 3D volume containing the cell labels.
    :param omission_rate: The proportion of the cells to be dropped.
    :param omitted_labels: The specific cell labels to be discarded.
    :param seed: Random seed.
    :param inplace: The perturbation is performed in place.
    :param cell_labels: A list with all labels in the image to avoid performing np.unique so many times.
    :return: The mask with the applied perturbation.
    """
    assert omission_rate > 0 or omitted_labels is not None

    if not inplace:
        mask = mask.copy()

    if seed is not None:
        np.random.seed(seed)

    if cell_labels is None:
        cell_labels = np.unique(mask)
    cell_labels = cell_labels[np.where(cell_labels != 0)]
    no_cells = len(cell_labels)

    if no_cells == 0:
        return mask

    if omitted_labels is not None:
        omitted_indices = []
        for i, ol in enumerate(omitted_labels):
            omi = np.where(cell_labels == ol)[0]
            if len(omi) == 0:
                continue
            else:
                omitted_indices.append(omi[0])
    else:
        omitted_indices = np.random.choice(range(no_cells), max(1, int(omission_rate * no_cells)), replace=False)

    cell_labels_copy = cell_labels.copy()
    cell_labels_copy[omitted_indices] = 0

    mask = map_array(mask, input_vals=cell_labels, output_vals=cell_labels_copy)

    return mask


def merge_cell_images(image_main, mask_main, image_extra, mask_extra, no_main_cells=None, threshold=0.8, inplace=False):
    """
    Merges an image belonging to the initial data set with an additional image with different cells
    in order to artificially perform the inclusion perturbation.
    :param image_main: The 2D image or 3D volume of the main data set scaled between 0 and 1.
    :param mask_main: The 2D mask or the 3D volume mask of the main data set.
    :param image_extra: The 2D image or 3D volume of the additional data set scaled between 0 and 1.
    :param mask_extra: The 2D mask or the 3D volume mask of the additional data set.
    :param no_main_cells: The total number of cells in the main data-set / volume
    :param threshold: The intensity threshold by which we include the cells from the additional image.
    :param inplace: The merging is performed in place.
    :return: Merged image, merged mask.
    """
    if not inplace:
        image_main = image_main.copy()
        mask_main = mask_main.copy()

    matching_indices = np.where(((mask_extra > 0) | (image_extra > threshold)) & (mask_main == 0))
    image_main[matching_indices] = image_extra[matching_indices]

    cell_labels_extra = np.unique(mask_extra)
    cell_labels_extra = cell_labels_extra[np.where(cell_labels_extra != 0)]

    if no_main_cells is None:
        no_main_cells = len(np.unique(mask_main)) - 1

    # Assuming the cell mask values are from 1 to no_cells in both the main and the additional mask
    for extra_cell in cell_labels_extra:
        mask_main[np.where((mask_main == 0) & (mask_extra == extra_cell))] = no_main_cells + extra_cell

    return image_main, mask_main


def perform_inclusion(mask, inclusion_point=None, inclusion_rate=0.0, included_labels=None, seed=None, inplace=False):
    """
    Perform the inclusion perturbation. We assume two types of cells are present in the mask ordered from
    1 to the number of main cells; and the number of main cells + 1 to the number of main + additional cells.
    :param mask:  A 2D image or a 3D volume containing the cell labels. It contains both the main and the additional cells.
    :param inclusion_point: The label value after/before which the labels are part of the additional set.
                            If negative the values are before (<=), if positive the values are after (>).
    :param inclusion_rate: The proportion of the cells we keep from the additional category.
    :param included_labels: The specific cell labels to be included with values from 1 to number of additional cells.
    :param inplace: The perturbation is performed in place.
    :param seed: Random seed.
    :return: The mask containing all main cells and a percentage of the additional cells.
    """
    assert inclusion_rate > 0 or included_labels is not None

    if not inplace:
        mask = mask.copy()

    cell_labels = np.unique(mask)

    if inclusion_point > 0:
        additional_cell_labels = cell_labels[np.where(cell_labels > inclusion_point)]
    elif inclusion_point < 0:
        additional_cell_labels = cell_labels[np.where((0 < cell_labels) & (cell_labels <= abs(inclusion_point)))]

    if len(additional_cell_labels) == 0:
        return mask

    if seed is not None:
        np.random.seed(seed)

    if included_labels is not None:
        discarded_cells = np.array([dc for dc in additional_cell_labels if dc not in included_labels])
    else:
        discarded_cells = np.random.choice(additional_cell_labels,
                                           max(1, int(np.floor((1 - inclusion_rate) * len(additional_cell_labels)))),
                                           replace=False)

    if len(discarded_cells) == 0:
        return mask

    discarded_cells_indices = []
    for i, dc in enumerate(discarded_cells):
        dci = np.where(cell_labels == dc)[0]
        if len(dci) == 0:
            continue
        else:
            discarded_cells_indices.append(dci[0])

    cell_labels_copy = cell_labels.copy()
    cell_labels_copy[discarded_cells_indices] = 0

    mask = map_array(mask, input_vals=cell_labels, output_vals=cell_labels_copy)

    return mask


def perform_bias(mask, qmax=2, seed=None, operation_iteration_list=None, bias_per_cell=False, inplace=False, kernel_size=3):
    """
    Add enlarging or shrinking bias to cell masks. It covers the cases when a mask is a slice with the bias applied
    either for the entire slice or for each cell. Also, it covers the cases when the mask is a volume and the bias is
    applied either for every slice or for each individual cell (no slice & cell).
    :param mask: A 2D image or a 3D volume containing the cell labels.
    :param qmax: The maximum number of iterations for which we perform the enlargement or shrinkage of the cells.
    :param operation_iteration_list: A list of tuples (operation, iteration) with each element corresponding to
    the perturbation applied to a cell. Operation is a character (e or d) and iteration an int.
    :param bias_per_cell: If true, the mask is split in multiple binary masks (one for each cell) and the bias is
    applied per cell.
    :param seed: Random seed.
    :param inplace: The perturbation is performed in place.
    :return: A binary label with the cells expanded or shrunk.
    """

    def get_random_operation_iterations():
        op = cv2.erode if np.random.rand() > 0.5 else cv2.dilate
        it = np.random.randint(1, qmax + 1)

        return op, it

    if not inplace:
        mask = mask.copy()

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Binarize the mask
    if bias_per_cell:
        mask = expand_labels(mask, n_cells=len(operation_iteration_list))
    else:
        mask[np.where(mask != 0)] = 1

    if seed is not None:
        np.random.seed(seed)

    mask = np.squeeze(mask)     # Make sure the mask does not have too many dimensions
    if len(mask.shape) == 2:
        if operation_iteration_list is None:
            operation, iterations = get_random_operation_iterations()
        else:
            operation, iterations = operation_iteration_list[0]
            operation = cv2.erode if operation == 'e' else cv2.dilate

        mask = operation(mask.astype('uint8'), kernel, iterations=iterations)
    else:
        for i in range(mask.shape[0]):
            if operation_iteration_list is None:
                operation, iterations = get_random_operation_iterations()
            else:
                operation, iterations = operation_iteration_list[i]
                operation = cv2.erode if operation == 'e' else cv2.dilate

            mask[i] = operation(mask[i], kernel, iterations=iterations)

    return mask
