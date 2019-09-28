import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import linalg as LA
from .utils import prune_rate, arg_nonzero_min

__all__ = ['weight_prune', 'filter_prune']

def weight_prune(model, pruning_perc, prev_masks=None, norm=False, device='cpu'):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''    
    idx = 0
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1 and p.requires_grad:
            if not prev_masks:
                if not norm:
                    all_weights += list(abs(p.cpu().data.numpy()).flatten())
                else:
                    all_weights += list(abs(p.cpu().data.numpy()).flatten() / LA.norm(abs(p.cpu().data.numpy()).flatten()))
            else:
                if not norm:
                    all_weights += list(abs(p.cpu().data.numpy() * prev_masks[idx].cpu().numpy()).flatten())
                else:
                    all_weights += list(abs(p.cpu().data.numpy() * prev_masks[idx].cpu().numpy()).flatten() / LA.norm(abs(p.cpu().data.numpy() * prev_masks[idx].cpu().numpy()).flatten()))
                idx += 1
    threshold = np.percentile(np.array(all_weights), pruning_perc)
    
    # generate mask
    idx = 0
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1 and p.requires_grad:
            if not prev_masks:
                if not norm:
                    pruned_inds = p.data.abs() > threshold
                else:
                    norm_value = torch.norm(p.data.abs(), p=2)
                    norm_value = torch.sum(norm_value)
                    pruned_inds = (p.data.abs() / norm_value) > threshold
            else:
                if not norm:
                    pruned_inds = p.data.abs() * prev_masks[idx] > threshold
                    idx += 1
                else:
                    norm_value = torch.norm(p.data.abs() * prev_masks[idx], p=2)
                    norm_value = torch.sum(norm_value)
                    pruned_inds = (p.data.abs() * prev_masks[idx] / norm_value) > threshold
                    idx += 1
                    
            masks.append(pruned_inds.type(p.dtype).to(device))
    return masks


def prune_one_filter(model, masks, norm, device='cpu'):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():
        if len(p.data.size()) != 1 and p.requires_grad: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                #masks.append(np.ones(p_np.shape).astype('float16'))
                masks.append(torch.from_numpy(np.ones(p_np.shape)).type(p.dtype).to(device))
                
            if len(p.data.size()) == 4:
                # find the scaled l2 norm for each filter this layer
                value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                    .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # it means fully_connected layer
            else:
                value_this_layer = np.square(p_np.sum(axis=1)/p_np.shape[1])
                
            # normalization (important)
            if norm:
                value_this_layer = value_this_layer / np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = torch.tensor(0).type(p.dtype).to(device)

    '''
    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))
    '''
    return masks


def filter_prune(model, pruning_perc, prev_masks=None, norm=True, device='cpu'):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    if not prev_masks:
        masks = []
    else:
        masks = prev_masks
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks, norm, device=device)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        #print('{:.2f} pruned'.format(current_pruning_perc))

    return masks