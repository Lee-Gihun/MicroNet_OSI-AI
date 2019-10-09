'''
Early Exiting EfficientNet. The network backbone consists of efficientnet architecture. 
The code is refactored from Pytorch implementation and early exiting module is added.

<Reference>

[1] Luke Melas-Kyriazi, GitHub repository, https://github.com/lukemelas/EfficientNet-PyTorch

'''


import math
import torch
from torch import nn
from torch.nn import functional as F

from ..utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    BatchNorm,
    GhostBatchNorm
)

from ..EfficientNet import EfficientNet
from ..pruning.layers import *
from .layers import *

ACTIVATION = {'swish': relu_fn,
              'celu': nn.CELU,
              'relu': nn.ReLU}


class EfficientNet_EarlyExiting(EfficientNet):
    """
    An early exiting module is added to pre-defined position. At test time, if the condition is meeted (if above threshold),
    exits from early exiting module and do not proceed further. If condition is not meeted, proceeds the whole network.
    
    returns (outputs, marks) at test time. outputs is logits from network and marks means:
    0 : results from the end of network
    1 : results from the exiting module
    """
    
    def __init__(self, blocks_args=None, global_params=None, early_exit=None):
        super().__init__(blocks_args, global_params)
        self._early_exit = early_exit
        
        self._exit = EarlyExitBlock(global_params, early_exit.in_channels, early_exit.final_channels)
        self._exit_cond_train = LogitCond(1) # do not exit anything when training
        self._exit_cond_test = LogitCond(early_exit.thres) # exit samples above thres
        
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """            
        if self.upsample:
            inputs = self.upsample(inputs)
            
        output = torch.zeros(inputs.size(0), 100).type(inputs.dtype).to(self._early_exit.device)
        exit_mark = torch.zeros(inputs.size(0)).long().type(inputs.dtype).to(self._early_exit.device)
        """ Returns output of the final convolution layer """

        # Stem
        x = self.act(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            
            if idx == self._early_exit.blocks_idx: # if early exiting block
                early_exit = self._exit(x) # get outputs of early exiting module
                
                if self.training:
                    cond_up, cond_down = self._exit_cond_train(early_exit)
                else:
                    cond_up, cond_down = self._exit_cond_test(early_exit)
                    
                output[cond_up] = early_exit[cond_up]
                exit_mark[cond_up] = 1 # mark as 1 if exited
                
                if (cond_down.sum().item() == 0) and (not self.training):
                    return output, exit_mark
                
                x = x[cond_down]

        # Head
        x = self.act(self._bn1(self._conv_head(x)))
                
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        
        if not self.training:
            output[cond_down] = x
            
            return output, exit_mark

        return early_exit
    
    def set_masks(self, masks, prev=False):
        if not prev:
            self.masks = masks
        else:
            masks = self.masks
            
        self._exit._depthwise_conv.set_mask(masks[0])
        self._exit._expand_conv.set_mask(masks[1])
        self._exit._fc.set_mask(masks[2])