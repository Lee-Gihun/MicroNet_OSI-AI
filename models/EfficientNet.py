'''
Network modules for EfficientNet. The network backbone consists of efficientnet architecture. 
The code is refactored from Pytorch implementation and pruning module is added.

<Reference>

[1] Luke Melas-Kyriazi, GitHub repository, https://github.com/lukemelas/EfficientNet-PyTorch

'''

import math
import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    BatchNorm,
    GhostBatchNorm
)

from .pruning.layers import *
        
ACTIVATION = {'swish': relu_fn,
              'celu': nn.CELU,
              'relu': nn.ReLU}
    
class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
            
        self.act = ACTIVATION[global_params.activation](**global_params.activation_param)
        
        self.bn = GhostBatchNorm if global_params.ghost_bn else BatchNorm
        
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = self.bn(num_features=oup)
            
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._conv = Conv2d(
                in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
                kernel_size=k, stride=s, bias=False)
            
        s = s if type(s) == int else s[0]
        if s >= 2:
            self.pooling = nn.AvgPool2d(3, stride=(s, s), padding=1)
            
        self._bn1 = self.bn(num_features=oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = self.bn(num_features=final_oup)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self.act(self._bn0(self._expand_conv(x)))
        x = self.act(self._bn1(self._conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self.act(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
    
class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        self.act = ACTIVATION[global_params.activation](**global_params.activation_param)
        
        self.bn = GhostBatchNorm if global_params.ghost_bn else BatchNorm

        self.upsample = None
        if global_params.resolution_coefficient != 1.0:
            self.upsample = nn.Upsample(int(math.ceil(32 * global_params.resolution_coefficient)), mode='nearest')
        
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(24, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = self.bn(num_features=out_channels)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(150, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = self.bn(num_features=out_channels)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        
        self._fc = MaskedLinear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self.act(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self.act(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """            
        if self.upsample:
            inputs = self.upsample(inputs)
            
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x
    
    def set_masks(self, masks, prev=False):
        """"""
        if not prev:
            self.masks = masks
        else:
            masks = self.masks
            
        self._conv_stem.set_mask(masks[0])
        idx = 1
        for block in self._blocks:
            if block._block_args.expand_ratio != 1:
                block._expand_conv.set_mask(masks[idx])
                idx += 1
                
            block._conv.set_mask(masks[idx])
            idx += 1
            
            if block.has_se:
                block._se_reduce.set_mask(masks[idx])
                block._se_expand.set_mask(masks[idx+1])
                idx += 2
            block._project_conv.set_mask(masks[idx])
            idx += 1
        self._conv_head.set_mask(masks[idx])
        self._fc.set_mask(masks[idx+1])