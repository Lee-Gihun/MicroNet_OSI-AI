import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import to_var, Identity

__all__ = ['MaskedLinear', 'MaskedConv2dDynamicSamePadding', 'MaskedConv2dStaticSamePadding']


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, requires_grad=False)
    
    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            self.weight.data = self.weight.data*mask_var.data
        return F.linear(x, self.weight, self.bias)

    
class MaskedConv2dDynamicSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.mask_flag = False
            
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        # print('mask shape: {}'.format(self.mask.data.size()))
        # print('weight shape {}'.format(self.weight.data.size()))
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, requires_grad=False)
    
    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
            
        if self.mask_flag == True:
            mask_var = self.get_mask()
            # print(self.weight)
            # print(self.mask)
            # print('weight/mask id: {} {}'.format(self.weight.get_device(), mask_var.get_device()))
            self.weight.data = self.weight.data * mask_var.data
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    
class MaskedConv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super(MaskedConv2dStaticSamePadding, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.mask_flag = False
    
        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()
            
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        # print('mask shape: {}'.format(self.mask.data.size()))
        # print('weight shape {}'.format(self.weight.data.size()))
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True
    
    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, requires_grad=False)
    
    def forward(self, x):
        x = self.static_padding(x)
        if self.mask_flag == True:
            mask_var = self.get_mask()
            # print(self.weight)
            # print(self.mask)
            # print('weight/mask id: {} {}'.format(self.weight.get_device(), mask_var.get_device()))
            self.weight.data = self.weight.data * mask_var.data
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)