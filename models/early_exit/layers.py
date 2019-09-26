import torch
from torch import nn
from ..utils import (
    relu_fn,
    get_same_padding_conv2d,
    BatchNorm,
    GhostBatchNorm
)

__all__ = ['EarlyExitBlock', 'LogitCond']

ACTIVATION = {'swish': relu_fn,
              'celu': nn.CELU,
              'relu': nn.ReLU}

class EarlyExitBlock(nn.Module):
    def __init__(self, global_params, in_channels=40, final_channels=150):
        super(EarlyExitBlock, self).__init__()
        
        # activation func
        self.act = ACTIVATION[global_params.activation](**global_params.activation_param)
        
        self.bn = GhostBatchNorm if global_params.ghost_bn else BatchNorm
        
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        
        # conv module
        self._depthwise_conv = Conv2d(in_channels, in_channels, kernel_size=3, bias=False, groups=in_channels)
        self._bn0 = self.bn(in_channels)
        
        self._projection_conv = Conv2d(in_channels, final_channels, kernel_size=1, bias=False)
        self._bn1 = self.bn(final_channels)
        
        # classifier module
        self._globalavgpool = nn.AdaptiveAvgPool2d(1)
        self._shallow_classifier = Conv2d(final_channels, global_params.num_classes, kernel_size=1, bias=True)
    
    def forward(self, x):
        # conv
        x = self.act(self._bn0(self._depthwise_conv(x)))
        x = self.act(self._bn1(self._projection_conv(x)))

        # classifier
        x = self._globalavgpool(x)
        x = self._shallow_classifier(x).squeeze()
        return x
    
    
class LogitCond(nn.Module):
    def __init__(self, thres=1.0):
        super(LogitCond, self).__init__()
        self.thres = thres
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        logits = self.softmax(outputs)
        max_logits, _ = torch.max(logits, dim=1)
        
        cond_up = (max_logits > self.thres)
        cond_down = (max_logits <= self.thres)
        
        return cond_up, cond_down