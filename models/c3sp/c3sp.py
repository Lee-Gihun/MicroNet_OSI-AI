import torch
from torch import nn

__all__ = ['SymmetricPad2d']

class SymmetricPad2d(nn.Module):
    """symmetric 0-pad to splited tensors and concat"""
    
    def __init__(self, pad=1):
        super(SymmetricPad2d, self).__init__()
        self.padding1 = nn.ZeroPad2d((pad, 0, pad ,0))
        self.padding2 = nn.ZeroPad2d((pad, 0, 0, pad))
        self.padding3 = nn.ZeroPad2d((0, pad, pad, 0))
        self.padding4 = nn.ZeroPad2d((0, pad, 0, pad))        

    def forward(self, x):
        sub = x.shape[1] // 4
        x1, x2, x3, x4 = x[:,:sub], x[:,sub:2*sub], x[:,2*sub:3*sub], x[:,3*sub:]
        x1, x2, x3, x4 = self.padding1(x1), self.padding2(x2), self.padding3(x3), self.padding4(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x