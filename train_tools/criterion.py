"""
<Reference>

Label Smoothing pytorch implementation:
[1] Sino Begonia, GitHub repository, https://github.com/diggerdu/VGGish_Genre_Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LabelSmoothingLoss', 'SoftLabelSmoothingLoss', 'OverHaulLoss']

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class SoftLabelSmoothingLoss(nn.Module):
    def __init__(self, classes=100, smoothing=0.0, dim=-1):
        super(SoftLabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        max_softval, _ = torch.max(logits, dim=1)
        target_softval = logits[range(logits.shape[0]), target]
        log_logits = logits.log()
        
        with torch.no_grad():
            smooth_target = torch.zeros_like(output)
            smooth_target.fill_(self.smoothing / (self.cls - 1))
            smooth_target.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        loss = torch.sum(-smooth_target * log_logits, dim=self.dim)
        #loss = loss * (1 + max_softval)
        #loss = loss * (1 + target_softval)
        loss = loss * (1 + max_softval + target_softval)
        loss = loss.mean()

        return loss       
    
class OverHaulLoss(nn.Module):
    def __init__(self, soft_label_smoothing=False, label_smoothing=False, classes=100, smoothing=0.0):
        super(OverHaulLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.soft_label_smoothing = soft_label_smoothing
        
        if soft_label_smoothing:
            self.loss = SoftLabelSmoothingLoss(classes, smoothing=smoothing)
        elif label_smoothing:
            self.loss = LabelSmoothingLoss(classes, smoothing=smoothing)
        else:
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self, output, target):
        if len(output) == 2:
            output = output[0]
        loss = self.loss(output, target)
        
        return loss
    