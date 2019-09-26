import torch
import torch.nn as nn

__all__ = ['LabelSmoothingLoss', 'OverHaulLoss']

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

class OverHaulLoss(nn.Module):
    def __init__(self, label_smoothing=False, classes=100, smoothing=0.0):
        super(OverHaulLoss, self).__init__()
        self.label_smoothing = label_smoothing
        
        if label_smoothing:
            self.loss = LabelSmoothingLoss(classes, smoothing=smoothing)
        else:
            self.loss = nn.CrossEntropyLoss()
        
    def forward(self, output, target):
        if len(output) == 2:
            output = output[0]
        if self.label_smoothing:
            loss = self.loss(output, target)
        else:
            loss = self.loss(output, target)
        
        return loss