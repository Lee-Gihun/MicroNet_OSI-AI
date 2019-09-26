import torch

__all__ = ['AverageMeter', 'accuracy', 'accuracy_stat', 'early_exit_pred_mark']

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def accuracy_stat(topk):
    """
    topk: list of topk accuracy values. Must be length of epochs

    return: average, maximum
    """
    return (sum(topk) / len(topk)), (max(topk))

def early_exit_pred_mark(output):
    if len(output) == 2:
        _, pred = torch.max(output[0], 1)
        return pred, output[1]
    
    _, pred = torch.max(output, 1)
    return pred