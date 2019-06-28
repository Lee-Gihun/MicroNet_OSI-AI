import torch
import torch.optim as optim
import torch.nn as nn

from utils import AverageMeter, accuracy

def test(opt, model, test_loader, device):
    model.eval()
    param = opt['train_test']['test']['param']
    CRITERION_MAP = {'cross_entropy': nn.CrossEntropyLoss
                    }
    
    criterion = CRITERION_MAP[opt['train_test']['criterion']]()
    
    losses = AverageMeter('loss')
    topk = AverageMeter(param['verbose'])
    for i, data in enumerate(test_loader):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)

        pred_label = model(image)
        loss = criterion(pred_label, label)

        topk_acc = accuracy(pred_label.data, label.data, topk=(param['topk'], ))
        losses.update(loss.item(), image.size(0))
        topk.update(topk_acc[0], image.size(0))

    return losses.avg, topk.avg