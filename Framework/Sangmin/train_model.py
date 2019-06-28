import torch
import torch.optim as optim
import torch.nn as nn

def train(opt, model, train_loader, device):
    model.train()
    param = opt['train_test']['train']['param']
    OPTIM_MAP = {'sgd': optim.SGD,
                  'adam': optim.Adam,
                  'adagrad': optim.Adagrad,
                  'rmsprop': optim.RMSprop
                 }
    CRITERION_MAP = {'cross_entropy': nn.CrossEntropyLoss
                    }
    
    optimizer = OPTIM_MAP[param['optim']](model.parameters(), 
                                          lr=param['learning_rate'],
                                          momentum=param['momentum'], 
                                          weight_decay=param['weight_decay'])
    criterion = CRITERION_MAP[opt['train_test']['criterion']]()
    
    losses = []
    for i, data in enumerate(train_loader):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)

        pred_label = model(image)
        loss = criterion(pred_label, label)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = sum(losses) / len(losses)

    return avg_loss