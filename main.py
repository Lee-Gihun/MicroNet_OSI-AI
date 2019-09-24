# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models

from data_utils import *
from train_tools import *
from models import *
from counting import *

DATASETTER = {'cifar10': cifar_10_setter,
              'cifar100': cifar_100_setter}
    
CRITERION = {'mse': nn.MSELoss,
             'cross_entropy': nn.CrossEntropyLoss,
             'label_smoothing': LabelSmoothingLoss}

OPTIMIZER = {'sgd': optim.SGD,
             'adam': optim.Adam,
             'adagrad': optim.Adagrad,
             'rmsprop': optim.RMSprop,
             'radam': RAdam}

SCHEDULER = {'step': lr_scheduler.StepLR,
             'multistep': lr_scheduler.MultiStepLR,
             'cosine': lr_scheduler.CosineAnnealingLR}

PRUNE_METHOD = {'weight': weight_prune,
                'filter': filter_prune}

def _get_dataset(opt):
    param = opt.data
    
    dataloaders, dataset_sizes = DATASETTER[param.dataset](batch_size=param.batch_size, 
                                                           valid_size=param.valid_size,
                                                           root=param.root,
                                                           fixed_valid=param.fixed_valid,
                                                           autoaugment=param.autoaugment,
                                                           aug_policy=param.aug_policy)
    
    return dataloaders, dataset_sizes

def _get_model(opt):
    param = opt.model.param
    
    # AutoML results 
    avail_resource = param.avail_resource
    resolution_coefficient = round(math.pow(param.resolution_coefficient, avail_resource), 2)
    
    print('[model information]')
    print('resolution_coefficient : %.2f' % resolution_coefficient)
    print('width_coefficient : %.2f' % param.width_coefficient)
    print('depth_coefficient : %.2f' % param.depth_coefficient)
    print('=' * 50)
    
    # if you want other blocks_args setting, define if here.
    blocks_args, global_params = efficientnet(blocks_args='default',
                                              activation=param.activation,
                                              activation_param=param.get('activation_param',  {}),
                                              resolution_coefficient=resolution_coefficient,
                                              width_coefficient=param.width_coefficient, 
                                              depth_coefficient=param.depth_coefficient, 
                                              image_size=param.image_size, 
                                              num_classes=param.num_classes)
    
    
    model = EfficientNet(blocks_args, 
                         global_params)
    
    model.to(opt.trainhandler.device)
    
    return model, blocks_args, global_params

def _count_flops_params(opt, blocks_args, global_params, sparsity=0):
    # define different value according to your structure
    conv_stem = {'kernel': 3, 'stride': 2, 'out_channel': 24}
    last_ops = {'out_channel': 150, 'num_classes': global_params.num_classes}
    activation = global_params.activation
    input_size = int(math.ceil(32 * global_params.resolution_coefficient))
    use_bias = False
    
    counter = MicroNetCounter(conv_stem, blocks_args, global_params, last_ops, activation, input_size, use_bias, add_bits_base=32, mul_bits_base=32)

    # Constants
    INPUT_BITS = opt.trainhandler.precision
    ACCUMULATOR_BITS = opt.trainhandler.precision
    PARAMETER_BITS = INPUT_BITS
    SUMMARIZE_BLOCKS = True
    if sparsity != 0:
        SPARSITY = sparsity / 100
    else:
        SPARSITY = sparsity

    params, flops = counter.print_summary(SPARSITY, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=SUMMARIZE_BLOCKS)
    print('flops: {:.4f}M, params: {:.4f}M'.format(flops, params))
    print('score: {:.4f} + {:.4f} = {:.4f}'.format(flops/(10490), params/(36.5 * 4), flops/(10490) + params/(36.5 * 4)))
    print('=' * 50)

def _get_trainhanlder(opt, model, dataloaders, dataset_sizes):    
    criterion = CRITERION[opt.criterion.algo](**opt.criterion.param)
    
    params = adapted_weight_decay(model, opt.optimizer.param.get('weight_decay', 1e-5))
    optimizer = OPTIMIZER[opt.optimizer.algo](params, **opt.optimizer.param)
    #optimizer = OPTIMIZER[opt.optimizer.algo](model.parameters(), **opt.optimizer.param)
    
    if opt.scheduler.enabled:
        scheduler = SCHEDULER[opt.scheduler.type](optimizer, **opt.scheduler.param)
    else:
        scheduler = None
        
    train_handler = TrainHandler(model, 
                                 dataloaders, 
                                 dataset_sizes, 
                                 criterion, 
                                 optimizer, 
                                 scheduler, 
                                 device=opt.trainhandler.device, 
                                 path=opt.trainhandler.path,
                                 mixup=opt.trainhandler.mixup.enabled,
                                 alpha=opt.trainhandler.mixup.alpha,
                                 precision=opt.trainhandler.precision)
    
    train_handler.set_name(opt.trainhandler.name)
    
    return train_handler

def __get_sparsity(opt, round):
    param = opt.model.prune
    
    if param.gradually:
        round_sparsity = param.sparsity * ((1 / param.rounds) * (round + 1))
    else:
        round_sparsity = math.pow(param.sparsity, ((1 / param.rounds) * (round + 1)))

    return round_sparsity
    
def __get_masks(opt, train_handler, round_sparsity, masks):    
    param = opt.model.prune
    
    if not masks:
        masks = PRUNE_METHOD[param.method](train_handler.model, round_sparsity, norm=param.norm, device=opt.trainhandler.device)
    else:
        masks = PRUNE_METHOD[param.method](train_handler.model, round_sparsity, prev_masks=masks, norm=param.norm, device=opt.trainhandler.device)

    return masks
        
def __get_model_name(opt, round):
    param = opt.model.prune
    
    if param.rounds == 1:
        model_name = opt.trainhandler.name + '_pruned_%s_%.2f' % (param.method, param.sparsity) + '_oneshot'
    else:
        model_name = opt.trainhandler.name + '_pruned_%s_%.2f' % (param.method, param.sparsity) + '_iterative_rounds_%d' % (round + 1)
        
    return model_name
    
def __reset_states(opt, train_handler):
    param = opt.model.prune
    
    if param.weight_reset:
        train_handler.reset_model_state()
    else:
        train_handler.optimizer.load_state_dict(train_handler.init_states['optimizer'])
        if train_handler.scheduler != None:
            train_handler.scheduler.load_state_dict(train_handler.init_states['scheduler'])

    return train_handler

def __update_states(opt, train_handler):
    params = adapted_weight_decay(train_handler.model, opt.model.prune.optimizer.get('weight_decay', 1e-5))
    optimizer = OPTIMIZER[opt.optimizer.algo](params, **opt.model.prune.optimizer)
    train_handler.optimizer.load_state_dict(optimizer.state_dict())
    if train_handler.scheduler != None:
        train_handler.scheduler.load_state_dict(SCHEDULER[opt.scheduler.type](optimizer, **opt.model.prune.scheduler).state_dict())

    return train_handler

def _pruning(opt, train_handler, blocks_args, global_params):
    masks = None
        
    for round in range(opt.model.prune.rounds):
        round_sparsity = __get_sparsity(opt, round)
        _count_flops_params(opt, blocks_args, global_params, sparsity=round_sparsity)

        masks = __get_masks(opt, train_handler, round_sparsity, masks)
        train_handler.model.set_masks(masks)
        train_handler.prune = True

        model_name = __get_model_name(opt, round)
        train_handler.set_name(model_name)

        train_handler.test_model(pretrained=True)
        
        train_handler = __reset_states(opt, train_handler)
        train_handler = __update_states(opt, train_handler)
        
        train_handler.train_model(num_epochs=opt.model.prune.num_epochs)
        train_handler.test_model()

def run(opt):
    dataloaders, dataset_sizes = _get_dataset(opt)

    model, blocks_args, global_params = _get_model(opt)

    _count_flops_params(opt, blocks_args, global_params)
        
    train_handler = _get_trainhanlder(opt, model, dataloaders, dataset_sizes)
    
    if not opt.model.pretrained.enabled:
        train_handler.train_model(num_epochs=opt.trainhandler.train.num_epochs)
    else:
        initial_model = torch.load(os.path.join(opt.model.pretrained.fpath, 'initial_model.pth'), map_location=opt.trainhandler.device)
        train_handler.init_states['model'] = copy.deepcopy(initial_model)
        
        fpath = opt.model.pretrained.fpath
        if 'pruned' in fpath:
            masks = []
            for p in train_handler.model.parameters():
                if len(p.data.size()) != 1:
                    p_np = p.data.cpu().numpy()
                    masks.append(torch.from_numpy(np.ones(p_np.shape)).type(p.dtype).to(train_handler.device))
                    train_handler.model.set_masks(masks)
                    
        pretrained_dict = torch.load(os.path.join(fpath, 'trained_model.pth'), map_location=opt.trainhandler.device)
        train_handler.model.load_state_dict(pretrained_dict)

    train_handler.test_model(pretrained=opt.model.pretrained.enabled)
    
    if opt.model.prune.enabled:
        _pruning(opt, train_handler, blocks_args, global_params)
        
if __name__ == "__main__":
    opt = ConfLoader(sys.argv[1]).opt
    
    run(opt)