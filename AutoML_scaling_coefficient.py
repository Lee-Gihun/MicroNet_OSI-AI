# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import math
import random
import shutil
import pickle
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import numpy as np

from PIL import Image, ImageEnhance, ImageOps

from hyperas import optim as hyperas_optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space

from data_utils import *
from train_tools import *
from models import *
from counting import *

def _logging():
    fpath = './results/AutoML/scaling_coefficient.log'
    logger = logging.getLogger('Scaling Coefficient')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.FileHandler(fpath)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def _get_conf():
    with open('./tmp.pickle', 'rb') as f:
        conf_name = pickle.load(f)
        
    opt = ConfLoader(conf_name).opt
    
    return opt

def _count_flops_params(blocks_args, global_params):
    opt = _get_conf()
    logger = _logging()
    
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
    SPARSITY = 0
    
    params, flops = counter.print_summary(SPARSITY, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=SUMMARIZE_BLOCKS)
    logger.info('flops: {:.4f}M, params: {:.4f}M'.format(flops, params))
    logger.info('score: {:.4f} + {:.4f} = {:.4f}'.format(flops/(10490), params/(36.5 * 4), flops/(10490) + params/(36.5 * 4)))
    
def data():
    opt = _get_conf()
    
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

    dataloaders, dataset_sizes = DATASETTER[opt.data.dataset](batch_size=opt.data.batch_size, 
                                                              valid_size=opt.data.valid_size,
                                                              root=opt.data.root,
                                                              fixed_valid=opt.data.fixed_valid,
                                                              autoaugment=opt.data.autoaugment,
                                                              aug_policy=opt.data.aug_policy)
    
    return dataloaders, dataset_sizes

def create_model(dataloaders, dataset_sizes):
    opt = _get_conf()
    logger = _logging()
    
    resolution_coefficient = round({{uniform(1.0, 1.4)}}, 2)
    width_coefficient = round(math.sqrt(2.0/math.pow(resolution_coefficient, 2)), 2)
    
    logger.info('resolution coefficient is %s' % resolution_coefficient)
    logger.info('width coefficient is %s' % width_coefficient)

    blocks_args, global_params = efficientnet(blocks_args='default',
                                              activation=opt.model.param.activation,
                                              activation_param=opt.model.param.get('activation_param', {}),
                                              resolution_coefficient=resolution_coefficient,
                                              width_coefficient=width_coefficient, 
                                              depth_coefficient=1.0, 
                                              image_size=opt.model.param.image_size, 
                                              num_classes=opt.model.param.num_classes)

    model = EfficientNet(blocks_args, 
                         global_params)
    
    model.to(opt.trainhandler.device)
    
    criterion = CRITERION[opt.criterion.algo](**opt.criterion.param) if opt.criterion.get('param') else CRITERION[opt.criterion.algo]()    

    optimizer = OPTIMIZER[opt.optimizer.algo](model.parameters(), **opt.optimizer.param) if opt.optimizer.get('param') else OPTIMIZER[opt.optimizer.algo](model.parameters())
    
    # if not use scheduler, you can skip in config json file
    if opt.scheduler.get('enabled', False):
        scheduler_type = lr_scheduler.MultiStepLR if opt.scheduler.type == 'multistep' else lr_scheduler.CosineAnnealingLR if opt.scheduler.type == 'cosine' else lr_scheduler.StepLR
        scheduler = scheduler_type(optimizer, **opt.scheduler.param)
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
    
    train_losses, valid_losses, train_accs, valid_accs = train_handler.train_model(num_epochs=opt.trainhandler.train.num_epochs)
    
    _, valid_loss = sorted(valid_losses, key = lambda x: x[1])[0]
    _, valid_acc = sorted(valid_accs, key = lambda x: x[1], reverse=True)[0]
    
    _count_flops_params(blocks_args, global_params)
    logger.info('Validation accuracy : %.2f' % (valid_acc * 100))
    logger.info('=' * 30)
    
    return {'loss': valid_loss, 'status': STATUS_OK, 'model': train_handler.model}
    
if __name__ == '__main__':
    conf_name = sys.argv[1]
    with open('./tmp.pickle', 'wb') as f:
        pickle.dump(conf_name, f)
        
    fpath = './results/AutoML'
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    if os.path.isfile('./results/AutoML/scaling_coefficient.log'):
        os.remove('./results/AutoML/scaling_coefficient.log')
    
    opt = ConfLoader(conf_name).opt
    logger = _logging()
    
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
    
    trials = Trials()
    best_run, best_model, space = hyperas_optim.minimize(model=create_model,
                                                         data=data,
                                                         algo=tpe.suggest,
                                                         functions=[_get_conf, _logging, _count_flops_params],
                                                         max_evals=1,
                                                         trials=trials,
                                                         eval_space=True,
                                                         return_space=True)
    
    logger.info('=' * 30)
    logger.info('Best performing model chosen hyper-parameters: %s' % best_run)
    logger.info('=' * 30)
    
    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        tmp = {}
        for k,v in list(vals.items()):
            tmp[k] = v[0]
        logger.info('Trial %d : %s' % (t, eval_hyperopt_space(space, tmp)))
        logger.info('=' * 30)
    
    os.remove('./tmp.pickle')