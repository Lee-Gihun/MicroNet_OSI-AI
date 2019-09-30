# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import pickle
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import numpy as np

from pthflops import count_ops
from hyperas import optim as hyperas_optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from hyperas.utils import eval_hyperopt_space

from data_utils import *
from train_tools import *
from models import *
from counting import *

def _logging():
    fpath = './results/AutoML/architecture_search.log'
    logger = logging.getLogger('Architecture Search')
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

def _get_num_params(model):
    num = 0
    for params in model.parameters():
        num += params.view(-1).shape[0]
    return num

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

    SCHEDULER = {'step': lr_scheduler.StepLR,
                 'multistep': lr_scheduler.MultiStepLR,
                 'cosine': lr_scheduler.CosineAnnealingLR}

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
    
    #don't know = {{choice(['i', 'dont', 'know'])}}
    
    stride_where = {{choice(['third', 'fourth', 'none'])}}
    if stride_where == 'five':
        blocks_args = [
            'r%s_k%s_s1_e1_i24_o16_se%.2f' % ({{choice([1, 2])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}), 
            'r%s_k%s_s1_e6_i16_o24_se%.2f' % ({{choice([1, 2])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}), 
            'r%s_k%s_s2_e6_i24_o40_se%.2f'% ({{choice([1, 2, 3])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i40_o56_se%.2f' % ({{choice([2, 3])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i56_o72_se%.2f' % ({{choice([2, 3])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s2_e6_i72_o88_se%.2f' % ({{choice([2, 3, 4])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i188_o104_se%.2f' % ({{choice([1, 2])}}, {{choice([2, 3])}}, {{uniform(0.2, 0.4)}})  
        ]
                    
    elif stride_where == 'six':
        blocks_args = [
            'r%s_k%s_s1_e1_i24_o16_se%.2f' % ({{choice([1, 2])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}), 
            'r%s_k%s_s1_e6_i16_o24_se%.2f' % ({{choice([1, 2])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}), 
            'r%s_k%s_s1_e6_i24_o40_se%.2f'% ({{choice([1, 2, 3])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s2_e6_i40_o56_se%.2f' % ({{choice([2, 3])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i56_o72_se%.2f' % ({{choice([2, 3])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s2_e6_i72_o88_se%.2f' % ({{choice([2, 3, 4])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i88_o104_se%.2f' % ({{choice([1, 2])}}, {{choice([2, 3])}}, {{uniform(0.2, 0.4)}})  
        ]
                    
    else:
        blocks_args = [
            'r%s_k%s_s1_e1_i24_o16_se%.2f' % ({{choice([1, 2])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}), 
            'r%s_k%s_s1_e6_i16_o24_se%.2f' % ({{choice([1, 2])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}), 
            'r%s_k%s_s1_e6_i24_o40_se%.2f'% ({{choice([1, 2, 3])}}, {{choice([3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i40_o56_se%.2f' % ({{choice([1, 2])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i56_o72_se%.2f' % ({{choice([1, 2])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s2_e6_i72_o88_se%.2f' % ({{choice([2, 3, 4])}}, {{choice([2, 3, 5])}}, {{uniform(0.2, 0.4)}}),  
            'r%s_k%s_s1_e6_i88_o104_se%.2f' % ({{choice([1, 2])}}, {{choice([2, 3])}}, {{uniform(0.2, 0.4)}})  
        ]
        
    blocks_args, global_params = efficientnet(blocks_args=blocks_args,
                                    activation='swish',
                                    activation_param={},
                                    resolution_coefficient=1,
                                    width_coefficient=1, 
                                    depth_coefficient=1, 
                                    image_size=opt.model.param.image_size, 
                                    num_classes=opt.model.param.num_classes)
    
    model = EfficientNet(blocks_args, 
                         global_params)
    
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
    
    logger.info("model valid loss and valid acc : {:.4f} and {:.2f}%".format(valid_loss, valid_acc*100))
    
    conv_stem = {'kernel': 3, 'stride': 2, 'out_channel': 24}
    last_ops = {'out_channel': 150, 'num_classes': 100}
    activation = 'swish'
    input_size = 32
    use_bias = False

    counter = MicroNetCounter(conv_stem, blocks_args, global_params, last_ops, activation, input_size, use_bias, add_bits_base=32, mul_bits_base=32)

    # Constants
    INPUT_BITS = 16
    ACCUMULATOR_BITS = 16
    PARAMETER_BITS = INPUT_BITS
    SUMMARIZE_BLOCKS = True
    SPARSITY = 0

    params, flops, _, _ = counter.print_summary(SPARSITY, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=SUMMARIZE_BLOCKS)
    
    logger.info("flops: {:.4f}M, params: {:.4f}MBytes".format(flops, params))
    logger.info('score: {:.4f} + {:.4f} = {:.4f}'.format(flops/(10490), params/(36.5*4), flops/(10490) + params/(36.5*4)))
    logger.info('#'*50)
    
    return {'loss': valid_loss, 'status': STATUS_OK, 'model': train_handler.model}
    
if __name__ == "__main__":
    conf_name = sys.argv[1]
    with open('./tmp.pickle', 'wb') as f:
        pickle.dump(conf_name, f)
        
    fpath = './results/AutoML'
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    if os.path.isfile('./results/AutoML/architecture_search.log'):
        os.remove('./results/AutoML/architecture_search.log')
    
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

    SCHEDULER = {'step': lr_scheduler.StepLR,
                 'multistep': lr_scheduler.MultiStepLR,
                 'cosine': lr_scheduler.CosineAnnealingLR}
    
    trials = Trials()
    best_run, best_model, space = hyperas_optim.minimize(model=create_model,
                                                         data=data,
                                                         algo=tpe.suggest,
                                                         functions=[_get_conf, _logging, _get_num_params],
                                                         max_evals=1,
                                                         trials=trials,
                                                         eval_space=True,
                                                         return_space=True)
    
    logger.info("Best performing model chosen hyper-parameters: %s" % best_run)
    
    dataloaders, dataset_sizes = DATASETTER[opt.data.dataset](batch_size=opt.data.batch_size, 
                                                                  valid_size=opt.data.valid_size,
                                                                  root=opt.data.root,
                                                                  fixed_valid=opt.data.fixed_valid,
                                                                  autoaugment=opt.data.autoaugment,
                                                                 aug_policy=opt.data.aug_policy)
    
    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        print("Trial %s vals: %s" % (t, vals))
        tmp = {}
        for k,v in list(vals.items()):
            tmp[k] = v[0]
        logger.info('Trial %d : %s' % (t, eval_hyperopt_space(space, tmp)))
        
    criterion = CRITERION[opt.criterion.algo](**opt.criterion.param) if opt.criterion.get('param') else CRITERION[opt.criterion.algo]()    

    optimizer = OPTIMIZER[opt.optimizer.algo](best_model.parameters(), **opt.optimizer.param) if opt.optimizer.get('param') else OPTIMIZER[opt.optimizer.algo](model.parameters())
    
    # if not use scheduler, you can skip in config json file
    if opt.scheduler.get('enabled', False):
        scheduler_type = lr_scheduler.MultiStepLR if opt.scheduler.type == 'multistep' else lr_scheduler.CosineAnnealingLR if opt.scheduler.type == 'cosine' else lr_scheduler.StepLR
        scheduler = scheduler_type(optimizer, **opt.scheduler.param)
    else:
        scheduler = None
        
    train_handler = TrainHandler(best_model, 
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
        
    train_handler.test_model()
    
    os.remove('./tmp.pickle')