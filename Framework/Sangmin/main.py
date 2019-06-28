# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import ujson
import torch
import pickle
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from load_data import load_dataset
from load_model import load_model
from train_model import train
from test_model import test

# load json config file
def _load_opt(conf_fname):
    with open(conf_fname, 'r') as conf:
        opt = ujson.load(conf)
    
    return opt
    
# logging
def _logging(opt):
    fpath = os.path.join(opt['dir']['log'], opt['log']['fpath'])
    if os.path.isfile(fpath):
        os.remove(fpath)

    logger = logging.getLogger(opt['log']['logger'])
    logger.setLevel(logging.DEBUG)
    # create file handler
    handler = logging.FileHandler(fpath)
    handler.setLevel(logging.DEBUG)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    
    return logger

# make dir
def _make_dirs(opt):
    dirs = opt['dir']
    for _, d in dirs.items():
        if not os.path.isdir(d):
            os.makedirs(d)

# load data
def _load_data(opt, logger):
    _begin = datetime.datetime.now()
    train_loader, test_loader = load_dataset(opt)
    _end = datetime.datetime.now()
    
    logger.info('(%s) elapsed for loading %s dataset' % (str(_end - _begin), opt['data']['dataset']))
    
    return train_loader, test_loader

# load neural network model
def _load_model(opt, device, logger):
    _begin = datetime.datetime.now()
    # make various load_model.py
    model = load_model(opt, device)
    _end = datetime.datetime.now()
    
    logger.info('(%s) elapsed for loading %s model' % (str(_end - _begin), opt['model']['algo']))
    
    return model

# train model
def __train_model(opt, model, train_loader, device):
    avg_loss = train(opt, model, train_loader, device)
    
    return avg_loss

# test model
def __test_model(opt, model, test_loader, device):
    losses, topk = test(opt, model, test_loader, device)
    
    return losses, topk


def __get_best_acc(test_accuracy):
    best_acc = 0.
    for acc in test_accuracy:
        if acc[1] > best_acc:
            best_acc = acc[1]

    return best_acc
    
# train and test over epochs
def _train_test_model(opt, model, train_loader, test_loader, device, logger):
    _begin = datetime.datetime.now()
    
    train_loss = []
    test_accuracy = []
    for epoch in tqdm(range(opt['train_test']['num_epochs'])):
        # later, use validation dataset and EarlyStopping
        train_loss.append(__train_model(opt, model, train_loader, device))
        test_accuracy.append(__test_model(opt, model, test_loader, device))
        logger.info('Epoch {} - train loss: {}'.format(epoch, train_loss[-1]))

    
    logger.info('best {}: {}'.format(opt['train_test']['test']['param']['verbose'],
                                     __get_best_acc(test_accuracy)))
    
    _end = datetime.datetime.now()
    
    logger.info('(%s) elapsed for training and testing model by (%d) epochs' % (str(_end - _begin), opt['train_test']['num_epochs']))
    
    torch.save(model.state_dict(), os.path.join(opt['dir']['model'], opt['train_test']['fpath']['model']))
    # when you want to load model
    #model = _load_model(opt, device, logger)
    #model.load_state_dict(torch.load(os.path.join(opt['dir']['model'], opt['train_test']['fpath']['model'])))
    
    return train_loss, test_accuracy

# plot results
def __plot_results(opt, train_loss, test_accuracy):
    fig = plt.figure(figsize = (8, 5))
    plt.title('Train loss & Test Loss graph')
    
    colormap = ['r', 'b']
    label = ['Train Loss', 'Test Loss']
    
    max_epoch = max(len(train_loss), len(test_accuracy))
    
    plt.plot(train_loss, colormap[0],  label=label[0])

    test_acc = []
    for result in test_accuracy:
        test_acc.append(result[0])
    plt.plot(test_acc, colormap[1],  label=label[1])
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss and Accuracy')
    plt.title('Best Test Accuracy is %.3f' % __get_best_acc(test_accuracy))
    plt.ylim(0,4)
    plt.xlim(0, max_epoch)
    plt.legend()
    plt.savefig(os.path.join(opt['dir']['results'], opt['plot_store']['fpath']['img']))

# store results
def __store_results(opt, train_loss, test_accuracy):
    with open(os.path.join(opt['dir']['results'], opt['plot_store']['fpath']['train_loss']), 'wb') as f:
        pickle.dump(train_loss, f)
    
    with open(os.path.join(opt['dir']['results'], opt['plot_store']['fpath']['test_acc']), 'wb') as f:
        pickle.dump(test_accuracy, f)

# plot and store results
def _plot_store_results(opt, train_loss, test_accuracy, logger):
    _begin = datetime.datetime.now()
    __plot_results(opt, train_loss, test_accuracy)
    __store_results(opt, train_loss, test_accuracy)
    
    _end = datetime.datetime.now()
    
    logger.info('(%s) elapsed for plotting and recording results' % (str(_end - _begin)))

# run main function
def run(conf_fname):
    opt = _load_opt(conf_fname)
    logger = _logging(opt)
    _make_dirs(opt)
    logger.info('Start model prediction!')                                                               
    # if it is possible, use gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = _load_data(opt, logger)
    model = _load_model(opt, device, logger)
    
    train_loss, test_accuracy = _train_test_model(opt, model, train_loader, test_loader, device, logger)
        
    _plot_store_results(opt, train_loss, test_accuracy, logger) 

if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2])