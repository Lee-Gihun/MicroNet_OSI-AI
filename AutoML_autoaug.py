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
    fpath = './results/AutoML/cifar100_autoaug_policy.log'
    logger = logging.getLogger('Autoaugment Policy')
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
    
def data():
    # it just for processing, meaningless
    dataloader = None
    dataset_size = None
    
    return dataloader, dataset_size

def create_model(dataloader, dataset_size):
    class SubPolicy():
        def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
            ranges = {
                "shearX": np.linspace(0, 0.3, 10),
                "shearY": np.linspace(0, 0.3, 10),
                "translateX": np.linspace(0, 150 / 331, 10),
                "translateY": np.linspace(0, 150 / 331, 10),
                "rotate": np.linspace(0, 30, 10),
                "color": np.linspace(0.0, 0.9, 10),
                "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
                "solarize": np.linspace(256, 0, 10),
                "contrast": np.linspace(0.0, 0.9, 10),
                "sharpness": np.linspace(0.0, 0.9, 10),
                "brightness": np.linspace(0.0, 0.9, 10),
                "autocontrast": [0] * 10,
                "equalize": [0] * 10,
                "invert": [0] * 10
            }

            # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
            def rotate_with_fill(img, magnitude):
                rot = img.convert("RGBA").rotate(magnitude)
                return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

            func = {
                "shearX": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                    Image.BICUBIC, fillcolor=fillcolor),
                "shearY": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                    Image.BICUBIC, fillcolor=fillcolor),
                "translateX": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                    fillcolor=fillcolor),
                "translateY": lambda img, magnitude: img.transform(
                    img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                    fillcolor=fillcolor),
                "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
                # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
                "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
                "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
                "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
                "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
                "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
                "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                    1 + magnitude * random.choice([-1, 1])),
                "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
                "equalize": lambda img, magnitude: ImageOps.equalize(img),
                "invert": lambda img, magnitude: ImageOps.invert(img)
            }

            # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
            #     operation1, ranges[operation1][magnitude_idx1],
            #     operation2, ranges[operation2][magnitude_idx2])
            self.p1 = p1
            self.operation1 = func[operation1]
            self.magnitude1 = ranges[operation1][magnitude_idx1]
            self.p2 = p2
            self.operation2 = func[operation2]
            self.magnitude2 = ranges[operation2][magnitude_idx2]


        def __call__(self, img):
            if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
            if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
            return img

    class Autoaug():
        def __init__(self, fillcolor=(128, 128, 128)):
            self.policies = [
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),

                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),

                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),

                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),

                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor),
                SubPolicy({{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, {{uniform(0, 1.0)}}, {{choice(["shearX", "shearY", "translateX", "translateY", "rotate", "color", "posterize", "solarize", "contrast", "sharpness", "brightness", "autocontrast", "equalize", "invert"])}}, {{choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}}, fillcolor)
            ]
            
        def __call__(self, img):
            policy_idx = random.randint(0, len(self.policies) - 1)
            return self.policies[policy_idx](img)

        def __repr__(self):
            return 'AutoAugment CIFAR100 Policy'

    opt = _get_conf()
    logger = _logging()
    if os.path.isdir(opt.data.root):
        shutil.rmtree(opt.data.root)
    
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
                                                              aug_policy=Autoaug())
    
    avail_resource = opt.model.param.avail_resource
    resolution_coefficient = opt.model.param.resolution_coefficient
    resolution_coefficient = round(math.pow(resolution_coefficient, avail_resource), 2)

    blocks_args, global_params = efficientnet(blocks_args='default',
                                              activation=opt.model.param.activation,
                                              activation_param=opt.model.param.get('activation_param', {}),
                                              resolution_coefficient=resolution_coefficient,
                                              width_coefficient=opt.model.param.width_coefficient, 
                                              depth_coefficient=opt.model.param.depth_coefficient, 
                                              image_size=opt.model.param.image_size, 
                                              num_classes=opt.model.param.num_classes)
    
    #meaningless = {{choice(['No', 'meaning'])}}
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
    
    logger.info('Validation accuracy : %.2f' % (valid_acc * 100))
    
    return {'loss': valid_loss, 'status': STATUS_OK, 'model': train_handler.model}
    
if __name__ == '__main__':
    conf_name = sys.argv[1]
    with open('./tmp.pickle', 'wb') as f:
        pickle.dump(conf_name, f)
        
    fpath = './results/AutoML'
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    if os.path.isfile('./results/AutoML/cifar100_autoaug_policy.log'):
        os.remove('./results/AutoML/cifar100_autoaug_policy.log')
    
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
                                                         functions=[_get_conf, _logging],
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