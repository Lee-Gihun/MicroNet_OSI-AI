import torch
import torch.nn as nn
from torchvision import datasets, models
from torch.utils.data import Subset
import torchvision.transforms as transforms
import random

from .autoaugment import *

__all__ = ['cifar_10_setter', 'cifar_100_setter']

def cifar_10_setter(batch_size=128, valid_size=5000, pin_memory=False, num_workers=4, root='./data/cifar', download=True, fixed_valid=True, autoaugment=False):
    if fixed_valid:
        random.seed(2019)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.4914, 0.4822, 0.4465]
    stdv = [0.2023, 0.1994, 0.2010]
    train_transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ]
    if autoaugment:
        train_transform_list.insert(0, CIFAR10Policy())

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    batch_size = batch_size
    valid_size = valid_size
    
    # Datasets
    cifar10_train_set = datasets.CIFAR10(root, train=True, transform=train_transforms, download=download) # train transform applied
    cifar10_valid_set = datasets.CIFAR10(root, train=True, transform=test_transforms, download=download) # test transform applied
    valid_list = random.sample(range(0, len(cifar10_train_set)), valid_size)
    train_list = [x for x in range(len(cifar10_valid_set))]
    train_list = list(set(train_list)-set(valid_list))

    cifar10_train_set = Subset(cifar10_train_set, train_list)
    cifar10_valid_set = Subset(cifar10_valid_set, valid_list)
    cifar10_test_set = datasets.CIFAR10(root, train=False, transform=test_transforms, download=download)

    train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(cifar10_valid_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(cifar10_test_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(cifar10_train_set), 'valid' : len(cifar10_valid_set), 'test' : len(cifar10_test_set)}
    
    return dataloaders, dataset_sizes


def cifar_100_setter(batch_size=128, valid_size=5000, pin_memory=False, num_workers=4, root='./data/cifar', download=True, fixed_valid=True, autoaugment=False, aug_policy='cifar100'):
    if fixed_valid:
        random.seed(2019)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv)
    ]
    if autoaugment:
        if aug_policy == 'cifar10':
            train_transform_list.insert(0, CIFAR10Policy())
        elif aug_policy == 'cifar100':
            train_transform_list.insert(0, CIFAR100Policy())
        elif aug_policy == 'cifar100_2':
            train_transform_list.insert(0, CIFAR100Policy2())
        elif aug_policy == 'cifar100_3':
            train_transform_list.insert(0, CIFAR100Policy3())
        else:
            train_transform_list.insert(0, aug_policy)

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    
    batch_size = batch_size
    valid_size = valid_size
    
    # Datasets
    cifar100_train_set = datasets.CIFAR100(root, train=True, transform=train_transforms, download=download)
    cifar100_valid_set = datasets.CIFAR100(root, train=True, transform=test_transforms, download=download)
    valid_list = random.sample(range(0, len(cifar100_train_set)), valid_size)
    train_list = [x for x in range(len(cifar100_valid_set))]
    train_list = list(set(train_list)-set(valid_list))

    cifar100_train_set = Subset(cifar100_train_set, train_list)
    cifar100_valid_set = Subset(cifar100_valid_set, valid_list)
    cifar100_test_set = datasets.CIFAR100(root, train=False, transform=test_transforms, download=download)

    train_loader = torch.utils.data.DataLoader(cifar100_train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(cifar100_valid_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(cifar100_test_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(cifar100_train_set), 'valid' : len(cifar100_valid_set), 'test' : len(cifar100_test_set)}
    
    return dataloaders, dataset_sizes