import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATASET_MAP = {'mnist': datasets.MNIST,
               'cifar10': datasets.CIFAR10,
               'cifar100': datasets.CIFAR100,
               'imagenet': datasets.ImageNet}

def load_dataset(opt):
    train_dataset = DATASET_MAP[opt['data']['dataset']](root=opt['dir']['data'], train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = DATASET_MAP[opt['data']['dataset']](root=opt['dir']['data'], train=False, transform=transforms.ToTensor())
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt['data']['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt['data']['batch_size'], shuffle=False, num_workers=6, pin_memory=True)
    
    return train_loader, test_loader