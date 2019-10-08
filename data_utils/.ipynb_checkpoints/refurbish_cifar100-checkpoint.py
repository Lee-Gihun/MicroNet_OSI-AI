"""
Refurs mislabeled data and uncertain data from cifar-100 dataset.
The dataset is obtained using <Prestopping: How Does Early Stopping Help Generalization Against Label Noise?> under ICLR 2020 Review.
The dataset is obtained from 7 runs using DenseNet-40 and chose best result.


<Reference>

Refurred Dataset is offered by anonymous authors.
[1] Anonymous Authors, <Prestopping: How Does Early Stopping Help Generalization Against Label Noise?>
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os, wget

__all__ = ['RefurbishCIFAR100']

class RefurbishCIFAR100(Dataset):
    def __init__(self, root, refur=False, use_certain=False, transforms=None, target_transforms=None, download=True, verbose=True):
        root = root + '/refurbished_cifar100'
        self._download(root)
        self.dataset, self.certain_indices = self._data_mapper(root, refur, use_certain, verbose)
        self.use_certain = use_certain
        self.transforms, self.target_transforms = transforms, target_transforms
        
    
    def _data_mapper(self, root, refur, use_certain, verbose):
        raw_data = np.fromfile(root+'/train_data.bin', dtype='uint8')
        raw_data = raw_data.reshape(50000, 4 + 4 + 3*32*32) # idx(4), label(4), img(3*32*32)
        df_label = pd.read_csv(root+'/refurbishment.csv', names=['index', 'refurred'])
        
        dataset, certain_indices = [], []
        refur_count = 0
        dummy_flag = True
        
        for index in range(50000):
            data = raw_data[index][8:].reshape(3, 32, 32)
            data = data.transpose((1, 2, 0))
            label = self._bytes_to_int(raw_data[index][4:8])
            assert(index == self._bytes_to_int([index][:4]))
            
            if refur:
                temp = df_label.loc[df_label['index']==index]['refurred'].values[0]
                if temp != -1:
                    if label != temp:
                        refur_count += 1
                        
                    label = temp
                    certain_indices.append(index)
                else:
                    if dummy_flag:
                        if verbose:
                            print('\nsample index %d is uncertain but included to meet ghost batch size.\n' % index)
                        certain_indices.append(index)
                        dummy_flag = False
                    
            dataset.append((data, label))
            
            if (index+1) % 10000 == 0 and verbose:
                print('index %d of refurred set has processed' % (index+1))
        
        uncertain = (50000-len(certain_indices)) if use_certain else 0
        if verbose:
            print('----------------------------------------------------------------')
            print('%d samples have refurred and %d uncertain samples have excluded\n' % (refur_count, uncertain))
            
        return dataset, certain_indices
            
        
    def _bytes_to_int(self, bytes_array):
        result = 0
        for b in bytes_array:
            result = result * 256 + int(b)
        return result
    
    
    def __getitem__(self, index):
        if self.use_certain:
            img, target = self.dataset[self.certain_indices[index]]
        else:
            img, target = self.dataset[index]
        
        img = Image.fromarray(img, 'RGB')
            
        if self.transforms is not None:
            img = self.transforms(img)
            
        if self.target_transforms is not None:
            target = self.target_transforms(target)
            
        return img, target

    
    def __len__(self):
        if self.use_certain:
            return len(self.certain_indices)
        else:
            return len(self.dataset)
        
    def _download(self, root):
        if not os.path.isdir(root):
            os.makedirs(root)
            
        refurbish_path = root+'/refurbishment.csv'
        train_bin_path = root+'/train_data.bin'
        
        refurbish_exist = os.path.isfile(refurbish_path)
        train_bin_exist = os.path.isfile(train_bin_path)
        
        if not refurbish_exist:
            url = 'https://www.dropbox.com/s/izxfv9fko4hds7u/refurbishment.csv?dl=1'
            print('\nrefurbished label downloading...')
            wget.download(url, refurbish_path)
            
        
        if not train_bin_exist:
            print('\ntrain data binary file downloading...')
            url = 'https://www.dropbox.com/s/picdidpo5aziqcf/train_data.bin?dl=1'
            wget.download(url, train_bin_path)
            
        print('data and labels are setted! \n')
        