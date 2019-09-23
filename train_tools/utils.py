import os
import json
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['ConfLoader', 'model_saver', 'result_logger', 'plotter', 'EarlyStopping']


class ConfLoader:
    """ Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """
    class DictWithAttributeAccess(dict):
        """ This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key']
        """
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()
        
    def __load_conf(self):
        with open(self.conf_name, 'r') as conf:
            opt = json.load(conf, object_hook = lambda dict: self.DictWithAttributeAccess(dict))
        
        return opt
    
    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)
        
        return opt


def directory_setter(path='./results', make_dir=False):
    if not os.path.exists(path) and make_dir:
        os.makedirs(path) # make dir if not exist
        print('directory %s is created' % path)
        
    if not os.path.isdir(path):
        raise NotADirectoryError('%s is not valid. set make_dir=True to make dir.' % path)
        

def model_saver(trained_model, initial_model, model_info=None, result_path='./results', model_name='model', make_dir=True):
    save_path = result_path + '/trained_models/' + model_name
    directory_setter(save_path, make_dir)


    """
    saves model weights and model description.
    """
    # save model description if exsists
    info_path = os.path.join(save_path, 'model_info.json')
    trained_model_path = os.path.join(save_path, 'trained_model.pth')
    initial_model_path = os.path.join(save_path, 'initial_model.pth')
    
    if model_info:
        with open(info_path, 'w') as fp:
            json.dump(model_info, fp)
    
    wts = trained_model.state_dict()
    torch.save(wts, trained_model_path)
    print('trained model saved as %s' % trained_model_path)
    
    ini_wts = initial_model
    torch.save(initial_model, initial_model_path)
    print('initial model saved as %s' % initial_model_path)
    
    

def result_logger(result_dict, epoch_num, result_path='./results', model_name='model', make_dir=True):
    """
    saves train results as .csv file
    """
    log_path = result_path + '/logs'
    file_name = model_name + '_results.csv'
    directory_setter(log_path, make_dir)
    save_path = os.path.join(log_path, file_name)
    header = ','.join(result_dict.keys()) + '\n'
    
    with open(save_path, 'w') as f:
        f.write(header)
        for i in range(epoch_num):
            row = []
            
            for item in result_dict.values():
                if type(item) is not list:
                    row.append('')

                elif item[i][1] is not None:
                    assert item[i][0] == (i+1), 'Not aligned epoch indices'
                    elem = round(item[i][1], 5)
                    row.append(str(elem))
                    
                else:
                    row.append('')
            
            # write each row
            f.write(','.join(row) + '\n')
            
        sep = len(result_dict.keys()) - 2
        f.write(','*sep + '%0.5f, %0.5f'% (result_dict['test_loss'], result_dict['test_acc']))
        
    print('results are logged at: \'%s' % save_path)    
    
    
def plotter(train, valid, test, mode, result_path='./results', model_name='model', make_dir=True, plot_freq=0.05):
    
    """
    plots loss or accuracy graph for train/valid logs, and saves as .png file.
    train, valid : list of tuples. ex) [(0, 0.08), (1, 0.56)...]
    test : float
    mode : 'loss' or 'accuracy'
    """
    save_path = result_path + '/graphs'
    directory_setter(save_path, make_dir)
    fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')    
    plt.plot(*zip(*train), 'k', linestyle='-', label='train_%s'%mode)
    plt.plot(*zip(*valid), 'r^', linestyle='--', label='valid_%s'%mode)
    plt.plot(len(train), test, 'bo', label='test_{}({:0.4f})'.format(mode, test))
    plt.xlabel('Epoch')
    plt.ylabel(mode)
    plt.legend()
    plt.grid()
    plt.xticks([x+1 for x in range(len(train)) if (x+1) % (len(train)*plot_freq) == 0])
    fname = os.path.join(save_path, '{}_{}_{}.png'.format(model_name, 'graph', mode))
    plt.savefig(fname)
    print('{} plot saved at {}'.format(mode, fname))


class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        self.best_model = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss