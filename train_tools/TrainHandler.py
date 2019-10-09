import torch
import numpy as np
from .utils import *
from .topk_acc import *

import copy
import time
import os

__all__ = ['TrainHandler']

class TrainHandler():
    def __init__(self, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, device=None, path='./results', mixup=False, alpha=1.0, precision=32, prune=False, early_exit=False):
        # If device is None, get default device
        if device == None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.device_idx = self.__get_device_idx(self.device)

        self.model         = model.to(self.device)
        self.dataloaders   = dataloaders
        self.dataset_sizes = dataset_sizes
        self.optimizer     = optimizer
        self.criterion     = criterion
        self.prediction    = lambda outputs : torch.max(outputs, 1)[1]
        self.scheduler     = scheduler if scheduler != None else None
        self.path          = path
        self.num_epochs    = 0
        self.mixup         = mixup
        self.alpha         = alpha
        self.prune         = prune
        self.early_exit    = early_exit
        
        # Set model precision if precision is specified
        self.precision = precision
        if precision == 16:
            self.model.half()
                    
        # Initialize result log dictionary
        self.result_log = self.__init_result_log() 

        # Save initial state to the file
        self.init_states = self.__save_init_states(self.model, self.optimizer, self.scheduler)

        # Initialize model info dictionary
        self.model_info = self.__init_model_info(self.model, self.dataloaders, self.optimizer, self.scheduler)
                
        # Set default file name to save results
        self.name = self.model_info['model']
        
    def __get_device_idx(self, device):
        if self.device == 'cpu':
            device_idx = None
        else:
            if isinstance(device, str):
                device_idx = int(device[-1])
            elif isinstance(device, torch.device):
                device_idx = device.index

        return device_idx

    def __save_init_states(self, model, optimizer, scheduler):
        init_states = {}

        init_states['model']     = copy.deepcopy(model.state_dict())
        init_states['optimizer'] = copy.deepcopy(optimizer.state_dict())
        if scheduler != None:
            init_states['scheduler'] = copy.deepcopy(scheduler.state_dict())

        return init_states

    def __init_result_log(self):
        result_log = {}

        # Initalize result_log dictionary
        result_log['train_loss'] = []
        result_log['valid_loss'] = []
        result_log['train_acc']  = []
        result_log['valid_acc']  = []
        result_log['test_loss']  = 0.0
        result_log['test_acc']   = 0.0

        return result_log

    def __init_model_info(self, model, dataloaders, optimizer, scheduler):
        model_info = {}

        model_info['model']          = model.__class__.__name__
        model_info['dataset_info']   = dataloaders['test'].dataset.__class__.__name__
        model_info['optimizer']      = copy.deepcopy(optimizer.state_dict())
        model_info['scheduler']      = copy.deepcopy(scheduler.state_dict()) if scheduler != None else None
        model_info['performance']    = None
        model_info['memo']           = None

        return model_info

    def _get_current_device(self):
        if self.device == 'cpu':
            return self.device

        return torch.cuda.get_device_name(self.device_idx)

    def _update_result_dict(self, log_freq, train_losses, valid_losses, train_accs, valid_accs):
        self.result_log['train_loss'] += train_losses[-log_freq:]
        self.result_log['valid_loss'] += valid_losses[-log_freq:]
        self.result_log['train_acc']  += train_accs[-log_freq:]
        self.result_log['valid_acc']  += valid_accs[-log_freq:]

    def _print_train_stat(self, epoch, num_epochs, epoch_elapse, train_loss, train_acc, valid_loss, valid_acc, learning_rate, early_exit=False, valid_early_exits=None):
        print('[Epoch {}/{}] Elapsed {}s/it'.format(epoch, num_epochs, epoch_elapse))
        print('[{}] Loss - {:.4f}, Acc - {:2.2f}%, Learning Rate - {}'.format('Train', train_loss, train_acc * 100, learning_rate))
        print('[{}] Loss - {:.4f}, Acc - {:2.2f}%'.format('Valid', valid_loss, valid_acc * 100))
        if early_exit:
            valid_early_exit = valid_early_exits[-1][1]
            print('[{}] Early_Exit percentage - {:.2f}'.format('Valid', valid_early_exit * 100))
        #print('Memory Usage: {:.2f} MB'.format(torch.cuda.memory_allocated(self.device_idx) / 1024 / 1024))
        #print('Memory Cached: {:.2f} MB'.format(torch.cuda.memory_cached(self.device_idx) / 1024 / 1024))
        #print('Max Memory Usage: {:.2f} MB'.format(torch.cuda.max_memory_allocated(self.device_idx) / 1024 / 1024))
        #print('Max Memory Cached: {:.2f} MB'.format(torch.cuda.max_memory_cached(self.device_idx) / 1024 / 1024))
        print()
        print('-' * 50)

    def _get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def __mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).to(self.device)
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def __mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _epoch_phase(self, phase):
        if phase == 'train':
            self.model.train()
            if self.scheduler != None:
                self.scheduler.step()
        else:
            self.model.eval()

        running_loss = 0.0
        running_correct = 0.0
        if self.early_exit:
            early_exit = 0.0

        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            if self.precision == 16:
                inputs = inputs.type(torch.HalfTensor)
                inputs = inputs.to(self.device)
                        
            # Zero out parameter gradients
            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                # Forward pass
                if not self.mixup or phase != 'train':
                    outputs = self.model(inputs)
                    preds = self.prediction(outputs)
                    loss = self.criterion(outputs, labels)
                else:
                    inputs, labels_a, labels_b, lam = self.__mixup_data(inputs, labels, self.alpha)
                    outputs = self.model(inputs)
                    preds = self.prediction(outputs)
                    loss = self.__mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                    
                if self.early_exit and phase != 'train':
                    preds, exit_mark = preds[0], preds[1]
                    early_exit += torch.sum(exit_mark == 1)
                    
                if phase == 'train':
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    if self.prune:
                        self.model.set_masks(None, prev=True)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)

        if self.dataset_sizes[phase] == 0:
            if self.early_exit and phase != 'train':
                return 0.0, 0.0, 0.0
            
            return 0.0, 0.0
            
        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc  = (running_correct.double() / self.dataset_sizes[phase]).item()
        if self.early_exit and phase != 'train':
            epoch_early_exit = (early_exit.double() / self.dataset_sizes[phase]).item()
            return epoch_loss, epoch_acc, epoch_early_exit

        return epoch_loss, epoch_acc

    def _test_phase(self, topk):        
        self.model.eval()
        avg_meter = [AverageMeter('top{} accuracy'.format(k)) for k in topk]
        losses = AverageMeter('loss')

        running_loss = 0.0
        running_correct = 0.0
        if self.early_exit:
            early_exit = 0.0

        for inputs, labels in self.dataloaders['test']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
                        
            if self.precision == 16:
                inputs = inputs.type(torch.HalfTensor)
                inputs = inputs.to(self.device)
                
            # Zero out parameter gradients
            self.optimizer.zero_grad()

            with torch.no_grad():
                # Inference
                outputs = self.model(inputs)
                preds = self.prediction(outputs)
                loss = self.criterion(outputs, labels)
                
                if self.early_exit:
                    preds, exit_mark = preds[0], preds[1]
                    early_exit += torch.sum(exit_mark == 1)
                    
                    outputs = outputs[0]

                # Get topk accuracy
                topk_acc = accuracy(outputs.data, labels.data, topk=topk)
                losses.update(loss.item(), inputs.size(0))
                for i, topk_meter in enumerate(avg_meter):
                    topk_meter.update(topk_acc[i], inputs.size(0))

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)

        test_loss = running_loss / self.dataset_sizes['test']
        test_acc  = (running_correct.double() / self.dataset_sizes['test']).item()
        topk_avg = [topk.avg for topk in avg_meter]
        if self.early_exit:
            test_early_exit = (early_exit.double() / self.dataset_sizes['test']).item()
            return test_loss, test_acc, topk_avg, test_early_exit

        return test_loss, test_acc, topk_avg


    def train_model(self, num_epochs=200, valid_freq=1, log_freq=1, print_freq=1, early_stop=False, patience=10, verbose=False):
        since = time.time()
        train_losses, valid_losses, train_accs, valid_accs = [], [], [], []
        valid_early_exits = [(0, 0.0)]
        self.num_epochs = num_epochs

        if early_stop:
            early_stopping = EarlyStopping(patience, verbose)
            
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf

        print('')
        print('=' * 50)
        print('Train start on device: {}'.format(self._get_current_device()))
        print('=' * 50, '\n')
        for epoch in range(1, self.num_epochs + 1):
            # Each epoch have training and validation phases
            epoch_start = time.time()
            train_loss, train_acc = self._epoch_phase('train')
            if epoch % valid_freq == 0:
                if self.early_exit:
                    valid_loss, valid_acc, valid_early_exit = self._epoch_phase('valid')
                else:
                    valid_loss, valid_acc = self._epoch_phase('valid')
                    
            epoch_elapse = round(time.time() - epoch_start, 3)
            
            # Save loss and accuracy statistics
            train_losses.append((epoch, train_loss))
            train_accs.append((epoch, train_acc))
                
            if epoch % valid_freq == 0:
                valid_losses.append((epoch, valid_loss))
                valid_accs.append((epoch, valid_acc))
                if self.early_exit:
                    valid_early_exits.append((epoch, valid_early_exit))
                
            # Update the best validation accuracy
            if valid_loss <= best_loss:
                best_loss = valid_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                
            # Write log based on log_freq
            if epoch % log_freq == 0:
                self._update_result_dict(log_freq, train_losses, valid_losses, train_accs, valid_accs)
                result_logger(self.result_log, epoch, self.path, self.name)

            # Print stat based on print_freq
            if epoch % print_freq == 0:
                self._print_train_stat(epoch, self.num_epochs, epoch_elapse, train_loss, train_acc, valid_loss, valid_acc, self._get_learning_rate(), early_exit=self.early_exit, valid_early_exits=valid_early_exits)

            if early_stop:
                early_stopping(valid_loss, self.model)
                if early_stopping.early_stop:
                    print('Early Stopping')
                    best_model_wts = early_stopping.best_model
                    break
                
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))
        print('=' * 50, '\n')

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        return train_losses, valid_losses, train_accs, valid_accs

    def test_model(self, topk=(1,), plot_freq=0.1, device=None, pretrained=False):
        if (device != None) and (device != self.device):
            self.device = device
            self.model.to(device)

        if self.early_exit:
            test_loss, test_acc, topk_avg, test_early_exit = self._test_phase(topk)
        else:
            test_loss, test_acc, topk_avg = self._test_phase(topk)

        # Test result
        print('[{}] Loss - {:.4f}, Acc - {:2.2f}%'.format('Test', test_loss, test_acc * 100))
        for k, k_avg in zip(topk, topk_avg):
            print('[{}] Top {} accuracy: {:2.2f}% '.format('Test', k, round(k_avg.item(), 2)))
        print()
        
        if not pretrained:
            self.result_log['test_loss'] = test_loss
            self.result_log['test_acc']  = test_acc
            result_logger(self.result_log, self.num_epochs, self.path, self.name)
        
            plotter(self.result_log['train_loss'], self.result_log['valid_loss'], self.result_log['test_loss'], 'loss', self.path, self.name, True, plot_freq)
            plotter(self.result_log['train_acc'], self.result_log['valid_acc'], self.result_log['test_acc'], 'accuracy', self.path, self.name, True, plot_freq)

            # Save test result to model info
            self.model_info['performance'] = {'loss': test_loss, 'accuracy': test_acc}
            model_saver(self.model, self.init_states['model'], self.model_info, self.path, self.name)
            
            self.result_log = self.__init_result_log()
            self.model_info['performance'] = None

        if self.early_exit:
            return test_loss, test_acc, test_early_exit
        
        return test_loss, test_acc

    def set_name(self, name):
        self.name = name
    
    def set_criterion(self, criterion):
        self.criterion = criterion
    
    def set_prediction(self, prediction):
        self.prediction = prediction

    def set_memo(self, memo):
        """
        memo: String. Description of current model
        """
        self.model_info['memo'] = memo

    def reset_model_state(self):
        self.model.load_state_dict(self.init_states['model'], strict=False)

        # Load saved initial state file
        #self.model.load_state_dict(self.init_states['model'])
        self.optimizer.load_state_dict(self.init_states['optimizer'])
        if self.scheduler != None:
            self.scheduler.load_state_dict(self.init_states['scheduler'])
