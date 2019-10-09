import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def top1_pred(outputs, half=False):
    max_logit, pred = torch.max(outputs, 1)
    if half:
        pred = pred.half()
    return max_logit, pred


class EarlyExitInspector():
    def __init__(self, Network, device, pred=None, exiting_flops_ratio=0, exit_condition=0, half=False):
        self.Network = Network.to(device).eval()
        self.pred = pred if (pred is not None) else top1_pred 
        self.device = device
        self.exiting_flops_ratio = exiting_flops_ratio
        self._condition_setter(exit_condition)
        self.half = half
        
        if half:
            self.Network.half()
        
    
    def predictor(self, dataloaders, dataset_sizes, phase, exit_condition=(None)):        
        self._condition_setter(exit_condition)
        
        exit_correct, final_correct = 0, 0
        exit_count, final_count = 0, 0
        size = dataset_sizes[phase]
                 
        with torch.no_grad():
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if self.half:
                    inputs, labels = inputs.half(), labels.half()
                
                # Inference from Network
                outputs, mark = self.Network(inputs)
                _, pred = self.pred(outputs, self.half)
                
                exit_mark, final_mark = (mark == 1), (mark == 0) 

                exit_correct += (pred[exit_mark] == labels[exit_mark]).sum().item()
                final_correct += (pred[final_mark] == labels[final_mark]).sum().item()
                
                exit_count += exit_mark.sum().item()
                final_count += final_mark.sum().item()
                
        total_acc = round(((exit_correct + final_correct) / size), 4)
        exit_acc = -1 if exit_count == 0 else round((exit_correct/exit_count), 4)
        final_acc = -1 if final_count == 0 else round((final_correct/final_count), 4)
        
        exit_ratio = round((exit_count/size), 4)
        final_ratio = round((final_count/size), 4)
        
        score = self._flop_checker(exit_ratio, final_ratio, self.exiting_flops_ratio)
        
        return total_acc, (exit_acc, final_acc), (exit_ratio, final_ratio), score
    
    
    def score_validator(self, dataloaders, dataset_sizes, condition_range=(0.5, 0.99), grid=0.01, phase='valid'):
        baseline_acc1 = self._baseline_setter(dataloaders, dataset_sizes, phase, mode='total')
        baseline_acc2 = self._baseline_setter(dataloaders, dataset_sizes, phase, mode='exit')
        total_acc_list, exit_ratio_list, final_ratio_list, score_list = [], [], [], []
        exit_acc_list, final_acc_list = [], []
        start, end = condition_range
        
        if (end - start) == 0:
            condition_list = [end]
        else:
            condition_list = [start + x*grid for x in range(int((end-start)/grid))]

        for condition_value in condition_list:
            self._condition_setter(condition_value)

            total_acc, (exit_acc, final_acc), (exit_ratio, final_ratio), score = \
            self.predictor(dataloaders, dataset_sizes, phase)

            total_acc_list.append(total_acc)
            exit_acc_list.append(exit_acc)
            final_acc_list.append(final_acc)

            exit_ratio_list.append(exit_ratio)
            final_ratio_list.append(final_ratio)

            score_list.append(score)
                
        return total_acc_list, (exit_acc_list, final_acc_list), condition_list, \
    (exit_ratio_list, final_ratio_list), score_list, baseline_acc1, baseline_acc2
        
        
    def _flop_checker(self, exit_ratio, final_ratio, exit_flops):
        exit_score = exit_flops * exit_ratio
        final_score = 1 * final_ratio
        
        total_score = exit_score + final_score
        total_score = round(total_score, 4)
        return total_score
    
    
    def _condition_setter(self, exit_condition):
        if exit_condition is not None:
            self.Network._exit_cond_test.thres = exit_condition

    
    def _baseline_setter(self, dataloaders, dataset_sizes, phase, mode='total'):
        if mode == 'total':
            self._condition_setter(1)
        elif mode == 'exit':
            self._condition_setter(0)
        
        correct = 0
        size = dataset_sizes[phase]
                 
        with torch.no_grad():
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if self.half:
                    inputs, labels = inputs.half(), labels.half()
                
                # Inference from Network
                outputs, mark = self.Network(inputs)
                _, pred = self.pred(outputs, self.half)
                
                correct += (pred == labels).sum().item()

        baseline_acc = round((correct / size), 4)

        return baseline_acc

    
    def _logit_dist_inspector(self, dataloaders, dataset_sizes, phase, exit_condition=0):
        self._condition_setter(exit_condition)
        
        max_logit_co, max_logit_inco = [], []
        entropy_co, entropy_inco = [], []
        
        size = dataset_sizes[phase]
        
        with torch.no_grad():
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if self.half:
                    inputs, labels = inputs.half(), labels.half()
                    
                outputs, _ = self.Network(inputs)
                _, preds = self.pred(outputs, self.half)
                
                soft_out = F.softmax(outputs, dim=1)
                
                top10_out, _ = torch.topk(soft_out, 10)
                
                entropy = top10_out * -top10_out.log()
                entropy = entropy.sum(dim=1)

                # Get correct tensor
                correct_tensor = preds == labels
                incorrect_tensor = preds != labels

                co_values, _ = soft_out[correct_tensor].max(dim=1)
                co_values = co_values.tolist()
                inco_values, _ = soft_out[incorrect_tensor].max(dim=1)
                inco_values = inco_values.tolist()

                co_entropy = entropy[correct_tensor]
                co_entropy = co_entropy.tolist()
                inco_entropy = entropy[incorrect_tensor]
                inco_entropy = inco_entropy.tolist()

                max_logit_co += co_values
                max_logit_inco += inco_values

                entropy_co += co_entropy
                entropy_inco += inco_entropy

        return (max_logit_co, max_logit_inco), (entropy_co, entropy_inco)
    

def plotter(total_acc_list, exit2_acc_list, final_acc_list, \
            condition2_list, exit2_ratio_list, score_list, \
            baseline_acc1, baseline_acc2, max_logit_co, max_logit_inco, entropy_co, entropy_inco, name='test'):
    #fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 15))
    fig, ((ax1, ax2), (ax5, ax6)) = plt.subplots(2, 2, figsize=(10, 12))
    
    fig.suptitle(name, fontsize=20)

    ax1.set(title='Exiting Ratio vs Total Accuracy', xlabel='Exit Ratio', ylabel='Accuracy')
    ax1.axhline(y=baseline_acc1, color='black', linestyle='--', alpha=0.6)
    ax1.plot(exit2_ratio_list, total_acc_list, color='r', marker='o', markersize=4)
    ax1.set_xlim(0.1, 0.60)
    ax1.set_ylim(0.76, 0.81)
    ax1.legend(['baseline(%0.4f)' % baseline_acc1, 'trade-off'])
    ax1.grid()

    ax2.set(title='FlopsScore vs Total Accuracy', xlabel='FlopsScore', ylabel='Total Accuracy')
    ax2.axhline(y=baseline_acc1, color='black', linestyle='--', alpha=0.6)
    ax2.plot(score_list, total_acc_list, color='blue', alpha=0.8, marker='o', markersize=4)
    ax2.scatter(1, baseline_acc1, color='black')
    ax2.set_xlim(0.4, 0.9)
    ax2.set_ylim(0.76, 0.81)
    ax2.legend(['baseline(%0.4f)' % baseline_acc1, 'trade-off'])
    ax2.grid()
    
    """
    ax3.set(title='Exiting Ratio vs Large Accurcy', xlabel='Exit Ratio', ylabel='Exit Accuracy')
    ax3.axhline(y=baseline_acc1, color='black', linestyle='--', alpha=0.6)
    ax3.plot(exit2_ratio_list, final_acc_list, marker='.')
    ax3.set_xlim(0.1, 0.6)
    ax3.set_ylim(0.5, 0.8)
    ax3.legend(['baseline(%0.4f)' % baseline_acc1, 'trade-off'])
    ax3.grid()
    
    ax4.set(title='Exiting Ratio vs Exit Accuracy (Base: %0.4f)'%baseline_acc2, xlabel='Exit Ratio', ylabel='Exit Accuracy')
    ax4.plot(exit2_ratio_list, exit2_acc_list, marker='.')
    ax4.set_xlim(0.1, 0.6)
    ax4.set_ylim(0.7, 1.0)
    ax4.legend(['trade-off'])
    ax4.grid()
    """
    
    ax5.set(title='Max softmax outputs(Exit Module)', xlabel='softmax max value', ylabel='data count')
    ax5.hist(max_logit_co, color='brown', bins=[x*0.05 for x in range(21)], alpha=0.5)
    ax5.hist(max_logit_inco, color='gray', bins=[x*0.05 for x in range(21)], alpha=0.5)
    ax5.set_ylim(0, 2500)
    ax5.legend(['correct samples', 'incorrect samples'])
    ax5.grid()
    
    ax6.set(title='Entropy of Top10 softmax outputs(Exit Module)', xlabel='Top10 softmax entropy', ylabel='data count')
    ax6.hist(entropy_co, color='brown', bins=[x*0.05 for x in range(45)], alpha=0.5)
    ax6.hist(entropy_inco, color='gray', bins=[x*0.05 for x in range(45)], alpha=0.5)
    ax6.set_ylim(0, 1200)
    ax6.legend(['correct samples', 'incorrect samples'])
    ax6.grid()
    
    fig.show()
    fig.savefig('./results/inspection/%s.png' % name)
    print('%s inspection graph is saved' % name)