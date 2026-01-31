"""
Taken from https://github.com/Bjarten/early-stopping-pytorch
"""
import copy
import numpy as np

import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=5, delta=1e-5, evaluation_mode='loss', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 1e-5 
            evaluation_mode (str): To use score or loss to evaluate model performance
        """
        self.patience = patience
        self.delta = delta
        self.evaluation_mode = evaluation_mode
        self.trace_func = trace_func
        
        self.counter = 0
        self.best_score = -np.Inf
        self.best_loss = np.Inf
        self.best_model = None
        
        self.early_stop = False

    def __call__(self, val_result, model=None):
        if self.patience == -1:
            return
        
        if self.evaluation_mode == 'loss':
            if val_result < self.best_loss:
                self.best_loss = val_result
                if model is not None:
                    self.model = copy.deepcopy(model.state_dict())
                self.counter = 0
                self.early_stop = False
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        elif self.evaluation_mode == 'score':
            if val_result > self.best_score:
                self.best_score = val_result
                if model is not None:
                    self.model = copy.deepcopy(model.state_dict())
                self.counter = 0
                self.early_stop = False
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    
        # elif self.evaluation_mode == 'score':
        #     if (val_score > self.best_score or 
        #         (val_score == self.best_score and 
        #          val_loss < self.best_loss)):
        #         self.best_score = val_score
        #         self.best_loss = val_loss
        #         if model is not None:
        #             self.model = copy.deepcopy(model.state_dict())
        #         self.counter = 0
        #         self.early_stop = False
        #     elif val_score < self.best_score:
        #         self.counter += 1
        #         if self.counter >= self.patience:
        #             self.early_stop = True
           