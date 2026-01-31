import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import gc
import copy
import numpy as np

from script.pytorchtools import EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim

# For reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Trainer(object):
    def __init__(
        self,
        args,
        train_loader,
        val_loader,
        model,
        device
    ):
        self.args = args
        self.model = model
        self.device = device
        
        # Initialize model
        self.model.to(device=self.device)
        
        # Initialize dataloader
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize optimizer
        optim_kwargs = {}
        if 'lr' in self.args:
            optim_kwargs['lr'] = self.args['lr']
        self.optimizer_fn = self.init_optimizer(**optim_kwargs)
        del optim_kwargs
        
        # Initialize loss function
        self.loss_fn = self.init_loss()
        
    def init_optimizer(self, **kwargs):
        return optim.Adam(self.model.parameters(), **kwargs)
    
    def init_loss(self):
        if self.args['loss_func'] == 'mse':
            return nn.MSELoss()
        elif self.args['loss_func'] == 'cross_entropy':
            return nn.CrossEntropy()
            
    def train_one_epoch(self):
        train_loss = []
        for _, data in enumerate(self.train_loader):
            self.optimizer_fn.zero_grad()
            
            # Load data
            X = data.to(device=self.device, dtype=torch.float)
                
            # Train the model
            logits = self.model(X)
            
            # Calculate loss
            loss = self.loss_fn(logits, X)
            loss.backward()
            self.optimizer_fn.step()
            
            # Save results
            train_loss.append(loss.cpu().detach().numpy())
            del X, logits, loss
            gc.collect()
        return np.mean(train_loss)
    
    def train(self):
        epochs = self.args['epochs']
        
        train_loss_history = []
        val_loss_history = []
        
        early_stopping = EarlyStopping()
        
        for e in range(epochs):
            print('Epoch {:d}/{:d}.'.format(e + 1, epochs), end='\r', flush=True)
            
            # Train the model
            self.model.train()
            train_loss = self.train_one_epoch()
            train_loss_history.append(train_loss)
            
            # Evaluate the model
            self.model.eval()
            val_loss = self.evaluate()
            val_loss_history.append(val_loss)
            
            # Print log
            self.print_log(train_loss, val_loss)
            
            # Early stopping
            if self.args['use_early_stopping']:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    break
            
            del train_loss, val_loss
            gc.collect()
        
        logs = {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }
        
        if self.args['use_early_stopping']:
            return early_stopping.model, logs
        else:
            return copy.deepcopy(self.model.state_dict()), logs
    
    @torch.no_grad()
    def evaluate(self):
        val_loss = []
        for _, data in enumerate(self.val_loader):
            # Load data
            X = data.to(device=self.device, dtype=torch.float)
            
            # Train the model
            logits = self.model(X)
            
            # Calculate loss
            loss = self.loss_fn(logits, X)
            
            # Save results
            val_loss.append(loss.cpu().detach().numpy())
            del X, logits, loss
            gc.collect()
        return np.mean(val_loss)
    
    def print_log(self, train_loss, val_loss):
        log_str = 'train loss: {:.5e}\t'.format(train_loss)
        log_str += 'val loss: {:.5e}'.format(val_loss)
        print(log_str, flush=True)


class MaskTrainer(object):
    def __init__(
        self,
        args,
        train_loader,
        val_loader,
        model,
        device
    ):
        self.args = args
        self.model = model
        self.device = device
        
        # Initialize model
        self.model.to(device=self.device)
        
        # Initialize dataloader
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize optimizer
        optim_kwargs = {}
        if 'lr' in self.args:
            optim_kwargs['lr'] = self.args['lr']
        self.optimizer_fn = self.init_optimizer(**optim_kwargs)
        del optim_kwargs
        
        # Initialize loss function
        self.loss_fn = self.init_loss()
        
    def init_optimizer(self, **kwargs):
        return optim.Adam(self.model.parameters(), **kwargs)
    
    def init_loss(self):
        if self.args['loss_func'] == 'mse':
            return nn.MSELoss()
        elif self.args['loss_func'] == 'cross_entropy':
            return nn.CrossEntropy()
            
    def train_one_epoch(self):
        train_loss = []
        for _, data in enumerate(self.train_loader):
            self.optimizer_fn.zero_grad()
            
            # Load data
            X = data.to(device=self.device, dtype=torch.float)
            # X.size() is (n_samples, n_questions, em_dim)
            
            # The last question of X contain social isolation
            # information. Mask that and do the estimation
            # Extract ground truth
            y = X[:, -1, :]
            X[:, -1, :] = 0
                
            # Train the model
            logits = self.model(X)
            loss = self.loss_fn(logits[:, -1, :], y)
            loss.backward()
            self.optimizer_fn.step()
            
            # Save results
            train_loss.append(loss.cpu().detach().numpy())
            del X, y, logits, loss
            gc.collect()
        return np.mean(train_loss)
    
    def train(self):
        epochs = self.args['epochs']
        
        train_loss_history = []
        val_loss_history = []
        
        early_stopping = EarlyStopping()
        
        for e in range(epochs):
            print('Epoch {:d}/{:d}.'.format(e + 1, epochs), end='\r', flush=True)
            
            # Train the model
            self.model.train()
            train_loss = self.train_one_epoch()
            train_loss_history.append(train_loss)
            
            # Evaluate the model
            self.model.eval()
            val_loss = self.evaluate()
            val_loss_history.append(val_loss)
            
            # Print log
            self.print_log(train_loss, val_loss)
            
            # Early stopping
            if self.args['use_early_stopping']:
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    break
            
            del train_loss, val_loss
            gc.collect()
        
        logs = {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }
        
        if self.args['use_early_stopping']:
            return early_stopping.model, logs
        else:
            return copy.deepcopy(self.model.state_dict()), logs
    
    @torch.no_grad()
    def evaluate(self):
        val_loss = []
        for _, data in enumerate(self.val_loader):
            # Load data
            X = data.to(device=self.device, dtype=torch.float)
            
            # The last question of X contain social isolation
            # information. Mask that and do the estimation
            # Extract ground truth
            y = X[:, -1, :]
            X[:, -1, :] = 0
                
            # Train the model
            logits = self.model(X)
            loss = self.loss_fn(logits[:, -1, :], y)
            
            # Save results
            val_loss.append(loss.cpu().detach().numpy())
            del X, y, logits, loss
            gc.collect()
        return np.mean(val_loss)
    
    def print_log(self, train_loss, val_loss):
        log_str = 'train loss: {:.5e}\t'.format(train_loss)
        log_str += 'val loss: {:.5e}'.format(val_loss)
        print(log_str, flush=True)
        
        
@torch.no_grad()
def apply_model(model, dataloader, device):
    model.eval()

    true = []
    logits = []

    for _, data in enumerate(dataloader):
        # Load data
        X = data.to(device=device, dtype=torch.float)
        
        # Apply the model
        out = model(X)

        # Save
        true.append(X.detach().cpu().numpy())
        logits.append(out.detach().cpu().numpy())

        del X, out
    return np.vstack(true), np.vstack(logits)


def generate_mask(size, mask_ratio):
    n_samples, channels, seq_len = size
    
    mask_len = int(mask_ratio * seq_len)

    # Initialize a mask tensor with all ones
    mask = torch.ones((n_samples, channels, seq_len), dtype=torch.bool)

    for i in range(n_samples):
        # Randomly choose `mask_len` unique positions to mask
        mask_positions = torch.randperm(seq_len)[:mask_len]
        mask[i, :, mask_positions] = False  # Apply across all channels

    return mask