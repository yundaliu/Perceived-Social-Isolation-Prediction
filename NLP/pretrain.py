####################################################
# Auto-Encoder
# Input: 18 questions for each survey
# Output: 1 latent embedding
####################################################
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import yaml
import itertools

import torch

from script.directory import social_isolation_v2_dir
import script.features as script_features
from script.extract_text_embeddings import convert_text_to_embeddings
from script import dataset
from script import trainer as trainer_class
from script import network

# Define directories
output_data_dir = os.path.join(
    social_isolation_v2_dir,
    'out_pretrain',
    'data'
)

output_model_dir = os.path.join(
    '/scratch3',
    'workspace',
    'yundaliu_umass_edu-simple'
)

config_dir = os.path.join(
    social_isolation_v2_dir,
    'config'
)


def main():
    ########################################
    # Load parameters
    ########################################
    args = parse_arguments()
    print(f'PREV:                      {args.prev}', flush=True)
    print(f'TASK_ID:                   {args.task_id}', flush=True)
    print(f'DATA_CONFIG_NUMBER:        {args.data_config_number}', flush=True)
    print(f'NETWORK_CONFIG_NUMBER:     {args.network_config_number}', flush=True)
    print(f'TRAIN_CONFIG_NUMBER:       {args.train_config_number}', flush=True)
    print('\n', flush=True)
    
    ########################################
    # Load config files
    ########################################
    # Data config file
    args.data_config = load_config(
        os.path.join(
            config_dir,
            'Data',
            'Data_{:02d}.yaml'.format(args.data_config_number))
    )
    print('Data Config', flush=True)
    print_config(args.data_config)

    # Network config file
    args.network_config = load_config(
        os.path.join(
            config_dir,
            'Network',
            'Network_{:02d}.yaml'.format(args.network_config_number))
    )
    print('Network Config', flush=True)
    print_config(args.network_config)
    
    # Train config file
    args.train_config = load_config(
        os.path.join(
            config_dir,
            'Train',
            'Train_{:02d}.yaml'.format(args.train_config_number))
    )
    print('Train Config', flush=True)
    print_config(args.train_config)

    ########################################
    # Load dataset
    ########################################
    print('Initializing dataset...', flush=True)
    train_loader, val_loader, test_loader = dataset.dataset_factory(args)
    
    ########################################
    # Save
    ########################################
    folder = 'Data_{:02d}_Network_{:02d}_Train_{:02d}_v1'.format(
        args.data_config_number,
        args.network_config_number,
        args.train_config_number,
    )

    ########################################
    # Fine tuning
    ########################################
    print('Fine tuning...', flush=True)
    device = check_cuda()
    best_model, best_param, logs = fine_tuning(
        args,
        train_loader,
        val_loader,
        device,
        folder = folder
    )
    
    ########################################
    # Test model
    ########################################
    print('Applying model...', flush=True)
    true, logits = trainer_class.apply_model(
        best_model,
        test_loader,
        device
    )
    
    ########################################
    # Save
    ########################################
    # Save data
    output = {
        # 'y_true': true,
        # 'y_scores': logits,
        'best_param': best_param,
        'logs': logs,
        'test_loss': np.mean(np.square(true - logits))
    }
    out_folder = os.path.join(output_data_dir, folder)
    os.makedirs(out_folder, exist_ok=True)
    out_file = '{:02d}.npz'.format(args.task_id)
    np.savez(os.path.join(out_folder, out_file), **output)
    del output, out_folder, out_file
    
    # Save model
    out_folder = os.path.join(output_model_dir, folder)
    os.makedirs(out_folder, exist_ok=True)
    out_file = '{:02d}.pt'.format(args.task_id)
    torch.save(best_model.state_dict(), os.path.join(out_folder, out_file))
    del out_folder, out_file
    
    print('DONE.', flush=True)
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prev',
        type=int,
        default=0
    )
    parser.add_argument(
        '--task_id',
        required=True,
        type=int
    )
    parser.add_argument(
        '--data_config_number',
        required=True,
        type=int,
        default=1
    )
    parser.add_argument(
        '--network_config_number',
        type=int,
        default=1
    )
    parser.add_argument(
        '--train_config_number',
        type=int,
        default=1
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as stream:
        return yaml.safe_load(stream)
    
    
def print_config(config_dict):
    for key, val in config_dict.items():
        print(key, '\t', val, flush=True)
    print('\n', flush=True)
    
    
def check_cuda(verbose=True):
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
        if verbose:
            print(f'CUDA VERSION: {torch.backends.cudnn.version()}')
            print(f'Using Device: {device}')
        return device
    else:
        raise EnvironmentError('Could not access GPU.')
        

def fine_tuning(args, train_loader, val_loader, device, folder):
    # Save best results
    best_model = None
    best_param = None
    best_val_result = np.inf
    logs = dict()
    
    # Process parameters
    param_grid = args.train_config['param_grid']
    param_keys = param_grid.keys()
    param_vals = param_grid.values()
    for params in itertools.product(*param_vals):
        # Add params to args
        for key, val in zip(param_keys, params):
            args.train_config[key] = val
            if isinstance(val, str):
                print('{:s}: {:s}'.format(key, val), end='\t')
            else:
                print('{:s}: {:.5e}'.format(key, val), end='\t')
        print('\n')

        # Define model
        model = network.init_model(
            args.network_config,
            train_loader.dataset.X.shape
        )
        
        # Define trainer
        trainer = trainer_class.Trainer(
            args=args.train_config,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device
        )
        
        # Train the model
        model_state_dict, log = trainer.train()
        
        # Update the model
        model.load_state_dict(model_state_dict)
        
        # Save
        logs[tuple(params)] = log
        
        # Update the val result
        if args.train_config['use_early_stopping']:
            local_val_result = np.min(log['val_loss_history'])
        else:
            local_val_result = log['val_loss_history'][-1]
            
        if local_val_result < best_val_result:
            best_model = copy.deepcopy(model)
            best_param = tuple(params)
            best_val_result = local_val_result
            
            # Save model
            # folder = 'Data_{:02d}_Network_{:02d}_Train_{:02d}'.format(
                # args.data_config_number,
                # args.network_config_number,
                # args.train_config_number,
            # )
            
            out_folder = os.path.join(output_model_dir, folder)
            os.makedirs(out_folder, exist_ok=True)
            out_file = '{:02d}.pt'.format(args.task_id)
            torch.save(best_model.state_dict(), os.path.join(out_folder, out_file))
            del out_folder, out_file
            
    return best_model, best_param, logs


if __name__ == '__main__':
    main()