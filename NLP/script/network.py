import torch
import torch.nn as nn

from script.mlp import MLPEncoderDecoder, MLPClassifier
from script.conv1D import Conv1DAutoencoder


def init_model(args, input_shape):
    if args['model'] == 'MLPEncoderDecoder':
        model = MLPEncoderDecoder(
            input_shape=input_shape,
            hidden_dims=args['hidden_dim'],
            latent_dim=args['latent_dim']
        )
    elif args['model'] == 'Conv1DAutoencoder':
        model = Conv1DAutoencoder(
            in_channels=input_shape[1],
            hidden_channels=args['hidden_channels'],
            latent_dim=args['latent_dim'],
            seq_len=input_shape[2],
            kernel_size=args['kernel_size'],
            stride=args['stride'],
            padding=args['padding'],
            dilation=args['dilation']
        )
    elif args['model'] == 'MLPClassifier':
        model = MLPClassifier(
            input_shape=input_shape,
            encoder_hidden_dims=args['hidden_dim'],
            latent_dim=args['latent_dim'],
            classifier_hidden_dims=args['classifier_hidden_dim']
        )
    
    return model