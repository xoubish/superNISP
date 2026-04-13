import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import yaml

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'final_sweep_loss',  # This matches what we log above
        'goal': 'minimize'
    },
    'parameters': {
        # Architecture parameters
        'num_rrdb': {
            'values': [6, 8, 12]
        },
        'features': {
            'values': [32, 64, 96]
        },
        # Training parameters
        'batch_size': {
            'values': [8, 16]
        },
        'lr_stage1': {
            'distribution': 'log_uniform_values',
            'min': 5e-5,
            'max': 5e-4
        },
        'lr_stage2': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-4
        },
        # 'num_epochs_stage1': {
        #     'values': [20, 30, 40, 50]
        # },
        # 'num_epochs_stage2': {
        #     'values': [20, 30, 40, 50]
        # },
        
        # Loss function weights
        'l1_weight_stage1': {
            'values': [0.6, 0.7, 0.8]
        },
        # 'l1_weight_stage2': {
        #     'values': [0.2, 0.3, 0.4]
        # },
        'mse_weight_stage2': {
            'values': [0.1, 0.2, 0.3, .4]
        },
        'ssim_weight_stage2': {
            'values': [0.05, 0.1, 0.15, .2, .3, .4, .5]
        },
        
        # Regularization
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'gradient_clip_norm': {
            'values': [0.5, 1.0]
        }
    }
}

best_params = {
        # Model architecture
        'num_rrdb': 8,  # Update with your best value
        'features': 64,  # Update with your best value
        
        # Training parameters
        'batch_size': 16,  # Update with your best value
        'num_epochs_stage1': 50,  # Can use more epochs for final training
        'num_epochs_stage2': 50,  # Can use more epochs for final training
        
        # Learning rates
        'lr_stage1': 0.00005662124820190853,  # Update with your best value
        'lr_stage2': 0.00009223193223543192,  # Update with your best value
        
        # Loss weights
        'l1_weight_stage1': 0.8,  # Update with your best value
        'mse_weight_stage2': 0.1,  # Update with your best value
        'ssim_weight_stage2': 0.5,  # Update with your best value
        
        # Regularization
        'weight_decay': 0.00001215158933763565,  # Update with your best value
        'gradient_clip_norm': 1,  # Update with your best value

        'normalization': 'z_score',
}