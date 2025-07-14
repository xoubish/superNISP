import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import yaml

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'stage2/val_ssim_loss',  # Primary metric to optimize
        'goal': 'minimize'
    },
    'parameters': {
        # Architecture parameters
        'num_rrdb': {
            'values': [6, 8, 12, 16]
        },
        'features': {
            'values': [32, 64, 96, 128]
        },
        'growth_rate': {
            'values': [16, 24, 32, 48]
        },
        
        # Training parameters
        'batch_size': {
            'values': [4, 8, 16, 24]
        },
        'lr_stage1': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3
        },
        'lr_stage2': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-4
        },
        
        # Loss function weights
        'l1_weight_stage1': {
            'distribution': 'uniform',
            'min': 0.5,
            'max': 0.9
        },
        'l1_weight_stage2': {
            'distribution': 'uniform',
            'min': 0.3,
            'max': 0.7
        },
        'mse_weight_stage2': {
            'distribution': 'uniform',
            'min': 0.2,
            'max': 0.5
        },
        'ssim_weight_stage2': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.4
        },
        
        # Regularization
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3
        },
        'gradient_clip_norm': {
            'values': [0.1, 0.5, 1.0, 2.0]
        },
        
        # Scheduler parameters
        # 'scheduler_type': {
        #     'values': ['cosine', 'step', 'exponential']
        # },
        # 'scheduler_patience': {
        #     'values': [5, 10, 15, 20]  # for ReduceLROnPlateau
        # },
        
        # # Data augmentation
        # 'augmentation_prob': {
        #     'distribution': 'uniform',
        #     'min': 0.3,
        #     'max': 0.8
        # },
        
        # Normalization method
        'normalization_method': {
            'values': ['flux_preserving', 'adaptive_hist', 'percentile', 'z_score']
        }
    }
}