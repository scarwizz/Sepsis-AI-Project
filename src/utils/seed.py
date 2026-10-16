"""
Reproducibility and Deterministic Seeding Utilities.
Ensures identical pseudorandom generation across Python, NumPy, PyTorch CPU/CUDA, and DataLoaders.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Sets seeds across all random number generators and forces cuDNN determinism.
    
    Args:
        seed (int): The integer random seed.
        deterministic_cudnn (bool): Whether to enforce torch.backends.cudnn.deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function to ensure worker processes inherit deterministic seeds.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
