import os
import random
import numpy as np
import torch

def set_all_seeds(seed: int = 42, deterministic_cuda: bool = True):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"All seeds set to {seed}")


def get_deterministic_generator(seed: int = 42, device: str = 'cpu'):
    """Get a generator object for explicit RNG control."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g