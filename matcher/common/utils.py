r""" Helper functions """
import random

import torch
import numpy as np


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to('cuda:0')
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


def remove_background(images: torch.tensor, masks: torch.tensor):
    r""" Remove background from images """
    # Images have shape (BS, NUM_SHOT, C, H, W)
    # Masks have shape (BS, NUM_SHOT, H, W)
    images = images * masks.unsqueeze(2).float()
    return images
