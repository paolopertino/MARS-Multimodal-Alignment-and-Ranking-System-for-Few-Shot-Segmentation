import numpy as np


def inclusion_percentage(mask_a, mask_b):
    """
    Calculate the percentage of mask_a pixels that are within mask_b.
    Args:
        mask_a (np.ndarray): The smaller mask.
        mask_b (np.ndarray): The larger mask.
    Returns:
        float: Inclusion percentage of mask_a within mask_b.
    """
    overlap = np.logical_and(mask_a, mask_b)
    return np.sum(overlap) / np.sum(mask_a) if np.sum(mask_a) > 0 else 0
