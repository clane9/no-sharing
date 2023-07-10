from typing import Optional

import torch


def distance_weights(
    height: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the matrix of euclidean distance weights for a grid of shape (height,
    height). Returns a tensor shape (height^2, height^2).
    """
    # (height,)
    ind = torch.arange(height, dtype=torch.float32, device=device)
    # (height^2, 2)
    coord = torch.cartesian_prod(ind, ind)
    # (height^2, height^2)
    dist = torch.cdist(coord, coord)
    return dist


def random_sample(
    size: int, max_samples: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate a random sample without replacement. Returns a tensor shape (max_samples,).
    """
    indices = torch.randperm(size, device=device)[:max_samples]
    return indices
