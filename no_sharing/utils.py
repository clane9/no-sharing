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
    size: int, num_samples: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate a random sample without replacement. Returns a tensor shape (num_samples,).
    """
    x = torch.rand(size, device=device)
    _, indices = torch.topk(x, num_samples, sorted=False)
    return indices
