from typing import Optional

import torch


def distance_weights(
    height: int, in_height: Optional[int] = None, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute the matrix of euclidean distances between all pairs points of points on two
    superimposed grids, possibly of different heights.

    Returns a tensor shape (height^2, in_height^2).
    """
    in_height = in_height or height

    # (height,)
    x = torch.linspace(0, height, height, device=device)
    # (in_height,)
    x_in = torch.linspace(0, height, in_height, device=device)

    # (height^2, 2)
    coord = torch.cartesian_prod(x, x)
    # (in_height^2, 2)
    coord_in = torch.cartesian_prod(x_in, x_in)

    # (height^2, in_height^2)
    dist = torch.cdist(coord, coord_in)
    return dist
