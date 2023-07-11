from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .utils import distance_weights


class LocalInfoNCELoss(nn.Module):
    """
    A contrastive loss function that promotes a smooth representation map over a grid.

    The expected input is a grid of feature columns, shape `(batch_size, height^2, dim)`.

    The loss is computed independently for each column in the batch. A weighted average
    of nearby columns is used as the positive for each column. The negatives are drawn
    from other images in the batch.
    """

    def __init__(
        self,
        height: int,
        sigma: float = 2.0,
        temperature: float = 0.1,
        num_negative: int = 4096,
    ):
        super().__init__()
        self.height = height
        self.sigma = sigma
        self.temperature = temperature
        self.num_negative = num_negative

        # Gaussian neighborhood weights, shape (height^2, height^2)
        dist = distance_weights(height)
        weight = -(0.5 / sigma**2) * dist.square()
        # TODO: fill diagonal necessary?
        weight.fill_diagonal_(-torch.inf)
        weight = torch.softmax(weight, dim=1)

        self.weight: torch.Tensor
        self.register_buffer("weight", weight)

    def forward(self, embedding: torch.Tensor):
        # TODO: Should we detach the positive or negative examples?
        # TODO: Do we need a memory bank? DDP gather?
        N, L, C = embedding.shape
        assert L == self.height**2
        assert N >= 8, "insufficient batch size"

        device = embedding.device
        embedding = F.normalize(embedding, dim=-1)

        # Positive similarity: each column embedding compared to a weighted average of
        # its neighbors.
        # TODO: Could also consider including each neighbor individually as a positive.
        # Not sure what the difference would be.
        # (N, L, C)
        target = torch.matmul(self.weight, embedding)
        # (N, L)
        sim_pos = torch.sum(embedding * target, dim=-1) / self.temperature

        # Negative similarity: Random sample of embeddings from other images in the batch.
        # TODO: Could also consider including negatives from the same image but outside
        # the neighborhood. But not sure I want to enforce contrast within the map.
        # TODO: for some reason the embedding is not contiguous. Figure out why.
        flattened = embedding.reshape(-1, C)
        neg_indices = torch.randperm(N * L, device=device)[: self.num_negative]
        # (num_neg, C)
        negatives = flattened[neg_indices]
        # (N, L, num_neg)
        sim_neg = torch.matmul(embedding, negatives.t()) / self.temperature

        # Mask of negative examples from other samples in the batch
        # (N, 1, num_neg)
        batch_idx = neg_indices // L
        neg_mask = torch.arange(N, device=device)[:, None, None] != batch_idx

        # Offset of numerical stability
        offset = sim_pos.detach().amax()
        pos = torch.exp(sim_pos - offset)
        neg = torch.sum(torch.exp(sim_neg - offset) * neg_mask, dim=-1)
        loss = -torch.mean(torch.log(pos / (pos + neg)))
        return loss

    def extra_repr(self) -> str:
        return (
            f"height={self.height}, sigma={self.sigma}, "
            f"temperature={self.temperature}, num_negative={self.num_negative}"
        )


class WiringCost(nn.Module):
    """
    An L1 penalty weighted by the wiring distance on a grid.
    """

    def __init__(
        self,
        height: int,
        in_height: Optional[int] = None,
        lambd: float = 1.0,
    ):
        super().__init__()
        self.height = height
        self.in_height = in_height or height
        self.lambd = lambd

        # Euclidean distance weights, shape (height^2, in_height^2)
        dist = distance_weights(height, in_height=in_height)

        # Offset distance in the z direction
        # TODO: this could be a hyper-parameter
        dist = torch.sqrt(1.0 + dist**2)
        self.dist: torch.Tensor
        self.register_buffer("dist", dist)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.shape[-2:] == (self.height**2, self.in_height**2)

        # TODO: should the cost just increase linearly with wiring distance like this?
        cost = torch.sum(weight.abs() * self.dist, dim=(-2, -1))
        cost = (self.lambd / self.height**2) * cost.mean()
        return cost

    def extra_repr(self) -> str:
        return f"height={self.height}, in_height={self.in_height}, lambd={self.lambd}"
