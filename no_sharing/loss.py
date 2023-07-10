import torch
import torch.nn.functional as F
from torch import nn

from .utils import distance_weights, random_sample


class LocalInfoNCELoss(nn.Module):
    """
    A contrastive loss function that promotes a smooth representation map over a grid.
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
        # (N, L, C)
        target = torch.matmul(self.weight, embedding)
        # (N, L)
        sim_pos = torch.sum(embedding * target, dim=-1) / self.temperature

        # Negative similarity: Random sample of embeddings from other images in the batch.
        # Should hopefully prevent collapse.
        # TODO: for some reason the embedding is not contiguous. Figure out why.
        flattened = embedding.reshape(-1, C)
        neg_indices = random_sample(
            size=N * L, max_samples=self.num_negative, device=device
        )
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

    def __init__(self, height: int, lambd: float = 1.0):
        super().__init__()
        self.height = height
        self.lambd = lambd

        # euclidean distance weights, shape (height^2, height^2)
        dist = distance_weights(height)
        self.dist: torch.Tensor
        self.register_buffer("dist", dist)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        L = self.height**2
        assert weight.shape[-2:] == (L, L)

        # TODO: should the cost just increase linearly with wiring distance like this?
        cost = weight.abs() * self.dist
        cost = cost.sum(dim=(-2, -1))
        cost = (self.lambd / L) * cost.mean()
        return cost

    def extra_repr(self) -> str:
        return f"height={self.height}, lambd={self.lambd}"
