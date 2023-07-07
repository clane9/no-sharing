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

        self.criterion = nn.CrossEntropyLoss()

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

        device = embedding.device
        embedding = F.normalize(embedding, dim=-1)

        # Positive similarity: each column embedding compared to a weighted average of
        # its neighbors.
        # (N, L, C)
        target = torch.matmul(self.weight, embedding)
        # (N, L)
        sim_pos = torch.sum(embedding * target, dim=-1)

        # Negative similarity: Random sample of embeddings from the full batch.
        # Should hopefully prevent collapse.
        flattened = embedding.view(-1, C)
        neg_indices = random_sample(
            size=N * L, num_samples=self.num_negative, device=device
        )
        # (num_neg, C)
        negatives = flattened[neg_indices]
        # (N, L, num_neg)
        sim_neg = torch.matmul(embedding, negatives.t())

        logits = torch.cat([sim_pos[..., None], sim_neg], dim=2)
        logits = logits.view(N * L, 1 + self.num_negative)
        target = torch.zeros(N * L, dtype=torch.int64, device=device)

        loss = self.criterion(logits, target)
        return loss

    def extra_repr(self) -> str:
        return (
            f"height={self.height}, sigma={self.sigma}, "
            f"temperature={self.temperature}, num_negative={self.num_negative}"
        )


class WiringCost(nn.Module):
    """
    An L2 weight decay penalty weighted by the wiring distance on a grid.
    """

    def __init__(
        self,
        height: int,
        lambd: float = 1.0,
    ):
        super().__init__()
        self.height = height
        self.lambd = lambd

        # euclidean distance weights, shape (height^2, height^2)
        dist = distance_weights(height)
        self.dist: torch.Tensor
        self.register_buffer("dist", dist)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.shape[-2:] == (self.height**2, self.height**2)
        loss = torch.sum(weight.square() * self.dist)
        return loss

    def extra_repr(self) -> str:
        return f"height={self.height}, lambd={self.lambd}, squ"
