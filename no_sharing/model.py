import math
from typing import Callable, Optional

import torch
from timm.layers import DropPath, PatchEmbed
from torch import nn

from .loss import LocalInfoNCELoss, WiringCost

Layer = Callable[..., nn.Module]


class Pool(nn.Module):
    """
    Learned spatial pooling over a grid of feature columns.
    """

    def __init__(self, height: int, drop: float = 0.0):
        super().__init__()
        self.height = height

        self.drop = nn.Dropout(drop)
        self.weight = nn.Parameter(torch.empty(height**2, height**2))
        self.reset_parameters()

    def reset_parameters(self):
        # copied from nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L, C)
        # output: (N, L, C)
        x = torch.matmul(self.drop(self.weight), x)
        return x

    def extra_repr(self) -> str:
        return f"height={self.height}, dropout={self.dropout}"


class MultiLinear(nn.Module):
    """
    An independent linear layer for each item in a sequence.
    """

    def __init__(
        self, seq_length: int, in_features: int, out_features: int, bias: bool = True
    ):
        super().__init__()
        self.seq_length = seq_length
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((seq_length, out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(seq_length, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Adapted from nn.Linear
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (batch_size, seq_length, in_features)
        # output: (batch_size, seq_length, out_features)

        # TODO: any way to write this with fewer transposes? GPT wasn't very helpful.
        # (sl, bs, if)
        input = torch.transpose(input, 0, 1)
        # (sl, if, of)
        weight = torch.transpose(self.weight, 1, 2)
        # (sl, bs, of)
        output = torch.matmul(input, weight)
        # (bs, sl, of)
        output = torch.transpose(output, 0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return "seq_length={}, in_features={}, out_features={}, bias={}".format(
            self.seq_length, self.in_features, self.out_features, self.bias is not None
        )


class ColumnMlp(nn.Module):
    """
    An independent MLP for each feature column in a grid.
    """

    def __init__(
        self,
        height: int,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Layer = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = MultiLinear(height**2, in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = MultiLinear(height**2, hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """
    A network block consisting of learned spatial pooling and per-column MLP.

    The expected input is a grid of feature columns, shape `(batch_size, height^2, dim)`.

    Also defines a loss consisting of a wiring cost for the spatial pooling maps, and a
    local contrastive loss to promote a smooth representation map.
    """

    def __init__(
        self,
        height: int,
        dim: int,
        mlp_ratio: 1 / 8.0,
        pool_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Layer = nn.GELU,
        wiring_lambd: float = 1.0,
        contrast_simga: float = 2.0,
        detach: bool = False,
    ):
        super().__init__()
        self.height = height
        self.dim = dim
        self.detach = detach

        self.pool = Pool(height, dropout=pool_drop)
        self.mlp = ColumnMlp(
            height,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
            dropout=proj_drop,
        )
        # Don't want to share weights across the grid, so no afine params
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cost = WiringCost(height, lambd=wiring_lambd)
        self.criterion = LocalInfoNCELoss(height, sigma=contrast_simga)
        self._output: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Don't backprop to earlier layers
        if self.detach:
            x = x.detach()
        y = self.pool(x)
        y = self.mlp(y)
        y = self.norm(y)
        y = x + self.drop_path(y)
        self._output = y
        return y

    def loss(self) -> torch.Tensor:
        assert self._output is not None

        loss = self.cost(self.pool.weight) + self.criterion(self._output)
        self._output = None
        return loss

    def extra_repr(self) -> str:
        return f"detach={self.detach}"


class Mapformer(nn.Module):
    """
    A brain-inspired topographic vision model with learned contrastive weight sharing.

    Patterned after the vision transformer in timm.
    """

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 512,
        depth: int = 12,
        mlp_ratio: float = 1 / 8.0,
        pool_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: Layer = nn.GELU,
        wiring_lambd: float = 1.0,
        contrast_sigma: float = 2.0,
        detach: bool = False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        height = math.isqrt(num_patches)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    height=height,
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    pool_drop=pool_drop_rate,
                    proj_drop=proj_drop_rate,
                    drop_path=dpr[ii],
                    act_layer=act_layer,
                    wiring_lambd=wiring_lambd,
                    contrast_sigma=contrast_sigma,
                    detach=(detach and ii > 0),
                )
                for ii in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        return x

    def loss(self) -> torch.Tensor:
        loss = sum(block.loss() for block in self.blocks)
        loss = loss / len(self.blocks)
        return loss
