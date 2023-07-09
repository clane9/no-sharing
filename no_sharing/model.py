import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from timm.layers import DropPath, PatchEmbed
from torch import nn

from .loss import LocalInfoNCELoss, WiringCost

Layer = Callable[..., nn.Module]


class ColumnAttention(nn.Module):
    """
    Spatial plus content attention over a grid of feature columns.

    Each column learns an independent constant query for selecting relevant content, as
    well as a spatial receptive field bias. The final attention mask is computed as::

        softmax(query @ input.T + bias)
    """

    def __init__(
        self,
        height: int,
        dim: int,
        drop: float = 0.0,
        attn: bool = True,
        bias: bool = True,
    ):
        assert attn or bias, "at least one of attn or bias required"

        super().__init__()
        self.height = height
        self.dim = dim

        # learned query for content attention
        # TODO: more than one query i.e. head?
        if attn:
            self.query = nn.Parameter(torch.empty(height**2, dim))
        else:
            self.register_parameter("query", None)

        # learned spatial attention bias
        if bias:
            self.bias = nn.Parameter(torch.empty(height**2, height**2))
        else:
            self.register_parameter("bias", None)

        self.drop = nn.Dropout(drop)
        self.reset_parameters()

    def reset_parameters(self):
        if self.query is not None:
            bound = 1 / math.sqrt(self.dim)
            nn.init.uniform_(self.query, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        N, L, _ = x.shape

        attn = 0
        if self.query is not None:
            attn = self.query @ x.transpose(-2, -1)
        if self.bias is not None:
            attn = attn + self.bias

        assert attn.shape == (N, L, L)
        attn = F.softmax(attn, dim=-1)

        x = self.drop(attn) @ x
        if return_attention:
            return x, attn
        return x

    def extra_repr(self) -> str:
        return (
            f"height={self.height}, dim={self.dim}, "
            f"attn={self.query is not None}, bias={self.bias is not None}"
        )


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
    A network block consisting of column-wise independent attention and MLP.

    The expected input is a grid of feature columns, shape `(batch_size, height^2, dim)`.

    Also defines a loss consisting of a wiring cost for the attention maps, and a local
    contrastive loss to promote a smooth representation map.
    """

    def __init__(
        self,
        height: int,
        dim: int,
        mlp_ratio: 1 / 8.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Layer = nn.GELU,
        content_attn: bool = True,
        spatial_bias: bool = True,
        wiring_lambd: float = 1.0,
        contrast_simga: float = 2.0,
        detach: bool = False,
    ):
        super().__init__()
        self.height = height
        self.dim = dim
        self.detach = detach

        # Don't want to share weights across the grid, so no affine params
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = ColumnAttention(
            height=height,
            dim=dim,
            drop=attn_drop,
            attn=content_attn,
            bias=spatial_bias,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = ColumnMlp(
            height=height,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cost = WiringCost(height, lambd=wiring_lambd)
        self.criterion = LocalInfoNCELoss(height, sigma=contrast_simga)

        self._attn_map: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Don't backprop to earlier layers
        if self.detach:
            x = x.detach()

        y = self.norm1(x)
        y, attn = self.attn(y, return_attention=True)
        x = x + self.drop_path1(y)

        y = self.norm2(x)
        y = self.mlp(y)
        x = x + self.drop_path2(y)

        self._attn_map = attn
        self._output = x
        return x

    def loss(self) -> torch.Tensor:
        assert self._attn_map is not None
        assert self._output is not None

        loss = self.cost(self._attn_map) + self.criterion(self._output)
        self._attn_map = None
        self._output = None
        return loss

    def extra_repr(self) -> str:
        return f"detach={self.detach}"


class Mapformer(nn.Module):
    """
    A brain-inspired topographic vision model with learned contrastive weight sharing.

    Patterned after the vision transformer in timm.

    Inspired by comments from Geoff Hinton on the Robot Brains podcast and recent work
    on topographic deep networks.
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
