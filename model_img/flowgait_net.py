"""FlowgaitNet: a ViViT-style spatial-temporal Transformer for radar gait sequences."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


PairLike = Union[int, Tuple[int, int]]


def pair(value: PairLike) -> Tuple[int, int]:
    """Convert an integer or pair into a two-dimensional tuple."""
    return value if isinstance(value, tuple) else (value, value)


class FeedForward(nn.Module):
    """Transformer feed-forward block with pre-normalization."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Attention(nn.Module):
    """Multi-head self-attention with pre-normalization."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        query, key, value = self.to_qkv(normalized).chunk(3, dim=-1)
        query, key, value = (
            rearrange(tensor, "b n (h d) -> b h n d", h=self.heads)
            for tensor in (query, key, value)
        )
        attention = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention = self.dropout(self.attend(attention))
        output = torch.matmul(attention, value)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.to_out(output)


class Transformer(nn.Module):
    """Stack of residual attention and feed-forward blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for attention, feed_forward in self.layers:
            inputs = attention(inputs) + inputs
            inputs = feed_forward(inputs) + inputs
        return self.norm(inputs)


class CenterLoss(nn.Module):
    """Pull embeddings toward their trainable identity centers."""

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers = self.centers.index_select(0, labels)
        return torch.sum((embeddings - centers) ** 2) / embeddings.size(0)


class FlowgaitNet(nn.Module):
    """Encode each radar frame spatially, then encode the frame sequence."""

    def __init__(
        self,
        *,
        image_size: PairLike,
        patch_size: PairLike,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        device_id: int,
        pool: str = "cls",
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        out_feats: int = 256,
    ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        if image_height % patch_height or image_width % patch_width:
            raise ValueError("Image dimensions must be divisible by patch dimensions.")
        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be either 'cls' or 'mean'.")

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # These dimensions match the historical checkpoints and the primary configuration.
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=6)
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, 256))
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head_prev = nn.Linear(out_feats, 512)
        self.mlp_head = nn.Linear(256, num_classes)
        self.mlp_head2 = nn.Linear(dim, 5)

    def forward(
        self,
        images: torch.Tensor,
        is_test: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, frame_count, height, width = images.shape
        frames = images.view(-1, 1, height, width)
        features = self.to_patch_embedding(frames)
        _, patch_count, _ = features.shape

        frame_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=features.size(0))
        features = torch.cat((frame_tokens, features), dim=1)
        features = features + self.pos_embedding[:, : patch_count + 1]
        features = self.dropout(features)
        features = self.transformer(features)
        features = features.mean(dim=1) if self.pool == "mean" else features[:, 0]
        features = self.to_latent(features).view(batch_size, frame_count, -1)

        sequence_token = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((sequence_token, features), dim=1)
        sequence = self.transformer_encoder(sequence)
        embedding = sequence.mean(dim=1) if self.pool == "mean" else sequence[:, 0]
        if is_test:
            return embedding
        return self.mlp_head(embedding), embedding


__all__ = [
    "Attention",
    "CenterLoss",
    "FeedForward",
    "Transformer",
    "FlowgaitNet",
    "pair",
]
