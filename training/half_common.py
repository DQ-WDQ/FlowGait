"""Shared components for the retained half-supervised training variants."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from model_img.flowgait_net import FlowgaitNet


class FineTunedFlowgaitNet(FlowgaitNet):
    """FlowgaitNet with a trainable MLP identity head.

    By default the backbone is frozen and only the replacement classifier is
    trained, matching the existing half-supervised experiment setup.
    """

    def __init__(
        self,
        *,
        freeze_backbone: bool = True,
        train_temporal: bool = False,
        replace_classifier: bool = True,
        mlp_hidden_dim: int = 512,
        **model_kwargs,
    ) -> None:
        super().__init__(**model_kwargs)
        dim = model_kwargs["dim"]
        num_classes = model_kwargs["num_classes"]
        if replace_classifier:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mlp_hidden_dim, num_classes),
            )

        if freeze_backbone:
            for parameter in self.parameters():
                parameter.requires_grad = False
            if train_temporal:
                for parameter in self.transformer_encoder.parameters():
                    parameter.requires_grad = True
            for parameter in self.mlp_head.parameters():
                parameter.requires_grad = True


class OnlineTripletLoss(nn.Module):
    """Batch-hard triplet loss over cosine distance."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pairwise_distance = 1 - torch.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        same_identity = labels[:, None] == labels[None, :]
        hardest_positive = pairwise_distance.masked_fill(~same_identity, float("-inf")).max(dim=1)[0]
        hardest_negative = pairwise_distance.masked_fill(same_identity, float("inf")).min(dim=1)[0]
        valid = torch.isfinite(hardest_positive) & torch.isfinite(hardest_negative)
        if not valid.any():
            return embeddings.sum() * 0.0
        return torch.relu(
            hardest_positive[valid] - hardest_negative[valid] + self.margin
        ).mean()


class PseudoLabeledDataset(Dataset):
    """Wrap a dataset or subset with identity pseudo-labels."""

    def __init__(self, source_dataset: Dataset, pseudo_labels) -> None:
        if len(source_dataset) != len(pseudo_labels):
            raise ValueError("source_dataset and pseudo_labels must have equal lengths.")
        self.source_dataset = source_dataset
        self.pseudo_labels = pseudo_labels

    def __len__(self) -> int:
        return len(self.pseudo_labels)

    def __getitem__(self, index: int):
        image, _ = self.source_dataset[index]
        identity_label = torch.as_tensor(self.pseudo_labels[index], dtype=torch.long)
        return image, (identity_label, torch.tensor(-1, dtype=torch.long))
