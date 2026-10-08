"""Fine-tune FlowgaitNet with identity, triplet, and center losses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.rds import ImgRDSdatasetAction
from model_img.flowgait_net import CenterLoss
from training.engine import resolve_device
from training.half_common import FineTunedFlowgaitNet, OnlineTripletLoss


def load_compatible_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load only checkpoint tensors that match the current model structure."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    current_state = model.state_dict()
    compatible = {
        name: value
        for name, value in checkpoint.items()
        if name in current_state
        and isinstance(value, torch.Tensor)
        and value.shape == current_state[name].shape
        and not name.startswith("mlp_head.")
    }
    current_state.update(compatible)
    model.load_state_dict(current_state)
    print(f"Loaded {len(compatible)} compatible tensors from {checkpoint_path}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    identity_loss: nn.Module,
    triplet_loss: nn.Module,
    center_loss: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    identity_weight: float,
    triplet_weight: float,
    center_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "identity_loss": 0.0, "triplet_loss": 0.0, "center_loss": 0.0}
    correct = 0
    sample_count = 0

    with torch.set_grad_enabled(training):
        for inputs, labels in tqdm(loader, desc="train" if training else "valid", leave=False):
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            identity_labels = labels[0].to(device, non_blocking=device.type == "cuda")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits, features = model(inputs)
            identity_value = identity_loss(logits, identity_labels)
            triplet_value = triplet_loss(features, identity_labels)
            center_value = center_loss(features, identity_labels)
            loss = (
                identity_weight * identity_value
                + triplet_weight * triplet_value
                + center_weight * center_value
            )

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = inputs.size(0)
            sample_count += batch_size
            correct += (logits.argmax(dim=1) == identity_labels).sum().item()
            totals["loss"] += loss.item() * batch_size
            totals["identity_loss"] += identity_value.item() * batch_size
            totals["triplet_loss"] += triplet_value.item() * batch_size
            totals["center_loss"] += center_value.item() * batch_size

    if sample_count == 0:
        raise RuntimeError("The data loader produced no samples.")
    return {
        **{name: value / sample_count for name, value in totals.items()},
        "accuracy": correct / sample_count,
    }


def train(cfg: argparse.Namespace) -> None:
    device = resolve_device(cfg.device)
    save_dir = Path(cfg.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()
    train_dataset = ImgRDSdatasetAction(
        cfg.train_dir, action="1", transform=transform, if_crop=True
    )
    valid_dataset = ImgRDSdatasetAction(
        cfg.valid_dir, action="2", transform=transform, if_crop=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = FineTunedFlowgaitNet(
        image_size=(11, 220),
        patch_size=11,
        num_classes=cfg.num_classes,
        dim=256,
        depth=5,
        heads=8,
        mlp_dim=256,
        device_id=device.index or 0,
        channels=1,
        dropout=0.1,
        emb_dropout=0.1,
        freeze_backbone=True,
        train_temporal=True,
        replace_classifier=False,
    ).to(device)

    if cfg.pretrained_checkpoint:
        load_compatible_weights(model, Path(cfg.pretrained_checkpoint).expanduser(), device)

    identity_loss = nn.CrossEntropyLoss()
    triplet_loss = OnlineTripletLoss(margin=0.3)
    center_loss = CenterLoss(num_classes=cfg.num_classes, feat_dim=cfg.feat_dim).to(device)
    optimizer = AdamW(
        [
            {"params": model.transformer_encoder.parameters(), "lr": cfg.backbone_learning_rate},
            {"params": model.mlp_head.parameters(), "lr": cfg.head_learning_rate},
            {"params": center_loss.parameters(), "lr": cfg.center_learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )

    best_accuracy = float("-inf")
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, identity_loss, triplet_loss, center_loss, optimizer, device,
            identity_weight=cfg.identity_loss_weight,
            triplet_weight=cfg.triplet_loss_weight,
            center_weight=cfg.center_loss_weight,
        )
        valid_metrics = run_epoch(
            model, valid_loader, identity_loss, triplet_loss, center_loss, None, device,
            identity_weight=cfg.identity_loss_weight,
            triplet_weight=cfg.triplet_loss_weight,
            center_weight=cfg.center_loss_weight,
        )
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f} | "
            f"valid loss={valid_metrics['loss']:.4f}, acc={valid_metrics['accuracy']:.4f}"
        )
        torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch}.pth")
        if valid_metrics["accuracy"] > best_accuracy:
            best_accuracy = valid_metrics["accuracy"]
            torch.save(model.state_dict(), save_dir / "best_model.pth")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Root directory of training sequences")
    parser.add_argument("--valid-dir", required=True, help="Root directory of validation sequences")
    parser.add_argument("--pretrained-checkpoint", help="Optional FlowgaitNet checkpoint to fine-tune")
    parser.add_argument("--save-dir", default="runs/flowgaitnet-finetune")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--center-learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-loss-weight", type=float, default=0.1)
    parser.add_argument("--triplet-loss-weight", type=float, default=0.0)
    parser.add_argument("--center-loss-weight", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
