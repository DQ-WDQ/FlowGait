"""Shared training loop for identity classification with an optional center loss."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def resolve_device(device_name: str) -> torch.device:
    """Resolve ``auto`` or validate an explicitly requested device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return device


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    criterion_center: nn.Module,
    device: torch.device,
    *,
    training: bool,
    optimizer: Optional[torch.optim.Optimizer],
    id_loss_weight: float,
    center_loss_weight: float,
    epoch: int,
    epochs: int,
) -> dict[str, float]:
    model.train(training)
    total_loss = 0.0
    total_id_loss = 0.0
    total_center_loss = 0.0
    correct = 0
    count = 0

    phase = "train" if training else "valid"
    progress = tqdm(loader, desc=f"{phase} {epoch}/{epochs}", leave=False)
    with torch.set_grad_enabled(training):
        for inputs, labels in progress:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            identity_labels = labels[0].to(device, non_blocking=device.type == "cuda")

            if training:
                if optimizer is None:
                    raise ValueError("An optimizer is required for training epochs.")
                optimizer.zero_grad(set_to_none=True)

            logits, features = model(inputs)
            identity_loss = criterion(logits, identity_labels)
            center_loss = criterion_center(features, identity_labels)
            loss = id_loss_weight * identity_loss + center_loss_weight * center_loss

            if training:
                loss.backward()
                optimizer.step()

            batch_size = inputs.size(0)
            count += batch_size
            correct += (logits.argmax(dim=1) == identity_labels).sum().item()
            total_loss += loss.item() * batch_size
            total_id_loss += identity_loss.item() * batch_size
            total_center_loss += center_loss.item() * batch_size

    if count == 0:
        raise RuntimeError(f"The {phase} loader produced no samples.")

    return {
        "loss": total_loss / count,
        "id_loss": total_id_loss / count,
        "center_loss": total_center_loss / count,
        "accuracy": correct / count,
    }


def fit_identity_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    criterion_center: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    save_dir: Path,
    save_every: int = 10,
    id_loss_weight: float = 1.0,
    center_loss_weight: float = 0.0,
    valid_loader: Optional[DataLoader] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> None:
    """Fit a model, optionally select the best validation checkpoint, and save weights."""
    if epochs < 1 or save_every < 1:
        raise ValueError("epochs and save_every must be positive.")

    save_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = float("-inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            criterion_center,
            device,
            training=True,
            optimizer=optimizer,
            id_loss_weight=id_loss_weight,
            center_loss_weight=center_loss_weight,
            epoch=epoch,
            epochs=epochs,
        )
        print(
            f"Epoch {epoch}/{epochs} train | "
            f"loss={train_metrics['loss']:.4f} | "
            f"id_loss={train_metrics['id_loss']:.4f} | "
            f"center_loss={train_metrics['center_loss']:.4f} | "
            f"id_accuracy={train_metrics['accuracy']:.4f}"
        )

        if valid_loader is not None:
            valid_metrics = _run_epoch(
                model,
                valid_loader,
                criterion,
                criterion_center,
                device,
                training=False,
                optimizer=None,
                id_loss_weight=id_loss_weight,
                center_loss_weight=center_loss_weight,
                epoch=epoch,
                epochs=epochs,
            )
            print(
                f"Epoch {epoch}/{epochs} valid | "
                f"loss={valid_metrics['loss']:.4f} | "
                f"id_loss={valid_metrics['id_loss']:.4f} | "
                f"center_loss={valid_metrics['center_loss']:.4f} | "
                f"id_accuracy={valid_metrics['accuracy']:.4f}"
            )
            if valid_metrics["accuracy"] > best_accuracy:
                best_accuracy = valid_metrics["accuracy"]
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, save_dir / "best_model.pth")

        if scheduler is not None:
            scheduler.step()

        if epoch % save_every == 0:
            torch.save(model.state_dict(), save_dir / f"model_epoch_{epoch}.pth")

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_dir / "final_model.pth")
