"""Train FlowgaitNet on range-Doppler sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data.rds import ImgRDSdataset
from model_img.flowgait_net import CenterLoss, FlowgaitNet
from training.engine import fit_identity_model, resolve_device


def parse_args() -> argparse.Namespace:
    """Parse the public command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True, help="Training dataset root.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs/flowgaitnet"),
        help="Directory for checkpoints (default: runs/flowgaitnet).",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or a device such as cuda:0.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=121)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--id-loss-weight", type=float, default=1.0)
    parser.add_argument("--center-loss-weight", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Build the data pipeline and run training."""
    if args.batch_size < 1 or args.epochs < 1 or args.num_classes < 1:
        raise ValueError("batch-size, epochs, and num-classes must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if args.save_every < 1:
        raise ValueError("save-every must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if args.center_loss_weight < 0 or args.id_loss_weight < 0:
        raise ValueError("Loss weights cannot be negative.")

    train_dir = args.train_dir.expanduser().resolve()
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")

    save_dir = args.save_dir.expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    dataset = ImgRDSdataset(root_dir=train_dir, transform=transforms.ToTensor())
    if len(dataset) == 0:
        raise ValueError(f"No training samples were found in: {train_dir}")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = FlowgaitNet(
        image_size=(11, 220),
        patch_size=11,
        num_classes=args.num_classes,
        dim=256,
        depth=5,
        heads=8,
        mlp_dim=256,
        device_id=device.index or 0,
        channels=1,
        dropout=0.1,
        emb_dropout=0.1,
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_center = CenterLoss(num_classes=args.num_classes, feat_dim=256).to(device)

    trainable_parameters = list(model.parameters())
    if args.center_loss_weight > 0:
        trainable_parameters.extend(criterion_center.parameters())
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    print(f"Device: {device}")
    print(f"Training samples: {len(dataset)}")
    print(f"Checkpoint directory: {save_dir.resolve()}")
    fit_identity_model(
        model,
        train_loader,
        criterion,
        criterion_center,
        optimizer,
        device=device,
        epochs=args.epochs,
        save_dir=save_dir,
        save_every=args.save_every,
        id_loss_weight=args.id_loss_weight,
        center_loss_weight=args.center_loss_weight,
    )


if __name__ == "__main__":
    main(parse_args())
