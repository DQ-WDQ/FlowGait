"""Train FlowgaitNet on a selected set of gait actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data.rds import ImgRDSdatasetAction2
from model_img.flowgait_net import CenterLoss, FlowgaitNet
from training.engine import fit_identity_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True, help="Training dataset root.")
    parser.add_argument("--valid-dir", type=Path, required=True, help="Validation dataset root.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs/vit_action_baseline"),
        help="Directory for checkpoints.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or a device such as cuda:0.")
    parser.add_argument("--actions", nargs="+", default=["1"], help="Action folders to train on.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=88)
    parser.add_argument("--epochs", type=int, default=121)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--id-loss-weight", type=float, default=1.0)
    parser.add_argument("--center-loss-weight", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.epochs < 1 or args.num_classes < 1:
        raise ValueError("batch-size, epochs, and num-classes must be positive.")
    if args.num_workers < 0 or args.save_every < 1:
        raise ValueError("num-workers cannot be negative and save-every must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")
    if args.center_loss_weight < 0 or args.id_loss_weight < 0:
        raise ValueError("Loss weights cannot be negative.")

    train_dir = args.train_dir.expanduser().resolve()
    valid_dir = args.valid_dir.expanduser().resolve()
    for label, directory in (("Training", train_dir), ("Validation", valid_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")

    save_dir = args.save_dir.expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    transform = transforms.ToTensor()

    train_dataset = ImgRDSdatasetAction2(
        root_dir=train_dir,
        transform=transform,
        actions=args.actions,
    )
    valid_dataset = ImgRDSdatasetAction2(
        root_dir=valid_dir,
        transform=transform,
        actions=args.actions,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Training and validation datasets must both contain samples.")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)

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
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
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
        valid_loader=valid_loader,
    )


if __name__ == "__main__":
    main(parse_args())
