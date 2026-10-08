"""Dataset readers for the supported 20-frame range-Doppler workflow."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


class _RDSSequenceDataset(Dataset):
    """Load fixed-length grayscale frame sequences from identity/action folders."""

    def __init__(
        self,
        root_dir,
        *,
        actions: Iterable[str],
        transform=None,
        gallery_split: Optional[bool] = None,
        binary: bool = False,
        crop: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        self.actions = [str(action) for action in actions]
        if not self.actions:
            raise ValueError("At least one action folder must be selected.")
        self.action2label = {action: index for index, action in enumerate(self.actions)}
        self.transform = transform
        self.binary = binary
        self.crop = crop

        sequence_dirs = sorted(
            path for path in self.root_dir.rglob("*")
            if path.is_dir() and (path / "0.png").is_file()
        )
        selected = [path for path in sequence_dirs if path.parent.name in self.action2label]
        if gallery_split is not None:
            rng = random.Random(42)
            gallery_count = max(1, int(len(selected) * 0.2)) if selected else 0
            gallery_paths = set(rng.sample(selected, gallery_count))
            selected = [
                path for path in selected
                if (path in gallery_paths) == gallery_split
            ]
        self.sequence_dirs = selected

        identity_names = sorted(
            {path.parent.parent.name for path in sequence_dirs},
            key=lambda value: int(value),
        )
        self.id2label = {identity: index for index, identity in enumerate(identity_names)}

    def __len__(self) -> int:
        return len(self.sequence_dirs)

    def __getitem__(self, index: int):
        sequence_dir = self.sequence_dirs[index]
        frames = []
        for frame_index in range(20):
            frame_path = sequence_dir / f"{frame_index}.png"
            if not frame_path.is_file():
                raise FileNotFoundError(f"Expected frame is missing: {frame_path}")
            with Image.open(frame_path) as source:
                image = source.convert("L")
                if self.crop:
                    width, height = image.size
                    image = image.crop((17, 0, width - 17, height))
                if self.transform is not None:
                    image = self.transform(image)
                if self.binary:
                    image = image * (image > 0.6)
                frames.append(image)

        sequence = torch.stack(frames, dim=1).squeeze(0)
        identity = sequence_dir.parent.parent.name
        action = sequence_dir.parent.name
        labels = (
            torch.tensor(self.id2label[identity], dtype=torch.long),
            torch.tensor(self.action2label[action], dtype=torch.long),
        )
        return sequence, labels


class ImgRDSdataset(_RDSSequenceDataset):
    """Load all standard RDS action folders (1 through 5)."""

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, actions=("1", "2", "3", "4", "5"), transform=transform)


class ImgRDSdatasetAction2(_RDSSequenceDataset):
    """Load a caller-selected set of RDS action folders."""

    def __init__(self, root_dir, actions=("1", "5"), transform=None):
        super().__init__(root_dir, actions=actions, transform=transform)


class ImgRDSdatasetAction(_RDSSequenceDataset):
    """Load action sequences, optionally splitting action 1 into query/gallery."""

    def __init__(
        self,
        root_dir,
        action="1",
        if_bi=False,
        if_gallery=False,
        transform=None,
        if_crop=True,
    ):
        actions = [action] if isinstance(action, str) else action
        super().__init__(
            root_dir,
            actions=actions,
            transform=transform,
            gallery_split=if_gallery,
            binary=if_bi,
            crop=if_crop,
        )
