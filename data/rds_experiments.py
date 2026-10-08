"""Shared RDS reader for the retained semi-supervised experiments."""

from __future__ import annotations

import os
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def _load_sequence_sample(args):
    """Top-level worker so samples can be loaded by a multiprocessing pool."""
    sequence_path, id2label, action2label, transform, include_route = args
    sequence_dir = Path(sequence_path)
    sequence_parts = sequence_dir.name.split("_")
    if len(sequence_parts) < 3:
        raise ValueError(f"Unexpected RDS sequence directory name: {sequence_dir.name}")

    identity = int(sequence_dir.parent.parent.name)
    action = sequence_dir.parent.name
    if sequence_parts[2].startswith("n"):
        identity += 126

    route = None
    if include_route:
        if len(sequence_parts) < 4:
            raise ValueError(f"Route label is missing from sequence name: {sequence_dir.name}")
        route = int(sequence_parts[3])

    frames = []
    for frame_index in range(20):
        frame_path = sequence_dir / f"{frame_index}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(f"Expected frame is missing: {frame_path}")
        with Image.open(frame_path) as source:
            image = source.convert("L")
            width, height = image.size
            image = image.crop((17, 0, width - 17, height))
            if transform is not None:
                image = transform(image)
            frames.append(image)

    sequence = torch.stack(frames, dim=1).squeeze(0)
    labels = [
        torch.tensor(id2label[identity], dtype=torch.long),
        torch.tensor(action2label.get(action, -1), dtype=torch.long),
    ]
    if include_route:
        labels.append(torch.tensor(route, dtype=torch.long))
    return sequence, tuple(labels)


class RDSExperimentDataset(Dataset):
    """Load alternate RDS layouts used by retained research experiments.

    This reader preserves the shifted identity labels and optional route label
    needed by semi-supervised experiments. The maintained workflow uses
    :class:`data.rds._RDSSequenceDataset` instead.
    """

    def __init__(
        self,
        root_dir,
        actions: Sequence[str] = ("1",),
        *,
        transform=None,
        gallery_split: Optional[bool] = None,
        gallery_fraction: float = 0.5,
        action_classes: int = 5,
        preload: str = "none",
        include_route: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser()
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")
        if preload not in {"none", "sequential", "parallel"}:
            raise ValueError("preload must be 'none', 'sequential', or 'parallel'.")

        self.transform = transform
        self.include_route = include_route
        self.actions = tuple(str(action) for action in actions)
        self.action2label = {
            str(action): index for index, action in enumerate(range(1, action_classes + 1))
        }

        all_paths = sorted(
            Path(root)
            for root, _, files in os.walk(self.root_dir)
            if files
        )
        matching_paths = [path for path in all_paths if path.parent.name in self.actions]
        self.sequence_dirs = self._split_gallery(
            matching_paths,
            gallery_split=gallery_split,
            gallery_fraction=gallery_fraction,
        )

        identities = sorted(
            int(path.name)
            for path in self.root_dir.iterdir()
            if path.is_dir()
        )
        identities = sorted(identities + [identity + 126 for identity in identities])
        self.id2label = {identity: label for label, identity in enumerate(identities)}

        self.data_cache = None
        if preload == "sequential":
            self.data_cache = [
                self._load_sample(path)
                for path in tqdm(self.sequence_dirs, desc="Loading RDS samples")
            ]
        elif preload == "parallel":
            process_count = max(1, cpu_count() // 2)
            args = [
                (
                    str(path),
                    self.id2label,
                    self.action2label,
                    self.transform,
                    self.include_route,
                )
                for path in self.sequence_dirs
            ]
            with Pool(processes=process_count) as pool:
                self.data_cache = list(
                    tqdm(
                        pool.imap(_load_sequence_sample, args),
                        total=len(args),
                        desc="Preloading RDS samples",
                    )
                )

    @staticmethod
    def _split_gallery(paths, *, gallery_split, gallery_fraction):
        if gallery_split is None:
            return paths
        if not paths:
            return []

        gallery_paths = random.Random(42).sample(
            paths,
            max(1, int(len(paths) * gallery_fraction)),
        )
        remaining_paths = list(set(paths) - set(gallery_paths))
        return gallery_paths if gallery_split else remaining_paths

    def _load_sample(self, sequence_dir: Path):
        return _load_sequence_sample(
            (
                str(sequence_dir),
                self.id2label,
                self.action2label,
                self.transform,
                self.include_route,
            )
        )

    def __getitem__(self, index: int):
        if self.data_cache is not None:
            return self.data_cache[index]
        return self._load_sample(self.sequence_dirs[index])

    def __len__(self) -> int:
        return len(self.sequence_dirs)
