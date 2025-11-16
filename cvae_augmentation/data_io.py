"""
Data I/O utilities for cVAE training and sampling.

This module provides functions to load metadata, per-day files, and static maps,
as well as PyTorch Dataset classes for training.
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_split(path: str) -> Dict[str, List[int]]:
    """
    Load train/val/test split from JSON file.

    Args:
        path: Path to split.json

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to lists of day indices
    """
    with open(path, 'r') as f:
        split = json.load(f)
    return split


def load_categories(path: str) -> Dict[str, str]:
    """
    Load day categories from JSON file.

    Args:
        path: Path to categories.json

    Returns:
        Dictionary mapping day_id (str) to category (str):
        'dry', 'moderate', 'heavy_coast', 'heavy_interior'
    """
    with open(path, 'r') as f:
        categories = json.load(f)
    return categories


def load_thresholds(path: str) -> Dict[str, float]:
    """
    Load precipitation thresholds from JSON file.

    Args:
        path: Path to thresholds.json

    Returns:
        Dictionary with keys 'P95', 'P99' mapping to threshold values
    """
    with open(path, 'r') as f:
        thresholds = json.load(f)
    return thresholds


def load_h_w(path: str) -> Dict[str, int]:
    """
    Load high-resolution grid dimensions.

    Args:
        path: Path to H_W.json

    Returns:
        Dictionary with keys 'H', 'W'
    """
    with open(path, 'r') as f:
        h_w = json.load(f)
    return h_w


def load_day_X_lr(root: str, day_id: int) -> np.ndarray:
    """
    Load low-resolution input for a single day.

    Args:
        root: Root data directory (should contain X_lr subdirectory)
        day_id: Day index

    Returns:
        Array of shape (C, H_lr, W_lr) in float32
    """
    path = Path(root) / "X_lr" / f"day_{day_id:06d}.npy"
    data = np.load(path)
    return data.astype(np.float32)


def load_day_Y_hr(root: str, day_id: int) -> np.ndarray:
    """
    Load high-resolution precipitation for a single day.

    Args:
        root: Root data directory (should contain Y_hr subdirectory)
        day_id: Day index

    Returns:
        Array of shape (1, H, W) in float32
    """
    path = Path(root) / "Y_hr" / f"day_{day_id:06d}.npy"
    data = np.load(path)
    return data.astype(np.float32)


def load_statics(root: str, use_land_sea: bool = True,
                 use_dist_coast: bool = True,
                 use_elevation: bool = False) -> Dict[str, np.ndarray]:
    """
    Load static high-resolution maps.

    Args:
        root: Root data directory (should contain statics subdirectory)
        use_land_sea: Whether to load land/sea mask
        use_dist_coast: Whether to load distance to coast
        use_elevation: Whether to load elevation (if available)

    Returns:
        Dictionary mapping static map names to arrays of shape (1, H, W)
    """
    statics_dir = Path(root) / "statics"
    statics = {}

    if use_land_sea:
        path = statics_dir / "land_sea_mask.npy"
        if path.exists():
            statics['land_sea_mask'] = np.load(path).astype(np.float32)
        else:
            raise FileNotFoundError(f"land_sea_mask.npy not found at {path}")

    if use_dist_coast:
        path = statics_dir / "dist_to_coast_km.npy"
        if path.exists():
            statics['dist_to_coast_km'] = np.load(path).astype(np.float32)
        else:
            raise FileNotFoundError(f"dist_to_coast_km.npy not found at {path}")

    if use_elevation:
        path = statics_dir / "elevation_m.npy"
        if path.exists():
            statics['elevation_m'] = np.load(path).astype(np.float32)
        else:
            print(f"Warning: elevation_m.npy not found at {path}, skipping")

    return statics


class CvaeDataset(Dataset):
    """
    PyTorch Dataset for cVAE training/validation.

    Returns:
        X_lr: (C, H_lr, W_lr) low-resolution inputs
        Y_hr: (1, H, W) high-resolution precipitation
        S: (S_ch, H, W) static maps stacked
        mask_land: (1, H, W) land mask for budget loss
        day_id: int
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 use_land_sea: bool = True,
                 use_dist_coast: bool = True,
                 use_elevation: bool = False,
                 normalize_statics: bool = True):
        """
        Args:
            data_root: Root directory containing data/ subdirectory
            split: One of 'train', 'val', 'test'
            use_land_sea: Include land/sea mask in statics
            use_dist_coast: Include distance to coast in statics
            use_elevation: Include elevation in statics (if available)
            normalize_statics: Whether to normalize distance to coast
        """
        super().__init__()

        self.data_root = Path(data_root) / "data"
        self.split = split
        self.normalize_statics = normalize_statics

        # Load split
        split_dict = load_split(self.data_root / "metadata" / "split.json")
        self.day_ids = split_dict[split]

        print(f"[CvaeDataset] Loaded {split} split: {len(self.day_ids)} days")

        # Load statics (shared across all samples)
        statics_dict = load_statics(
            self.data_root.parent / "data",
            use_land_sea=use_land_sea,
            use_dist_coast=use_dist_coast,
            use_elevation=use_elevation
        )

        # Stack statics in order
        static_list = []
        self.static_names = []

        if use_land_sea and 'land_sea_mask' in statics_dict:
            static_list.append(statics_dict['land_sea_mask'])
            self.static_names.append('land_sea_mask')

        if use_dist_coast and 'dist_to_coast_km' in statics_dict:
            dist = statics_dict['dist_to_coast_km']
            if normalize_statics:
                # Normalize distance to coast to [0, 1] range
                dist_max = dist.max()
                if dist_max > 0:
                    dist = dist / dist_max
            static_list.append(dist)
            self.static_names.append('dist_to_coast_km')

        if use_elevation and 'elevation_m' in statics_dict:
            elev = statics_dict['elevation_m']
            if normalize_statics:
                # Normalize elevation
                elev_min, elev_max = elev.min(), elev.max()
                if elev_max > elev_min:
                    elev = (elev - elev_min) / (elev_max - elev_min)
            static_list.append(elev)
            self.static_names.append('elevation_m')

        if len(static_list) == 0:
            raise ValueError("No static maps loaded. Check configuration.")

        self.S = np.concatenate(static_list, axis=0)  # (S_ch, H, W)
        self.S = torch.from_numpy(self.S).float()

        # Extract land mask for budget loss
        if 'land_sea_mask' in statics_dict:
            self.mask_land = torch.from_numpy(statics_dict['land_sea_mask']).float()
        else:
            # If no land mask, use all ones (use all pixels)
            print("Warning: No land mask available, using all pixels for budget loss")
            self.mask_land = torch.ones(1, self.S.shape[1], self.S.shape[2])

        print(f"[CvaeDataset] Static maps: {self.static_names}, shape: {self.S.shape}")

    def __len__(self):
        return len(self.day_ids)

    def __getitem__(self, idx):
        day_id = self.day_ids[idx]

        # Load X_lr and Y_hr
        X_lr = load_day_X_lr(self.data_root.parent / "data", day_id)
        Y_hr = load_day_Y_hr(self.data_root.parent / "data", day_id)

        # Convert to torch tensors
        X_lr = torch.from_numpy(X_lr).float()
        Y_hr = torch.from_numpy(Y_hr).float()

        return {
            'X_lr': X_lr,           # (C, H_lr, W_lr)
            'Y_hr': Y_hr,           # (1, H, W)
            'S': self.S,            # (S_ch, H, W)
            'mask_land': self.mask_land,  # (1, H, W)
            'day_id': day_id
        }


def get_dataloaders(data_root: str,
                   batch_size: int = 4,
                   num_workers: int = 0,
                   use_land_sea: bool = True,
                   use_dist_coast: bool = True,
                   use_elevation: bool = False,
                   normalize_statics: bool = True) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train, val, and test dataloaders.

    Args:
        data_root: Root directory containing data/ subdirectory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        use_land_sea: Include land/sea mask
        use_dist_coast: Include distance to coast
        use_elevation: Include elevation
        normalize_statics: Normalize static maps

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = CvaeDataset(
        data_root, split='train',
        use_land_sea=use_land_sea,
        use_dist_coast=use_dist_coast,
        use_elevation=use_elevation,
        normalize_statics=normalize_statics
    )

    val_dataset = CvaeDataset(
        data_root, split='val',
        use_land_sea=use_land_sea,
        use_dist_coast=use_dist_coast,
        use_elevation=use_elevation,
        normalize_statics=normalize_statics
    )

    test_dataset = CvaeDataset(
        data_root, split='test',
        use_land_sea=use_land_sea,
        use_dist_coast=use_dist_coast,
        use_elevation=use_elevation,
        normalize_statics=normalize_statics
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
