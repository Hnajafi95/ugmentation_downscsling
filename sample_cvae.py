"""
Sampling script for cVAE model.

Generates synthetic high-resolution precipitation samples for heavy precipitation days
using a trained cVAE model.
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from model_cvae import CVAE
from data_io import load_split, load_categories, load_statics, load_day_X_lr, load_day_Y_hr


def denormalize_to_mmday(y_normalized, mean_pr, std_pr):
    """
    Convert normalized precipitation to mm/day.

    Args:
        y_normalized: (H, W) or (1, H, W) normalized log-transformed precipitation
        mean_pr: (H, W) mean of log1p(precipitation)
        std_pr: (H, W) std of log1p(precipitation)

    Returns:
        y_mmday: (H, W) or (1, H, W) precipitation in mm/day
    """
    # Handle 3D case (1, H, W)
    if y_normalized.ndim == 3 and y_normalized.shape[0] == 1:
        y_normalized_2d = y_normalized[0]  # (H, W)
        was_3d = True
    else:
        y_normalized_2d = y_normalized
        was_3d = False

    # Reverse z-score normalization
    log1p_precip = y_normalized_2d * std_pr + mean_pr  # (H, W)

    # Reverse log1p transform
    y_mmday = np.expm1(log1p_precip)  # exp(x) - 1

    # Clip negatives
    y_mmday = np.maximum(y_mmday, 0.0)

    # Restore 3D shape if needed
    if was_3d:
        y_mmday = y_mmday[np.newaxis, :, :]  # (1, H, W)

    return y_mmday


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', -1)
    print(f"Loaded checkpoint from epoch {epoch+1}")
    return model


def get_day_list(args, config):
    """
    Determine which days to generate samples for.

    Args:
        args: Command-line arguments
        config: Configuration dictionary

    Returns:
        day_ids: List of day indices to generate samples for
    """
    data_root = Path(config['data_root']) / "data"

    if args.days == "heavy_only":
        # Get all heavy days from categories
        categories_path = data_root / "metadata" / "categories.json"
        categories = load_categories(categories_path)

        day_ids = []
        for day_id_str, category in categories.items():
            if category in ["heavy_coast", "heavy_interior"]:
                day_ids.append(int(day_id_str))

        day_ids.sort()
        print(f"Found {len(day_ids)} heavy precipitation days")

    elif args.days.startswith("split:"):
        # Get days from a specific split
        split_name = args.days.split(":")[1]
        split_path = data_root / "metadata" / "split.json"
        split = load_split(split_path)

        if split_name not in split:
            raise ValueError(f"Unknown split: {split_name}. Available: {list(split.keys())}")

        day_ids = split[split_name]
        print(f"Using {len(day_ids)} days from split '{split_name}'")

    elif args.days.startswith("file:"):
        # Load days from a file
        file_path = args.days.split(":", 1)[1]
        with open(file_path, 'r') as f:
            day_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(day_ids)} days from {file_path}")

    else:
        raise ValueError(f"Unknown days specification: {args.days}")

    return day_ids


def main(args):
    """Main sampling function."""
    print("=" * 80)
    print("cVAE Sampling")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")

    # Override config with command-line arguments
    if args.mode is not None:
        config['sampling']['mode'] = args.mode
    if args.K is not None:
        config['sampling']['K'] = args.K
    if args.days is not None:
        config['sampling']['days'] = args.days
    if args.min_threshold is not None:
        config['sampling']['min_threshold'] = args.min_threshold

    print(f"\nSampling configuration:")
    print(f"  Mode: {config['sampling']['mode']}")
    print(f"  Samples per day (K): {config['sampling']['K']}")
    print(f"  Min threshold: {config['sampling']['min_threshold']}")

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    outputs_root = Path(config['outputs_root'])
    synth_dir = outputs_root / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {synth_dir}")

    # Load model
    print("\nCreating model...")
    model = CVAE(
        in_channels_X=config['model']['in_channels_X'],
        in_channels_Y=config['model']['in_channels_Y'],
        static_channels=config['model']['static_channels'],
        d_x=config['model']['d_x'],
        d_y=config['model']['d_y'],
        d_z=config['model']['d_z'],
        H=config['model']['H'],
        W=config['model']['W'],
        base_filters=config['model']['base_filters']
    ).to(device)

    # Load checkpoint
    checkpoint_path = args.checkpoint if args.checkpoint else outputs_root / "checkpoints" / "cvae_best.pt"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_checkpoint(checkpoint_path, model, device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Load static maps
    print("\nLoading static maps...")
    data_root = Path(config['data_root']) / "data"
    statics_dict = load_statics(
        data_root.parent / "data",
        use_land_sea=config['statics']['use_land_sea'],
        use_dist_coast=config['statics']['use_dist_coast'],
        use_elevation=config['statics']['use_elevation']
    )

    # Stack statics
    static_list = []
    if config['statics']['use_land_sea'] and 'land_sea_mask' in statics_dict:
        static_list.append(statics_dict['land_sea_mask'])
    if config['statics']['use_dist_coast'] and 'dist_to_coast_km' in statics_dict:
        dist = statics_dict['dist_to_coast_km']
        if config['statics']['normalize_statics']:
            dist_max = dist.max()
            if dist_max > 0:
                dist = dist / dist_max
        static_list.append(dist)
    if config['statics']['use_elevation'] and 'elevation_m' in statics_dict:
        elev = statics_dict['elevation_m']
        if config['statics']['normalize_statics']:
            elev_min, elev_max = elev.min(), elev.max()
            if elev_max > elev_min:
                elev = (elev - elev_min) / (elev_max - elev_min)
        static_list.append(elev)

    S = np.concatenate(static_list, axis=0)  # (S_ch, H, W)
    S_tensor = torch.from_numpy(S).float().to(device)  # (S_ch, H, W)

    # Get land mask for mass threshold check
    land_mask = statics_dict['land_sea_mask'][0]  # (H, W)

    # CRITICAL: Load normalization parameters for denormalization
    print("\nLoading normalization parameters...")
    norm_dir = data_root / "metadata"
    mean_path = norm_dir / "ERA5_mean_train.npy"
    std_path = norm_dir / "ERA5_std_train.npy"

    if mean_path.exists() and std_path.exists():
        mean_pr = np.load(mean_path)  # (H, W)
        std_pr = np.load(std_path)    # (H, W)
        print(f"✓ Loaded normalization parameters from {norm_dir}")
    else:
        raise FileNotFoundError(f"Normalization files not found in {norm_dir}")

    # Get day list
    day_ids = get_day_list(args, config)

    if len(day_ids) == 0:
        print("\nNo days to sample. Exiting.")
        return

    # Sample generation
    print("\n" + "=" * 80)
    print(f"Generating {config['sampling']['K']} samples for {len(day_ids)} days...")
    print("=" * 80)

    mode = config['sampling']['mode']
    K = config['sampling']['K']
    min_threshold = config['sampling']['min_threshold']
    temperature = config['sampling'].get('temperature', 1.0)

    # OPTIONAL: Pixel-wise threshold to match physical detection limits
    # Any pixel with value < pixel_threshold will be set to exactly 0.0
    # Default 0.1 mm/day matches typical rain gauge detection limits
    pixel_threshold = config['sampling'].get('pixel_threshold', 0.1)  # mm/day
    print(f"Pixel threshold: {pixel_threshold} mm/day (physical detection limit)")

    n_generated = 0
    n_discarded = 0

    with torch.no_grad():
        for day_id in tqdm(day_ids, desc="Generating samples"):
            # Load X_lr for this day
            X_lr_np = load_day_X_lr(data_root.parent / "data", day_id)  # (C, H_lr, W_lr)
            X_lr = torch.from_numpy(X_lr_np).float().unsqueeze(0).to(device)  # (1, C, H_lr, W_lr)

            # Expand S to batch size
            S_batch = S_tensor.unsqueeze(0)  # (1, S_ch, H, W)

            # If posterior mode, also load Y_hr to compute mu, logvar
            if mode == "posterior":
                Y_hr_normalized = load_day_Y_hr(data_root.parent / "data", day_id)  # (1, H, W) normalized
                Y_hr = torch.from_numpy(Y_hr_normalized).float().unsqueeze(0).to(device)  # (1, 1, H, W)

                # Encode to get posterior (returns spatial_X too!)
                mu, logvar, h_X, spatial_X = model.encode(X_lr, Y_hr, S_batch)

                # Compute reference mass for threshold check (in mm/day space)
                Y_hr_mmday = denormalize_to_mmday(Y_hr_normalized, mean_pr, std_pr)  # (1, H, W)
                mass_ref = float((Y_hr_mmday[0] * land_mask).sum())
            else:
                # Prior mode: sample z ~ N(0, I)
                mu = None
                logvar = None
                # Get both h_X and spatial_X from encoder
                h_X, spatial_X = model.encoder_X(X_lr)
                mass_ref = None  # No reference in prior mode

            # Generate K samples for this day
            for k in range(K):
                # Sample z
                if mode == "posterior":
                    # Sample from posterior
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std) * temperature
                    z = mu + eps * std
                else:
                    # Sample from prior
                    z = torch.randn(1, model.d_z, device=device) * temperature

                # Decode with spatial conditioning
                Y_hat = model.decode(z, h_X, spatial_X)  # (1, 1, H, W)

                # Convert to numpy (still in normalized space)
                Y_hat_normalized = Y_hat.cpu().numpy()[0]  # (1, H, W)

                # CRITICAL FIX: Denormalize BEFORE applying pixel threshold
                # The model outputs in log1p + z-scored space, we need mm/day for thresholding
                Y_hat_mmday = denormalize_to_mmday(Y_hat_normalized, mean_pr, std_pr)  # (1, H, W)

                # Apply pixel-wise thresholding in mm/day space
                # This fixes the "100% wet fraction" problem by zeroing out drizzle
                Y_hat_mmday[Y_hat_mmday < pixel_threshold] = 0.0

                # Check minimum threshold (only in posterior mode)
                # Use mm/day values for mass check
                if mode == "posterior" and mass_ref is not None:
                    mass_gen = float((Y_hat_mmday[0] * land_mask).sum())
                    if mass_gen < min_threshold * mass_ref:
                        n_discarded += 1
                        continue  # Discard this sample

                # Re-normalize before saving (to match training data format)
                # Convert back to normalized log-space for consistency with training data
                Y_hat_mmday_safe = np.maximum(Y_hat_mmday, 0.0)  # Ensure non-negative
                log1p_precip = np.log1p(Y_hat_mmday_safe)  # log(1 + x)
                Y_hat_renormalized = (log1p_precip - mean_pr) / (std_pr + 1e-8)  # Avoid div by zero

                # Save synthetic Y_hr in normalized format (to match real data)
                Y_hr_syn_path = synth_dir / f"day_{day_id:06d}_sample_{k:02d}.Y_hr_syn.npy"
                np.save(Y_hr_syn_path, Y_hat_renormalized.astype(np.float32))

                # Save reference X_lr (copy-through)
                X_lr_ref_path = synth_dir / f"day_{day_id:06d}_sample_{k:02d}.X_lr_ref.npy"
                np.save(X_lr_ref_path, X_lr_np.astype(np.float32))

                n_generated += 1

    print("\n" + "=" * 80)
    print("Sampling complete!")
    print(f"Generated: {n_generated} samples")
    print(f"Discarded: {n_discarded} samples (below mass threshold)")
    print(f"Output directory: {synth_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic precipitation samples using cVAE")
    parser.add_argument("--config", type=str,
                        default="/home/user/ugmentation_downscsling/cvae_augmentation_V1/config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: best checkpoint)")
    parser.add_argument("--mode", type=str, choices=["posterior", "prior"], default=None,
                        help="Sampling mode (overrides config)")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of samples per day (overrides config)")
    parser.add_argument("--days", type=str, default=None,
                        help="Which days to sample: 'heavy_only', 'split:val', 'file:path.txt' (overrides config)")
    parser.add_argument("--min_threshold", type=float, default=None,
                        help="Minimum mass threshold as fraction of reference (overrides config)")

    args = parser.parse_args()
    main(args)
