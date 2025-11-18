"""
Compute accurate P99 threshold in mm/day space from training data.

This script:
1. Loads normalized training data (Y_hr)
2. Loads normalization parameters (mean_pr, std_pr)
3. Denormalizes to mm/day space
4. Computes P99 across all land pixels
5. Saves p99_mmday.npy for use in simplified loss
"""

import numpy as np
from pathlib import Path
import argparse


def compute_p99_mmday(data_root, output_path=None):
    """
    Compute P99 in mm/day space from training data.

    Args:
        data_root: Path to cVAE data directory
        output_path: Where to save p99_mmday.npy (default: data_root/data/metadata/)

    Returns:
        p99_mmday: P99 threshold in mm/day
    """
    data_root = Path(data_root)

    # Load normalization parameters
    norm_dir = data_root / "data" / "metadata"
    mean_pr = np.load(norm_dir / "ERA5_mean_train.npy")  # (H, W)
    std_pr = np.load(norm_dir / "ERA5_std_train.npy")    # (H, W)
    land_mask = np.load(data_root / "data" / "statics" / "land_sea_mask.npy")  # (H, W)

    print(f"Loaded normalization parameters:")
    print(f"  mean_pr shape: {mean_pr.shape}")
    print(f"  std_pr shape: {std_pr.shape}")
    print(f"  land_mask shape: {land_mask.shape}")
    print(f"  Land pixels: {land_mask.sum()}")

    # Load normalized training data
    Y_hr_dir = data_root / "data" / "Y_hr"

    # Load split to get training day IDs
    import json
    with open(norm_dir / "split.json") as f:
        split = json.load(f)

    train_days = split['train']
    print(f"\nLoading {len(train_days)} training days...")

    # Collect all mm/day values from land pixels
    all_mm_values = []

    for i, day_id in enumerate(train_days):
        if (i + 1) % 1000 == 0:
            print(f"  Processing day {i+1}/{len(train_days)}...")

        # Load normalized data
        Y_norm = np.load(Y_hr_dir / f"day_{day_id:06d}.npy")  # (1, H, W) or (H, W)

        if Y_norm.ndim == 3:
            Y_norm = Y_norm[0]  # Remove channel dimension

        # Denormalize to log1p space: logp = y * std + mean (pixel-wise!)
        logp = Y_norm * std_pr + mean_pr

        # Convert to mm/day: mm = exp(logp) - 1
        mm = np.expm1(logp)

        # Extract land pixels only
        mm_land = mm[land_mask]

        # Collect values
        all_mm_values.append(mm_land)

    # Concatenate all values
    print("\nConcatenating all values...")
    all_mm_values = np.concatenate(all_mm_values)

    print(f"Total land pixel values: {len(all_mm_values):,}")
    print(f"  Min: {all_mm_values.min():.2f} mm/day")
    print(f"  Max: {all_mm_values.max():.2f} mm/day")
    print(f"  Mean: {all_mm_values.mean():.2f} mm/day")
    print(f"  Median: {np.median(all_mm_values):.2f} mm/day")

    # Compute percentiles
    p95 = np.percentile(all_mm_values, 95)
    p99 = np.percentile(all_mm_values, 99)
    p999 = np.percentile(all_mm_values, 99.9)

    print(f"\nPercentiles in mm/day:")
    print(f"  P95:  {p95:.2f}")
    print(f"  P99:  {p99:.2f}")
    print(f"  P99.9: {p999:.2f}")

    # Save P99
    if output_path is None:
        output_path = norm_dir / "p99_mmday.npy"
    else:
        output_path = Path(output_path)

    np.save(output_path, p99)
    print(f"\nSaved P99 (mm/day) = {p99:.2f} to {output_path}")

    return p99


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute P99 in mm/day space")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation",
        help="Path to cVAE data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for p99_mmday.npy (default: data_root/data/metadata/p99_mmday.npy)"
    )

    args = parser.parse_args()

    p99 = compute_p99_mmday(args.data_root, args.output)

    print("\n" + "="*80)
    print("Done! You can now retrain with:")
    print("  python train_cvae.py --config config_simplified.yaml")
    print("="*80)
