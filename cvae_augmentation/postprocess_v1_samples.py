"""
Post-process V1 samples to fix wet fraction issue.

V1 samples have perfect magnitudes but 100% wet fraction (drizzle everywhere).
This script applies pixel thresholding in mm/day space to create sharp wet/dry boundaries.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def denormalize_to_mmday(y_normalized, mean_pr, std_pr):
    """Convert normalized precipitation to mm/day."""
    if y_normalized.ndim == 3 and y_normalized.shape[0] == 1:
        y_normalized_2d = y_normalized[0]
        was_3d = True
    else:
        y_normalized_2d = y_normalized
        was_3d = False

    # Reverse z-score normalization
    log1p_precip = y_normalized_2d * std_pr + mean_pr
    # Reverse log1p transform
    y_mmday = np.expm1(log1p_precip)
    y_mmday = np.maximum(y_mmday, 0.0)

    if was_3d:
        y_mmday = y_mmday[np.newaxis, :, :]
    return y_mmday


def normalize_from_mmday(y_mmday, mean_pr, std_pr):
    """Convert mm/day back to normalized space."""
    if y_mmday.ndim == 3 and y_mmday.shape[0] == 1:
        y_mmday_2d = y_mmday[0]
        was_3d = True
    else:
        y_mmday_2d = y_mmday
        was_3d = False

    # Apply log1p transform
    y_mmday_safe = np.maximum(y_mmday_2d, 0.0)
    log1p_precip = np.log1p(y_mmday_safe)

    # Apply z-score normalization
    y_normalized = (log1p_precip - mean_pr) / (std_pr + 1e-8)

    if was_3d:
        y_normalized = y_normalized[np.newaxis, :, :]
    return y_normalized


def main(args):
    print("=" * 80)
    print("Post-Processing V1 Samples to Fix Wet Fraction")
    print("=" * 80)

    # Load normalization parameters
    norm_dir = Path(args.data_root) / "data" / "metadata"
    mean_pr = np.load(norm_dir / "ERA5_mean_train.npy")
    std_pr = np.load(norm_dir / "ERA5_std_train.npy")
    print(f"✓ Loaded normalization parameters from {norm_dir}")

    # Input/output directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all V1 samples
    sample_files = list(input_dir.glob("day_*_sample_*.Y_hr_syn.npy"))
    print(f"\nFound {len(sample_files)} V1 samples to process")
    print(f"Pixel threshold: {args.pixel_threshold} mm/day")

    # Process each sample
    n_processed = 0
    for sample_file in tqdm(sample_files, desc="Processing samples"):
        # Load normalized sample
        Y_normalized = np.load(sample_file)  # (1, H, W)

        # Denormalize to mm/day
        Y_mmday = denormalize_to_mmday(Y_normalized, mean_pr, std_pr)

        # Apply pixel threshold (FIX FOR WET FRACTION)
        Y_mmday[Y_mmday < args.pixel_threshold] = 0.0

        # Re-normalize
        Y_normalized_fixed = normalize_from_mmday(Y_mmday, mean_pr, std_pr)

        # Save to output directory
        output_file = output_dir / sample_file.name
        np.save(output_file, Y_normalized_fixed.astype(np.float32))
        n_processed += 1

    # Copy X_lr files (unchanged)
    x_lr_files = list(input_dir.glob("day_*_sample_*.X_lr_ref.npy"))
    for x_file in tqdm(x_lr_files, desc="Copying X_lr files"):
        output_file = output_dir / x_file.name
        np.save(output_file, np.load(x_file))

    print(f"\n✓ Processed {n_processed} samples")
    print(f"✓ Output directory: {output_dir}")
    print("=" * 80)
    print("\nNext steps:")
    print(f"  1. Evaluate: python evaluate_samples.py --synth_dir {output_dir}")
    print(f"  2. Visualize: python visualize_samples.py --synth_dir {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process V1 samples to fix wet fraction"
    )
    parser.add_argument("--input_dir", type=str,
                        default="outputs/cvae_simplified/synth",
                        help="V1 samples directory")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/cvae_v1_fixed/synth",
                        help="Output directory for fixed samples")
    parser.add_argument("--data_root", type=str,
                        default="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation",
                        help="Data root for normalization parameters")
    parser.add_argument("--pixel_threshold", type=float, default=1.0,
                        help="Pixel threshold in mm/day (default: 1.0)")

    args = parser.parse_args()
    main(args)
