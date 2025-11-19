#!/usr/bin/env python3
"""
Prepare augmented dataset for SRDRN training.

This script combines original SRDRN training data with cVAE-synthesized
heavy precipitation samples to create an augmented training dataset.

The augmented data follows the same format as SRDRN preprocessing output:
- predictors_train_mean_std_separate.npy: (T, H_lr, W_lr, C)
- obs_train_mean_std_single.npy: (T, H, W)
"""

import argparse
import numpy as np
import yaml
import json
from pathlib import Path
from glob import glob


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_original_srdrn_data(srdrn_data_dir):
    """
    Load original SRDRN training data.

    Returns:
        predictors: (T, H_lr, W_lr, C) low-res inputs
        obs: (T, H, W) high-res targets
        metadata: dict with other arrays (mean, std, etc.)
    """
    srdrn_dir = Path(srdrn_data_dir)

    # Load main arrays
    predictors = np.load(srdrn_dir / "predictors_train_mean_std_separate.npy")
    obs = np.load(srdrn_dir / "obs_train_mean_std_single.npy")

    print(f"Loaded original training data:")
    print(f"  Predictors shape: {predictors.shape}")
    print(f"  Obs shape: {obs.shape}")

    # Load metadata files
    metadata = {}
    metadata_files = [
        "ERA5_mean_train.npy",
        "ERA5_std_train.npy",
        "land_mask.npy",
        "train_time_coords.npy",
        "high_res_train_time_coords.npy",
        "train_indices.npy",
        "train_dates.npy",
        "low_res_lat.npy",
        "low_res_lon.npy",
        "high_res_lat.npy",
        "high_res_lon.npy",
        "variables.npy",
    ]

    for fname in metadata_files:
        fpath = srdrn_dir / fname
        if fpath.exists():
            metadata[fname] = np.load(fpath, allow_pickle=True)

    return predictors, obs, metadata


def load_synthesized_samples(synth_dir, config):
    """
    Load synthesized samples from cVAE output.

    Returns:
        synth_Y_hr: list of (1, H, W) arrays in z-space
        synth_X_lr: list of (C, H_lr, W_lr) arrays
        day_ids: list of day IDs
        sample_ids: list of sample indices (k values)
    """
    synth_dir = Path(synth_dir)

    # Find all synthesized Y_hr files
    # Format: day_XXXXXX_sample_YY.Y_hr_syn.npy
    synth_files = sorted(synth_dir.glob("day_*_sample_*.Y_hr_syn.npy"))

    if len(synth_files) == 0:
        raise FileNotFoundError(f"No synthesized samples found in {synth_dir}")

    print(f"Found {len(synth_files)} synthesized samples")

    synth_Y_hr = []
    synth_X_lr = []
    day_ids = []
    sample_ids = []

    for synth_file in synth_files:
        # Parse filename: day_000228_sample_00.Y_hr_syn.npy
        parts = synth_file.stem.split('_')
        day_id = int(parts[1])
        sample_id = int(parts[3].split('.')[0])

        # Load Y_hr (synthesized high-res)
        Y_hr = np.load(synth_file)

        # Load corresponding X_lr (reference low-res)
        # Format: day_XXXXXX_sample_YY.X_lr_ref.npy
        X_lr_file = synth_dir / f"day_{day_id:06d}_sample_{sample_id:02d}.X_lr_ref.npy"

        if not X_lr_file.exists():
            print(f"Warning: X_lr reference not found for day {day_id} sample {sample_id}, skipping")
            continue

        X_lr = np.load(X_lr_file)

        synth_Y_hr.append(Y_hr)
        synth_X_lr.append(X_lr)
        day_ids.append(day_id)
        sample_ids.append(sample_id)

    print(f"Loaded {len(synth_Y_hr)} complete sample pairs")

    return synth_Y_hr, synth_X_lr, day_ids, sample_ids


def apply_postprocessing(synth_Y_hr, config, options):
    """
    Apply optional post-processing to synthesized samples.

    Options:
        - threshold_filter: Set values below threshold to 0 (restore dry regions)
        - scale_extremes: Scale up extreme values
    """
    processed = []

    for Y_hr in synth_Y_hr:
        Y = Y_hr.copy()

        # Option 2: Threshold filtering to restore dry regions
        if options.get('threshold_filter', False):
            threshold = options.get('threshold_value', 0.1)  # in z-space
            Y = np.where(np.abs(Y) < threshold, 0, Y)

        # Option 3: Scale up extremes
        if options.get('scale_extremes', False):
            scale_factor = options.get('scale_factor', 1.1)
            percentile = options.get('scale_percentile', 90)

            # Compute threshold in z-space
            p_thresh = np.percentile(Y[Y > 0], percentile) if np.any(Y > 0) else 0
            high_mask = Y > p_thresh
            Y[high_mask] *= scale_factor

        processed.append(Y)

    return processed


def convert_to_srdrn_format(synth_Y_hr, synth_X_lr):
    """
    Convert synthesized samples to SRDRN format.

    cVAE format:
        Y_hr: (1, H, W) - channel first
        X_lr: (C, H_lr, W_lr) - channel first

    SRDRN format:
        obs: (H, W) - no channel dimension
        predictors: (H_lr, W_lr, C) - channel last
    """
    obs_list = []
    predictors_list = []

    for Y_hr, X_lr in zip(synth_Y_hr, synth_X_lr):
        # Y_hr: (1, H, W) -> (H, W)
        obs = Y_hr.squeeze(0) if Y_hr.ndim == 3 else Y_hr

        # X_lr: (C, H_lr, W_lr) -> (H_lr, W_lr, C)
        predictors = np.transpose(X_lr, (1, 2, 0))

        obs_list.append(obs)
        predictors_list.append(predictors)

    return obs_list, predictors_list


def create_augmented_dataset(original_predictors, original_obs,
                             synth_predictors, synth_obs,
                             metadata, day_ids, sample_ids):
    """
    Combine original and synthesized data into augmented dataset.
    """
    # Stack synthesized samples
    synth_obs_array = np.stack(synth_obs, axis=0)
    synth_pred_array = np.stack(synth_predictors, axis=0)

    print(f"\nSynthesized data shapes:")
    print(f"  Obs: {synth_obs_array.shape}")
    print(f"  Predictors: {synth_pred_array.shape}")

    # Concatenate with original data
    augmented_obs = np.concatenate([original_obs, synth_obs_array], axis=0)
    augmented_pred = np.concatenate([original_predictors, synth_pred_array], axis=0)

    print(f"\nAugmented data shapes:")
    print(f"  Obs: {augmented_obs.shape}")
    print(f"  Predictors: {augmented_pred.shape}")

    # Update indices
    original_n = len(original_obs)
    augmented_n = len(augmented_obs)

    augmented_indices = np.arange(augmented_n)

    # Create augmented dates/labels for the synthetic samples
    # Format: synth_dayID_sampleK
    original_dates = metadata.get('train_dates.npy', np.array([]))
    synth_dates = np.array([f"synth_{day_id:06d}_s{sample_id:02d}"
                           for day_id, sample_id in zip(day_ids, sample_ids)])

    if len(original_dates) > 0:
        augmented_dates = np.concatenate([original_dates, synth_dates])
    else:
        augmented_dates = synth_dates

    # Create augmented time coordinates
    # For synthetic samples, use a marker timestamp to distinguish them from real data
    # Synthetic samples don't have real historical timestamps since they're augmentations
    original_time_coords = metadata.get('train_time_coords.npy', None)
    if original_time_coords is not None:
        # Create marker timestamps for synthetic samples
        # Use a far-future date that's clearly synthetic: 2099-01-01 + sample offset
        base_synth_date = np.datetime64('2099-01-01')
        synth_time_coords = np.array([base_synth_date + np.timedelta64(i, 'D')
                                      for i in range(len(day_ids))], dtype='datetime64[D]')
        augmented_time_coords = np.concatenate([original_time_coords, synth_time_coords])
    else:
        augmented_time_coords = None

    return (augmented_pred, augmented_obs, augmented_indices, augmented_dates,
            augmented_time_coords)


def save_augmented_dataset(output_dir, augmented_pred, augmented_obs,
                          augmented_indices, augmented_dates, augmented_time_coords,
                          metadata, original_n, synth_n):
    """
    Save augmented dataset in SRDRN format.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main augmented arrays
    np.save(output_dir / "predictors_train_mean_std_separate.npy", augmented_pred)
    np.save(output_dir / "obs_train_mean_std_single.npy", augmented_obs)
    np.save(output_dir / "train_indices.npy", augmented_indices)
    np.save(output_dir / "train_dates.npy", augmented_dates)

    # Save augmented time coordinates (ensure datetime64 dtype as in preprocessing.py)
    if augmented_time_coords is not None:
        # Explicitly convert to datetime64 dtype to match preprocessing.py format
        time_coords_dt64 = augmented_time_coords.astype('datetime64')
        np.save(output_dir / "train_time_coords.npy", time_coords_dt64)
        np.save(output_dir / "high_res_train_time_coords.npy", time_coords_dt64)

    # Copy over other metadata files unchanged (excluding train time coords which are augmented above)
    copy_files = [
        "ERA5_mean_train.npy",
        "ERA5_std_train.npy",
        "land_mask.npy",
        "low_res_lat.npy",
        "low_res_lon.npy",
        "high_res_lat.npy",
        "high_res_lon.npy",
        "variables.npy",
        # Test data (unchanged)
        "predictors_test_mean_std_separate.npy",
        "obs_test_mean_std_single.npy",
        "test_indices.npy",
        "test_dates.npy",
        "test_time_coords.npy",
        "high_res_test_time_coords.npy",
    ]

    for fname in copy_files:
        if fname in metadata:
            np.save(output_dir / fname, metadata[fname])

    # Save augmentation info
    aug_info = {
        'original_samples': int(original_n),
        'synthetic_samples': int(synth_n),
        'total_samples': int(original_n + synth_n),
        'augmentation_ratio': float(synth_n / original_n) if original_n > 0 else 0,
        'note': 'Synthetic samples have marker timestamps (2099-01-01+) to distinguish from real data',
    }

    with open(output_dir / "augmentation_info.json", 'w') as f:
        json.dump(aug_info, f, indent=2)

    print(f"\nSaved augmented dataset to: {output_dir}")
    print(f"  Original samples: {original_n}")
    print(f"  Synthetic samples: {synth_n}")
    print(f"  Total samples: {original_n + synth_n}")
    print(f"  Augmentation ratio: {synth_n / original_n * 100:.1f}%")


def main(args):
    """Main function to prepare augmented dataset."""

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print("PREPARING AUGMENTED DATASET FOR SRDRN")
    print("=" * 60)

    # Load original SRDRN data
    print("\n[1/5] Loading original SRDRN training data...")
    original_pred, original_obs, metadata = load_original_srdrn_data(args.srdrn_data)

    # Also load test data into metadata for copying
    srdrn_dir = Path(args.srdrn_data)
    test_files = [
        "predictors_test_mean_std_separate.npy",
        "obs_test_mean_std_single.npy",
        "test_indices.npy",
        "test_dates.npy",
        "test_time_coords.npy",
        "high_res_test_time_coords.npy",
    ]
    for fname in test_files:
        fpath = srdrn_dir / fname
        if fpath.exists():
            metadata[fname] = np.load(fpath, allow_pickle=True)

    # Load synthesized samples
    print("\n[2/5] Loading synthesized samples...")
    synth_Y_hr, synth_X_lr, day_ids, sample_ids = load_synthesized_samples(
        args.synth_dir, config
    )

    # Apply post-processing options
    print("\n[3/5] Applying post-processing...")
    postprocess_options = {
        # Option 2: Threshold filtering (disabled by default)
        'threshold_filter': args.threshold_filter,
        'threshold_value': args.threshold_value,

        # Option 3: Scale extremes (disabled by default)
        'scale_extremes': args.scale_extremes,
        'scale_factor': args.scale_factor,
        'scale_percentile': args.scale_percentile,
    }

    if args.threshold_filter:
        print(f"  Threshold filtering enabled: values < {args.threshold_value} -> 0")
    if args.scale_extremes:
        print(f"  Scaling extremes: values > P{args.scale_percentile} * {args.scale_factor}")
    if not args.threshold_filter and not args.scale_extremes:
        print("  No post-processing applied (using raw synthesized samples)")

    synth_Y_hr = apply_postprocessing(synth_Y_hr, config, postprocess_options)

    # Convert to SRDRN format
    print("\n[4/5] Converting to SRDRN format...")
    synth_obs, synth_pred = convert_to_srdrn_format(synth_Y_hr, synth_X_lr)

    # Create augmented dataset
    (augmented_pred, augmented_obs, augmented_indices, augmented_dates,
     augmented_time_coords) = create_augmented_dataset(
        original_pred, original_obs,
        synth_pred, synth_obs,
        metadata, day_ids, sample_ids
    )

    # Save augmented dataset
    print("\n[5/5] Saving augmented dataset...")
    save_augmented_dataset(
        args.output_dir,
        augmented_pred, augmented_obs,
        augmented_indices, augmented_dates, augmented_time_coords,
        metadata,
        len(original_obs), len(synth_obs)
    )

    print("\n" + "=" * 60)
    print("AUGMENTED DATASET PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare augmented dataset for SRDRN training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to cVAE config file"
    )

    parser.add_argument(
        "--srdrn_data",
        type=str,
        required=True,
        help="Path to original SRDRN data directory (e.g., mydata_corrected)"
    )

    parser.add_argument(
        "--synth_dir",
        type=str,
        required=True,
        help="Path to synthesized samples directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for augmented dataset"
    )

    # Option 2: Threshold filtering (disabled by default)
    parser.add_argument(
        "--threshold_filter",
        action="store_true",
        help="Enable threshold filtering to restore dry regions"
    )

    parser.add_argument(
        "--threshold_value",
        type=float,
        default=0.1,
        help="Threshold value in z-space for filtering (default: 0.1)"
    )

    # Option 3: Scale extremes (disabled by default)
    parser.add_argument(
        "--scale_extremes",
        action="store_true",
        help="Enable scaling of extreme values"
    )

    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1.1,
        help="Scale factor for extreme values (default: 1.1)"
    )

    parser.add_argument(
        "--scale_percentile",
        type=float,
        default=90,
        help="Percentile threshold for scaling (default: 90)"
    )

    args = parser.parse_args()
    main(args)
