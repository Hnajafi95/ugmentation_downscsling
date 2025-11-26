"""
Data preparation script to convert existing SRDRN numpy files into per-day format for cVAE training.

This script:
1. Reads normalized numpy arrays from preprocessing.py output
2. Splits them into per-day files (X_lr, Y_hr)
3. Generates metadata (split.json, categories.json, thresholds.json, H_W.json)
4. Creates static maps (land_sea_mask, dist_to_coast_km)
"""

import numpy as np
import os
import json
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import argparse


def compute_distance_to_coast(land_mask_2d, lat, lon):
    """
    Compute distance to coast in kilometers from a land/sea mask.

    Args:
        land_mask_2d: (H, W) bool array, True=land, False=sea
        lat: (H,) latitude array
        lon: (W,) longitude array

    Returns:
        dist_km: (H, W) array of distances in km
    """
    # Compute distance transform (in pixels)
    # For land pixels: distance to nearest sea pixel
    # For sea pixels: set to 0
    dist_pixels = distance_transform_edt(land_mask_2d)

    # Estimate pixel size in km (approximate for Florida region)
    # Latitude spacing in degrees
    lat_spacing = np.abs(np.mean(np.diff(lat)))
    lon_spacing = np.abs(np.mean(np.diff(lon)))

    # Convert to km (1 degree lat ≈ 111 km, lon varies by latitude)
    mean_lat = np.mean(lat)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))

    # Use average of lat and lon spacing
    km_per_pixel = np.sqrt((lat_spacing * km_per_deg_lat)**2 +
                           (lon_spacing * km_per_deg_lon)**2)

    dist_km = dist_pixels * km_per_pixel

    return dist_km.astype(np.float32)


def categorize_days(Y_hr_array, land_mask_2d, p95, p99):
    """
    Categorize days as dry, moderate, heavy_coast, or heavy_interior.
    
    UPDATED: Use P99 as threshold for "heavy" to be more selective.

    Args:
        Y_hr_array: (T, H, W) normalized log-scale precipitation
        land_mask_2d: (H, W) bool land mask
        p95, p99: thresholds in normalized log-scale

    Returns:
        categories: dict mapping day_id (str) to category (str)
    """
    T = Y_hr_array.shape[0]
    categories = {}

    # Define coastal region (approximate - first/last 20% of each dimension)
    H, W = land_mask_2d.shape
    coastal_mask = np.zeros_like(land_mask_2d, dtype=bool)
    border = max(int(0.2 * H), int(0.2 * W))
    coastal_mask[:border, :] = True
    coastal_mask[-border:, :] = True
    coastal_mask[:, :border] = True
    coastal_mask[:, -border:] = True

    interior_mask = land_mask_2d & ~coastal_mask

    for t in range(T):
        day_data = Y_hr_array[t]

        # Get land values
        land_values = day_data[land_mask_2d]
        max_val = land_values.max() if len(land_values) > 0 else 0
        mean_val = land_values.mean() if len(land_values) > 0 else 0

        # UPDATED LOGIC: Use P99 for heavy threshold
        if max_val >= p99:
            # Heavy rain day - check if coastal or interior
            coastal_values = day_data[coastal_mask & land_mask_2d]
            interior_values = day_data[interior_mask]

            coastal_max = coastal_values.max() if len(coastal_values) > 0 else 0
            interior_max = interior_values.max() if len(interior_values) > 0 else 0

            if interior_max >= p99:
                categories[str(t)] = "heavy_interior"
            else:
                categories[str(t)] = "heavy_coast"
        elif max_val >= p95:
            # Moderate-to-strong rain (P95-P99 range)
            categories[str(t)] = "moderate"
        elif mean_val >= p95 * 0.1:
            # Light-to-moderate rain
            categories[str(t)] = "moderate"
        else:
            # Dry or very light rain
            categories[str(t)] = "dry"

    return categories

def main(args):
    """Main data preparation function."""

    print("=" * 80)
    print("cVAE Data Preparation")
    print("=" * 80)

    # Define paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    X_lr_dir = output_dir / "data" / "X_lr"
    Y_hr_dir = output_dir / "data" / "Y_hr"
    statics_dir = output_dir / "data" / "statics"
    metadata_dir = output_dir / "data" / "metadata"

    for d in [X_lr_dir, Y_hr_dir, statics_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/6] Loading data from {input_dir}...")

    # Load training data
    X_train = np.load(input_dir / "predictors_train_mean_std_separate.npy")  # (T, H_lr, W_lr, C)
    Y_train = np.load(input_dir / "obs_train_mean_std_single.npy")  # (T, H_hr, W_hr)

    # Load test data
    X_test = np.load(input_dir / "predictors_test_mean_std_separate.npy")
    Y_test = np.load(input_dir / "obs_test_mean_std_single.npy")

    # Load land mask and coordinates
    land_mask_2d = np.load(input_dir / "land_mask.npy")  # (H_hr, W_hr)
    high_res_lat = np.load(input_dir / "high_res_lat.npy")
    high_res_lon = np.load(input_dir / "high_res_lon.npy")
    low_res_lat = np.load(input_dir / "low_res_lat.npy")
    low_res_lon = np.load(input_dir / "low_res_lon.npy")

    # Load variable names
    try:
        variables = np.load(input_dir / "variables.npy", allow_pickle=True)
        print(f"   Variables: {list(variables)}")
    except:
        print("   Variables file not found, assuming default order")
        variables = None

    print(f"   X_train shape: {X_train.shape}")
    print(f"   Y_train shape: {Y_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   Y_test shape: {Y_test.shape}")
    print(f"   Land mask shape: {land_mask_2d.shape}")

    # Get dimensions
    T_train = X_train.shape[0]
    T_test = X_test.shape[0]
    H_lr, W_lr, C = X_train.shape[1:4]
    H_hr, W_hr = Y_train.shape[1:3]

    print(f"   Low-res grid: ({H_lr}, {W_lr})")
    print(f"   High-res grid: ({H_hr}, {W_hr})")
    print(f"   Channels: {C}")

    # Transpose X arrays to (T, C, H, W) format
    print("\n[2/6] Transposing arrays to (T, C, H, W) format...")
    X_train = np.transpose(X_train, (0, 3, 1, 2))  # (T, C, H_lr, W_lr)
    X_test = np.transpose(X_test, (0, 3, 1, 2))

    # Add channel dimension to Y if needed
    if Y_train.ndim == 3:
        Y_train = Y_train[:, np.newaxis, :, :]  # (T, 1, H_hr, W_hr)
        Y_test = Y_test[:, np.newaxis, :, :]

    print(f"   X_train shape (transposed): {X_train.shape}")
    print(f"   Y_train shape: {Y_train.shape}")

    # Validation split (use last 20% of training data)
    val_size = int(0.2 * T_train)
    train_size = T_train - val_size

    print(f"\n[3/6] Saving per-day files...")
    print(f"   Train: {train_size} days")
    print(f"   Val: {val_size} days")
    print(f"   Test: {T_test} days")

    # Save training days (day_000000 to day_N-1)
    for t in range(train_size):
        day_id = t
        np.save(X_lr_dir / f"day_{day_id:06d}.npy", X_train[t].astype(np.float32))
        np.save(Y_hr_dir / f"day_{day_id:06d}.npy", Y_train[t].astype(np.float32))
        if (t + 1) % 1000 == 0:
            print(f"   Saved {t+1}/{train_size} training days...")

    # Save validation days
    for t in range(val_size):
        day_id = train_size + t
        np.save(X_lr_dir / f"day_{day_id:06d}.npy", X_train[train_size + t].astype(np.float32))
        np.save(Y_hr_dir / f"day_{day_id:06d}.npy", Y_train[train_size + t].astype(np.float32))

    # Save test days
    for t in range(T_test):
        day_id = T_train + t
        np.save(X_lr_dir / f"day_{day_id:06d}.npy", X_test[t].astype(np.float32))
        np.save(Y_hr_dir / f"day_{day_id:06d}.npy", Y_test[t].astype(np.float32))
        if (t + 1) % 1000 == 0:
            print(f"   Saved {t+1}/{T_test} test days...")

    print(f"   Total days saved: {T_train + T_test}")

    # Create split dictionary
    print("\n[4/6] Creating metadata files...")
    split = {
        "train": list(range(train_size)),
        "val": list(range(train_size, T_train)),
        "test": list(range(T_train, T_train + T_test))
    }

    with open(metadata_dir / "split.json", 'w') as f:
        json.dump(split, f, indent=2)
    print(f"   Saved split.json")

    # Compute thresholds on training data (land pixels only)
    print("   Computing thresholds...")
    
    # OLD METHOD: P95/P99 of all pixels (too lenient)
    # land_values_train = Y_train[:train_size, 0, land_mask_2d].ravel()
    # p95 = float(np.percentile(land_values_train, 95))
    # p99 = float(np.percentile(land_values_train, 99))
    
    # NEW METHOD: P95/P99 of daily maximum values (more selective)
    daily_max_values = []
    for t in range(train_size):
        day_data = Y_train[t, 0]  # (H, W)
        land_values = day_data[land_mask_2d]
        if len(land_values) > 0:
            daily_max_values.append(land_values.max())
    
    daily_max_values = np.array(daily_max_values)
    p95 = float(np.percentile(daily_max_values, 95))
    p99 = float(np.percentile(daily_max_values, 99))
    p99_5 = float(np.percentile(daily_max_values, 99.5))
    
    print(f"   P95 of daily max: {p95:.4f}")
    print(f"   P99 of daily max: {p99:.4f}")
    print(f"   P99.5 of daily max: {p99_5:.4f}")
    
    thresholds = {
        "P95": p95,
        "P99": p99,
        "P99.5": p99_5
    }

    with open(metadata_dir / "thresholds.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"   Saved thresholds.json (P95={p95:.4f}, P99={p99:.4f})")

    # Copy normalization parameters and compute P99 in mm/day space
    print("\n   Copying normalization parameters and computing P99 in mm/day space...")

    # Check if normalization files exist in input directory
    mean_pr_src = input_dir / "ERA5_mean_train.npy"
    std_pr_src = input_dir / "ERA5_std_train.npy"

    if mean_pr_src.exists() and std_pr_src.exists():
        # Copy to metadata directory
        import shutil
        shutil.copy2(mean_pr_src, metadata_dir / "ERA5_mean_train.npy")
        shutil.copy2(std_pr_src, metadata_dir / "ERA5_std_train.npy")
        print(f"   ✓ Copied ERA5_mean_train.npy and ERA5_std_train.npy to metadata/")

        # Load for P99 computation
        mean_pr = np.load(mean_pr_src)  # (H, W)
        std_pr = np.load(std_pr_src)    # (H, W)

        # Compute P99 in mm/day space from training data
        print(f"   Computing P99 in mm/day space from {train_size} training days...")
        all_mm_values = []

        for t in range(train_size):
            # Get normalized data
            Y_norm = Y_train[t, 0]  # (H, W)

            # Denormalize to log1p space: logp = y * std + mean (pixel-wise!)
            logp = Y_norm * std_pr + mean_pr

            # Convert to mm/day: mm = exp(logp) - 1
            mm = np.expm1(logp)

            # Extract land pixels only
            mm_land = mm[land_mask_2d]
            all_mm_values.append(mm_land)

        # Concatenate and compute P99
        all_mm_values = np.concatenate(all_mm_values)
        p99_mmday = np.percentile(all_mm_values, 99)

        # Save P99 in mm/day
        np.save(metadata_dir / "p99_mmday.npy", p99_mmday)

        print(f"   ✓ Computed P99 (mm/day) = {p99_mmday:.2f}")
        print(f"     (From {len(all_mm_values):,} land pixel values)")
        print(f"     Range: [{all_mm_values.min():.2f}, {all_mm_values.max():.2f}] mm/day")
    else:
        print(f"   ⚠ WARNING: Normalization files not found in {input_dir}")
        print(f"     Expected: ERA5_mean_train.npy, ERA5_std_train.npy")
        print(f"     Skipping P99 computation in mm/day space")
        print(f"     Training will use estimated P99 (less accurate)")

    # Create H_W metadata
    h_w = {"H": int(H_hr), "W": int(W_hr)}
    with open(metadata_dir / "H_W.json", 'w') as f:
        json.dump(h_w, f, indent=2)
    print(f"   Saved H_W.json")

    # Categorize days
    print("   Categorizing days...")
    categories_train = categorize_days(Y_train[:train_size, 0], land_mask_2d, p95, p99)
    categories_val = categorize_days(Y_train[train_size:, 0], land_mask_2d, p95, p99)
    categories_test = categorize_days(Y_test[:, 0], land_mask_2d, p95, p99)

    # Combine and adjust indices
    categories = {}
    categories.update(categories_train)
    for k, v in categories_val.items():
        categories[str(int(k) + train_size)] = v
    for k, v in categories_test.items():
        categories[str(int(k) + T_train)] = v

    with open(metadata_dir / "categories.json", 'w') as f:
        json.dump(categories, f, indent=2)

    # Count categories
    cat_counts = {}
    for cat in categories.values():
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"   Saved categories.json")
    print(f"   Category distribution: {cat_counts}")

    # Create static maps
    print("\n[5/6] Creating static maps...")

    # Land/sea mask (1=land, 0=sea)
    land_sea_mask = land_mask_2d.astype(np.float32)[np.newaxis, :, :]  # (1, H, W)
    np.save(statics_dir / "land_sea_mask.npy", land_sea_mask)
    print(f"   Saved land_sea_mask.npy")

    # Distance to coast
    dist_to_coast = compute_distance_to_coast(land_mask_2d, high_res_lat, high_res_lon)
    dist_to_coast = dist_to_coast[np.newaxis, :, :]  # (1, H, W)
    np.save(statics_dir / "dist_to_coast_km.npy", dist_to_coast)
    print(f"   Saved dist_to_coast_km.npy")
    print(f"   Distance to coast range: [{dist_to_coast.min():.2f}, {dist_to_coast.max():.2f}] km")

    # Save coordinates for reference
    np.save(statics_dir / "high_res_lat.npy", high_res_lat)
    np.save(statics_dir / "high_res_lon.npy", high_res_lon)
    np.save(statics_dir / "low_res_lat.npy", low_res_lat)
    np.save(statics_dir / "low_res_lon.npy", low_res_lon)

    print("\n[6/6] Summary")
    print("=" * 80)
    print(f"Data prepared successfully!")
    print(f"Output directory: {output_dir}")
    print(f"\nDataset sizes:")
    print(f"  Train: {train_size} days")
    print(f"  Val:   {val_size} days")
    print(f"  Test:  {T_test} days")
    print(f"  Total: {T_train + T_test} days")
    print(f"\nCategory distribution:")
    for cat, count in sorted(cat_counts.items()):
        pct = 100.0 * count / (T_train + T_test)
        print(f"  {cat:20s}: {count:5d} ({pct:5.1f}%)")
    print(f"\nHeavy days (for augmentation):")
    heavy_count = cat_counts.get('heavy_coast', 0) + cat_counts.get('heavy_interior', 0)
    heavy_pct = 100.0 * heavy_count / (T_train + T_test)
    print(f"  Total heavy days: {heavy_count} ({heavy_pct:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for cVAE training")
    parser.add_argument("--input_dir", type=str,
                        default="/home/user/ugmentation_downscsling/mydata_corrected",
                        help="Directory containing preprocessed numpy files")
    parser.add_argument("--output_dir", type=str,
                        default="/home/user/ugmentation_downscsling/cvae_augmentation",
                        help="Output directory for cVAE data")

    args = parser.parse_args()
    main(args)
