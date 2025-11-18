"""
Parameter Analysis Script for cVAE Intensity Weighting

This script analyzes your precipitation data to recommend optimal parameters
for the intensity-weighted loss function:
- scale: Determines the reference point for weighting (where weight = 1.0)
- w_min, w_max: Weight bounds
- tail_boost: Extra multiplier for P99+ events

Run this after prepare_data.py to get data-driven parameter recommendations.
"""

import os
import argparse
import numpy as np
from pathlib import Path
import json


def load_data(data_root, srdrn_dir=None):
    """Load all necessary data for analysis.

    Args:
        data_root: Path to cVAE data directory (e.g., /scratch/.../cVAE_augmentation)
        srdrn_dir: Optional path to SRDRN output directory for normalization params
                   (e.g., /scratch/.../mydata_corrected)
    """
    data_root = Path(data_root)

    # Load split information
    split_path = data_root / "data" / "metadata" / "split.json"
    print(f"Loading split from: {split_path}")
    with open(split_path, 'r') as f:
        split = json.load(f)

    # Load normalization parameters
    # First try cVAE metadata folder, then fall back to SRDRN output folder
    metadata_dir = data_root / "data" / "metadata"
    mean_path = metadata_dir / "ERA5_mean_train.npy"
    std_path = metadata_dir / "ERA5_std_train.npy"

    if not mean_path.exists() or not std_path.exists():
        if srdrn_dir:
            srdrn_dir = Path(srdrn_dir)
            mean_path = srdrn_dir / "ERA5_mean_train.npy"
            std_path = srdrn_dir / "ERA5_std_train.npy"
            print(f"Loading normalization params from SRDRN dir: {srdrn_dir}")
        else:
            raise FileNotFoundError(
                f"Normalization files not found in {metadata_dir}. "
                f"Please specify --srdrn_dir to load from SRDRN output directory."
            )
    else:
        print(f"Loading normalization params from: {metadata_dir}")

    mean_pr = np.load(mean_path)
    std_pr = np.load(std_path)
    print(f"  Loaded mean_pr shape: {mean_pr.shape}, std_pr shape: {std_pr.shape}")

    # Load land mask
    statics_dir = data_root / "data" / "statics"
    land_mask_path = statics_dir / "land_sea_mask.npy"
    print(f"Loading land mask from: {land_mask_path}")
    land_mask = np.load(land_mask_path)[0]  # (H, W)

    # Load all training Y_hr data
    Y_hr_dir = data_root / "data" / "Y_hr"
    train_days = split['train']

    print(f"Loading {len(train_days)} training days from {Y_hr_dir}...")

    all_mm_values = []
    daily_max_mm = []
    daily_mean_mm = []

    for i, day_id in enumerate(train_days):
        Y_hr_path = Y_hr_dir / f"day_{day_id:06d}.npy"
        Y_hr_norm = np.load(Y_hr_path)[0]  # (H, W)

        # Denormalize to log1p space
        logp = Y_hr_norm * std_pr + mean_pr

        # Convert to mm/day
        mm = np.expm1(logp)

        # Collect land pixel values
        land_values = mm[land_mask > 0.5]
        all_mm_values.append(land_values)

        # Daily statistics
        daily_max_mm.append(land_values.max())
        daily_mean_mm.append(land_values.mean())

        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(train_days)} days...")

    all_mm_values = np.concatenate(all_mm_values)
    daily_max_mm = np.array(daily_max_mm)
    daily_mean_mm = np.array(daily_mean_mm)

    return {
        'all_mm': all_mm_values,
        'daily_max': daily_max_mm,
        'daily_mean': daily_mean_mm,
        'n_days': len(train_days),
        'n_land_pixels': int((land_mask > 0.5).sum()),
        'land_mask': land_mask
    }


def compute_percentiles(data):
    """Compute key percentiles for analysis."""
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    values = {}

    for p in percentiles:
        values[f'P{p}'] = float(np.percentile(data, p))

    return values


def analyze_weighting(all_mm, scale, w_min, w_max):
    """Analyze how weights are distributed with given parameters."""
    weights = np.clip(all_mm / scale, w_min, w_max)

    # Fraction of pixels at each weight level
    at_w_min = (weights == w_min).mean()
    at_w_max = (weights == w_max).mean()
    between = ((weights > w_min) & (weights < w_max)).mean()

    # Effective weight ratio
    weight_ratio = w_max / w_min

    # What fraction of total weight comes from different intensity levels
    total_weight = weights.sum()

    # Light rain: w = w_min
    light_weight = weights[weights == w_min].sum() / total_weight

    # Heavy rain: w >= 0.5 * w_max
    heavy_mask = weights >= 0.5 * w_max
    heavy_weight = weights[heavy_mask].sum() / total_weight

    # Extreme rain: w = w_max
    extreme_mask = weights == w_max
    extreme_weight = weights[extreme_mask].sum() / total_weight

    return {
        'pct_at_w_min': at_w_min * 100,
        'pct_at_w_max': at_w_max * 100,
        'pct_between': between * 100,
        'weight_ratio': weight_ratio,
        'pct_weight_from_light': light_weight * 100,
        'pct_weight_from_heavy': heavy_weight * 100,
        'pct_weight_from_extreme': extreme_weight * 100,
        'mean_weight': weights.mean(),
    }


def recommend_parameters(data_dict):
    """Generate parameter recommendations based on data analysis."""
    all_mm = data_dict['all_mm']

    # Compute percentiles
    percentiles = compute_percentiles(all_mm)

    # Current SRDRN scale
    srdrn_scale = 36.6

    # Recommendations
    recommendations = {}

    # 1. Scale recommendation
    # Scale should be set so that "moderate" rain gets weight ~1.0
    # P90-P95 is typically where moderate-heavy rain starts
    recommended_scale = percentiles['P90']
    recommendations['scale'] = {
        'srdrn_default': srdrn_scale,
        'recommended': recommended_scale,
        'based_on': 'P90 percentile',
        'rationale': 'P90 represents moderate-heavy rain; weight=1.0 at this level'
    }

    # 2. Tail boost recommendation
    # Analyze how many P99+ pixels exist and how much they should dominate
    p99_mm = percentiles['P99']
    n_p99_plus = (all_mm >= p99_mm).sum()
    pct_p99_plus = n_p99_plus / len(all_mm) * 100

    # If P99+ pixels are rare, need higher boost
    if pct_p99_plus < 0.5:
        recommended_tail_boost = 2.5
    elif pct_p99_plus < 1.0:
        recommended_tail_boost = 2.0
    else:
        recommended_tail_boost = 1.5

    recommendations['tail_boost'] = {
        'current': 1.5,
        'recommended': recommended_tail_boost,
        'pct_p99_plus_pixels': pct_p99_plus,
        'rationale': f'P99+ pixels are {pct_p99_plus:.3f}% of data; {"need higher boost to emphasize" if pct_p99_plus < 1.0 else "moderate boost sufficient"}'
    }

    # 3. Weight bounds
    # w_min: Should capture very light rain
    # w_max: Should limit extreme emphasis

    # Analyze weight distribution with recommended scale
    weight_analysis = analyze_weighting(all_mm, recommended_scale, 0.1, 2.0)

    # If too many pixels at w_min, scale may be too high
    if weight_analysis['pct_at_w_min'] > 90:
        recommendations['w_bounds_warning'] = 'Most pixels at w_min - consider lower scale or higher w_min'

    # If too few at w_max, may not emphasize extremes enough
    if weight_analysis['pct_at_w_max'] < 0.1:
        recommendations['w_bounds_warning'] = 'Very few pixels at w_max - extremes may not be emphasized enough'

    return percentiles, recommendations, weight_analysis


def main(args):
    print("=" * 80)
    print("Parameter Analysis for cVAE Intensity Weighting")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {args.data_root}")
    if args.srdrn_dir:
        print(f"Using SRDRN output dir for normalization: {args.srdrn_dir}")
    data_dict = load_data(args.data_root, args.srdrn_dir)

    print(f"\nData summary:")
    print(f"  Training days: {data_dict['n_days']}")
    print(f"  Land pixels per day: {data_dict['n_land_pixels']}")
    print(f"  Total land pixel-days: {len(data_dict['all_mm']):,}")

    # Basic statistics
    all_mm = data_dict['all_mm']
    print(f"\nPrecipitation statistics (mm/day):")
    print(f"  Min: {all_mm.min():.3f}")
    print(f"  Mean: {all_mm.mean():.3f}")
    print(f"  Max: {all_mm.max():.3f}")
    print(f"  Std: {all_mm.std():.3f}")

    # Percentile analysis
    percentiles, recommendations, weight_analysis = recommend_parameters(data_dict)

    print(f"\nPercentiles (mm/day):")
    for key, value in percentiles.items():
        print(f"  {key}: {value:.3f}")

    # Daily max analysis
    daily_max = data_dict['daily_max']
    print(f"\nDaily maximum precipitation:")
    print(f"  Mean daily max: {daily_max.mean():.3f} mm/day")
    print(f"  Max daily max: {daily_max.max():.3f} mm/day")
    print(f"  Days with max > P99: {(daily_max > percentiles['P99']).sum()}")

    # Parameter recommendations
    print("\n" + "=" * 80)
    print("PARAMETER RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n1. SCALE (reference for weight=1.0):")
    scale_rec = recommendations['scale']
    print(f"   SRDRN default: {scale_rec['srdrn_default']:.1f} mm/day")
    print(f"   Recommended:   {scale_rec['recommended']:.1f} mm/day")
    print(f"   Based on:      {scale_rec['based_on']}")
    print(f"   Rationale:     {scale_rec['rationale']}")

    print(f"\n2. TAIL_BOOST (extra multiplier for P99+):")
    boost_rec = recommendations['tail_boost']
    print(f"   Current:     {boost_rec['current']}")
    print(f"   Recommended: {boost_rec['recommended']}")
    print(f"   P99+ pixels: {boost_rec['pct_p99_plus_pixels']:.3f}%")
    print(f"   Rationale:   {boost_rec['rationale']}")

    # Weight distribution analysis
    print(f"\n3. WEIGHT DISTRIBUTION (with recommended scale={scale_rec['recommended']:.1f}):")
    print(f"   Pixels at w_min (0.1):  {weight_analysis['pct_at_w_min']:.1f}%")
    print(f"   Pixels at w_max (2.0):  {weight_analysis['pct_at_w_max']:.2f}%")
    print(f"   Pixels in between:      {weight_analysis['pct_between']:.1f}%")
    print(f"   Mean weight:            {weight_analysis['mean_weight']:.3f}")

    print(f"\n   Weight contribution to loss:")
    print(f"   - From light rain (w=w_min):   {weight_analysis['pct_weight_from_light']:.1f}%")
    print(f"   - From heavy rain (w>=w_max/2): {weight_analysis['pct_weight_from_heavy']:.1f}%")
    print(f"   - From extreme rain (w=w_max):  {weight_analysis['pct_weight_from_extreme']:.2f}%")

    if 'w_bounds_warning' in recommendations:
        print(f"\n   WARNING: {recommendations['w_bounds_warning']}")

    # Compare with different scales
    print("\n" + "=" * 80)
    print("SCALE COMPARISON")
    print("=" * 80)

    test_scales = [20.0, 30.0, 36.6, scale_rec['recommended'], 50.0, 75.0]
    test_scales = sorted(set(test_scales))

    print(f"\n{'Scale':<10} {'% at w_min':<12} {'% at w_max':<12} {'Mean Weight':<12} {'% Weight Heavy':<15}")
    print("-" * 65)

    for scale in test_scales:
        analysis = analyze_weighting(all_mm, scale, 0.1, 2.0)
        marker = " <-- SRDRN" if abs(scale - 36.6) < 0.1 else ""
        marker = " <-- Recommended" if abs(scale - scale_rec['recommended']) < 0.1 else marker
        print(f"{scale:<10.1f} {analysis['pct_at_w_min']:<12.1f} {analysis['pct_at_w_max']:<12.2f} "
              f"{analysis['mean_weight']:<12.3f} {analysis['pct_weight_from_heavy']:<15.1f}{marker}")

    # Effective weight multiplier for extremes
    print("\n" + "=" * 80)
    print("EFFECTIVE WEIGHT MULTIPLIER FOR EXTREMES")
    print("=" * 80)

    # With tail_boost, P99+ pixels get additional multiplier
    p99_mm = percentiles['P99']
    base_weight_p99 = min(p99_mm / scale_rec['recommended'], 2.0)

    for tail_boost in [1.0, 1.5, 2.0, 2.5, 3.0]:
        effective_multiplier = base_weight_p99 * tail_boost
        vs_light = effective_multiplier / 0.1  # vs w_min
        print(f"  tail_boost={tail_boost:.1f}: P99 effective weight = {effective_multiplier:.2f} "
              f"({vs_light:.0f}x vs light rain)")

    # Summary config recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIG UPDATE")
    print("=" * 80)
    print(f"""
# Update your config.yaml loss section:
loss:
  type: "simplified"
  scale: {scale_rec['recommended']:.1f}        # Based on your data's P90
  w_min: 0.1
  w_max: 2.0
  tail_boost: {boost_rec['recommended']:.1f}      # Based on P99+ frequency
  lambda_mass: 0.005
  beta_kl: 0.01
  warmup_epochs: 15
""")

    # Save results to file
    output_path = Path(args.data_root) / "data" / "metadata" / "parameter_analysis.json"
    results = {
        'percentiles': percentiles,
        'recommendations': {
            'scale': scale_rec['recommended'],
            'tail_boost': boost_rec['recommended'],
            'w_min': 0.1,
            'w_max': 2.0
        },
        'statistics': {
            'n_days': data_dict['n_days'],
            'n_land_pixels': data_dict['n_land_pixels'],
            'mean_mm': float(all_mm.mean()),
            'max_mm': float(all_mm.max()),
            'pct_p99_plus': boost_rec['pct_p99_plus_pixels']
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze data for cVAE parameter recommendations")
    parser.add_argument("--data_root", type=str,
                        default="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation",
                        help="Path to cVAE data directory (same as config data_root)")
    parser.add_argument("--srdrn_dir", type=str,
                        default="/scratch/user/u.hn319322/ondemand/Downscaling/mydata_corrected",
                        help="Path to SRDRN output directory for normalization params (if not in data_root)")

    args = parser.parse_args()
    main(args)
