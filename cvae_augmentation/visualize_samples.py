"""
Visualize and compare real vs synthesized precipitation samples.

Creates side-by-side map plots and computes statistics.
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors

from data_io import load_thresholds, CvaeDataset


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_normalization_params(config):
    """Load mean and std for denormalization."""
    norm_dir = Path(config['data_root']) / "data" / "metadata"

    mean_path = norm_dir / "ERA5_mean_train.npy"
    std_path = norm_dir / "ERA5_std_train.npy"

    if mean_path.exists() and std_path.exists():
        mean_pr = np.load(mean_path)
        std_pr = np.load(std_path)
        return mean_pr, std_pr
    else:
        raise FileNotFoundError(f"Normalization files not found in {norm_dir}")


def denormalize_to_mmday(y_normalized, mean_pr, std_pr):
    """Convert normalized precipitation to mm/day."""
    # Reshape for broadcasting if needed
    if mean_pr.ndim == 2 and y_normalized.ndim == 2:
        pass  # Already correct shape
    elif mean_pr.ndim == 2 and y_normalized.ndim == 3:
        mean_pr = mean_pr[np.newaxis, :, :]
        std_pr = std_pr[np.newaxis, :, :]

    log1p_precip = y_normalized * std_pr + mean_pr
    y_mmday = np.expm1(log1p_precip)
    y_mmday = np.maximum(y_mmday, 0.0)

    return y_mmday


def load_real_data(config, day_ids):
    """Load real precipitation data for given day IDs using CvaeDataset."""
    # Create dataset to load data
    dataset = CvaeDataset(
        data_root=config['data_root'],
        split='train',
        use_land_sea=True,
        use_dist_coast=True,
        use_elevation=False
    )

    # Build mapping from day_id to dataset index
    day_id_to_idx = {day_id: idx for idx, day_id in enumerate(dataset.day_ids)}

    real_data = []
    for day_id in day_ids:
        if day_id in day_id_to_idx:
            idx = day_id_to_idx[day_id]
            sample = dataset[idx]
            # Y_hr is (1, H, W), convert to numpy
            data = sample['Y_hr'].numpy()
            real_data.append(data)
        else:
            print(f"Warning: Real data not found for day {day_id} (not in train split)")
            real_data.append(None)

    return real_data


def load_synth_data(synth_dir, day_ids):
    """Load synthesized precipitation data for given day IDs."""
    synth_data = []

    for day_id in day_ids:
        # Find all samples for this day
        # Filename format: day_XXXXX_sample_YY.Y_hr_syn.npy
        day_samples = []
        for k in range(10):  # Check up to 10 samples per day
            file_path = synth_dir / f"day_{day_id:06d}_sample_{k:02d}.Y_hr_syn.npy"
            if file_path.exists():
                data = np.load(file_path)
                day_samples.append(data)

        if len(day_samples) > 0:
            synth_data.append(day_samples)
        else:
            synth_data.append(None)

    return synth_data


def compute_statistics(data_mmday, land_mask=None):
    """Compute precipitation statistics."""
    if land_mask is not None:
        data_flat = data_mmday[land_mask > 0]
    else:
        data_flat = data_mmday.flatten()

    # Filter non-zero values for percentile calculation
    nonzero = data_flat[data_flat > 0]

    stats = {
        'mean': float(np.mean(data_flat)),
        'std': float(np.std(data_flat)),
        'max': float(np.max(data_flat)),
        'min': float(np.min(data_flat)),
        'total_mass': float(np.sum(data_flat)),
        'wet_fraction': float(np.sum(data_flat > 0.1) / len(data_flat)),
    }

    if len(nonzero) > 0:
        stats['P50'] = float(np.percentile(nonzero, 50))
        stats['P90'] = float(np.percentile(nonzero, 90))
        stats['P95'] = float(np.percentile(nonzero, 95))
        stats['P99'] = float(np.percentile(nonzero, 99))
    else:
        stats['P50'] = stats['P90'] = stats['P95'] = stats['P99'] = 0.0

    return stats


def create_precip_colormap():
    """Create a precipitation colormap similar to meteorological standards."""
    colors = [
        '#FFFFFF',  # 0: white (no rain)
        '#C6DBEF',  # light blue
        '#9ECAE1',  #
        '#6BAED6',  #
        '#4292C6',  #
        '#2171B5',  # medium blue
        '#084594',  # dark blue
        '#FFFF00',  # yellow
        '#FFA500',  # orange
        '#FF0000',  # red
        '#8B0000',  # dark red
    ]

    bounds = [0, 0.1, 1, 2, 5, 10, 20, 30, 50, 75, 100, 200]
    cmap = mcolors.ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


def plot_comparison(real_mmday, synth_mmday_list, day_id, output_dir, land_mask=None):
    """Create side-by-side comparison plots."""

    cmap, norm, bounds = create_precip_colormap()

    n_synth = len(synth_mmday_list)
    n_cols = 1 + n_synth

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    # Plot real data
    ax = axes[0]
    im = ax.imshow(real_mmday, cmap=cmap, norm=norm, origin='lower')
    ax.set_title(f'Real Day {day_id}')
    ax.axis('off')

    # Compute and display stats
    real_stats = compute_statistics(real_mmday, land_mask)
    stats_text = f"Max: {real_stats['max']:.1f}\nP95: {real_stats['P95']:.1f}\nP99: {real_stats['P99']:.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot synthesized data
    for i, synth_mmday in enumerate(synth_mmday_list):
        ax = axes[i + 1]
        im = ax.imshow(synth_mmday, cmap=cmap, norm=norm, origin='lower')
        ax.set_title(f'Synth K={i}')
        ax.axis('off')

        # Compute and display stats
        synth_stats = compute_statistics(synth_mmday, land_mask)
        stats_text = f"Max: {synth_stats['max']:.1f}\nP95: {synth_stats['P95']:.1f}\nP99: {synth_stats['P99']:.1f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, extend='max')
    cbar.set_label('Precipitation (mm/day)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save figure
    output_path = output_dir / f"comparison_day_{day_id:05d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main(args):
    print("=" * 80)
    print("Visualize Real vs Synthesized Precipitation Samples")
    print("=" * 80)

    # Load config
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")

    # Set paths
    synth_dir = Path(args.synth_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synth directory: {synth_dir}")
    print(f"Output directory: {output_dir}")

    # Load normalization parameters
    print("\nLoading normalization parameters...")
    mean_pr, std_pr = load_normalization_params(config)

    # Load land mask
    land_mask_path = Path(config['data_root']) / "data" / "static" / "land_sea_mask.npy"
    if land_mask_path.exists():
        land_mask = np.load(land_mask_path)
        print(f"Loaded land mask: {land_mask.shape}")
    else:
        land_mask = None
        print("Land mask not found, using all pixels")

    # Load thresholds
    thresholds_path = Path(config['data_root']) / "data" / "metadata" / "thresholds.json"
    thresholds = load_thresholds(thresholds_path)

    # Find available synthesized samples
    # Filename format: day_XXXXX_sample_YY.Y_hr_syn.npy
    synth_files = list(synth_dir.glob("day_*_sample_*.Y_hr_syn.npy"))
    if len(synth_files) == 0:
        raise FileNotFoundError(f"No synthesized samples found in {synth_dir}")

    # Extract unique day IDs
    day_ids = set()
    for f in synth_files:
        # Parse filename: day_000228_sample_00.Y_hr_syn.npy
        parts = f.stem.split('_')
        day_id = int(parts[1])
        day_ids.add(day_id)

    day_ids = sorted(list(day_ids))
    print(f"\nFound {len(day_ids)} days with synthesized samples")

    # Limit to requested number
    if args.num_samples > 0:
        day_ids = day_ids[:args.num_samples]
    print(f"Processing {len(day_ids)} samples")

    # Load real and synth data
    print("\nLoading data...")
    real_data = load_real_data(config, day_ids)
    synth_data = load_synth_data(synth_dir, day_ids)

    # Aggregate statistics
    all_real_stats = []
    all_synth_stats = []

    # Process each day
    print("\nGenerating visualizations...")
    for i, day_id in enumerate(day_ids):
        if real_data[i] is None or synth_data[i] is None:
            print(f"  Skipping day {day_id} (missing data)")
            continue

        # Denormalize to mm/day
        real_mmday = denormalize_to_mmday(real_data[i], mean_pr, std_pr)
        synth_mmday_list = [denormalize_to_mmday(s, mean_pr, std_pr) for s in synth_data[i]]

        # Squeeze if needed
        if real_mmday.ndim == 3:
            real_mmday = real_mmday.squeeze()
        synth_mmday_list = [s.squeeze() if s.ndim == 3 else s for s in synth_mmday_list]

        # Plot comparison
        output_path = plot_comparison(real_mmday, synth_mmday_list, day_id, output_dir, land_mask)

        # Collect statistics
        real_stats = compute_statistics(real_mmday, land_mask)
        all_real_stats.append(real_stats)

        for synth_mmday in synth_mmday_list:
            synth_stats = compute_statistics(synth_mmday, land_mask)
            all_synth_stats.append(synth_stats)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(day_ids)} days")

    print(f"\nGenerated {len(day_ids)} comparison plots in {output_dir}")

    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    def aggregate_stats(stats_list):
        """Aggregate statistics across all samples."""
        keys = ['mean', 'max', 'P95', 'P99', 'total_mass', 'wet_fraction']
        agg = {}
        for key in keys:
            values = [s[key] for s in stats_list]
            agg[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return agg

    real_agg = aggregate_stats(all_real_stats)
    synth_agg = aggregate_stats(all_synth_stats)

    print("\n{:<15} {:>12} {:>12} {:>12}".format("Metric", "Real", "Synth", "Diff %"))
    print("-" * 55)

    for key in ['mean', 'max', 'P95', 'P99']:
        real_val = real_agg[key]['mean']
        synth_val = synth_agg[key]['mean']
        if real_val > 0:
            diff_pct = 100 * (synth_val - real_val) / real_val
        else:
            diff_pct = 0
        print(f"{key:<15} {real_val:>12.2f} {synth_val:>12.2f} {diff_pct:>+10.1f}%")

    # Wet fraction
    real_wet = real_agg['wet_fraction']['mean']
    synth_wet = synth_agg['wet_fraction']['mean']
    if real_wet > 0:
        wet_diff = 100 * (synth_wet - real_wet) / real_wet
    else:
        wet_diff = 0
    print(f"{'wet_fraction':<15} {real_wet:>12.3f} {synth_wet:>12.3f} {wet_diff:>+10.1f}%")

    # Total mass
    real_mass = real_agg['total_mass']['mean']
    synth_mass = synth_agg['total_mass']['mean']
    if real_mass > 0:
        mass_diff = 100 * (synth_mass - real_mass) / real_mass
    else:
        mass_diff = 0
    print(f"{'total_mass':<15} {real_mass:>12.1f} {synth_mass:>12.1f} {mass_diff:>+10.1f}%")

    print("=" * 55)

    # Save statistics to JSON
    stats_output = {
        'num_days': len(day_ids),
        'num_synth_samples': len(all_synth_stats),
        'real': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in real_agg.items()},
        'synth': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in synth_agg.items()}
    }

    stats_path = output_dir / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")

    # Assessment
    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)

    issues = []

    # Check P95
    p95_diff = abs(synth_agg['P95']['mean'] - real_agg['P95']['mean']) / real_agg['P95']['mean']
    if p95_diff > 0.2:
        issues.append(f"P95 differs by {p95_diff*100:.1f}% (threshold: 20%)")

    # Check P99
    p99_diff = abs(synth_agg['P99']['mean'] - real_agg['P99']['mean']) / real_agg['P99']['mean']
    if p99_diff > 0.3:
        issues.append(f"P99 differs by {p99_diff*100:.1f}% (threshold: 30%)")

    # Check mass
    mass_diff_pct = abs(mass_diff)
    if mass_diff_pct > 20:
        issues.append(f"Total mass differs by {mass_diff_pct:.1f}% (threshold: 20%)")

    # Check wet fraction
    wet_diff_pct = abs(wet_diff)
    if wet_diff_pct > 30:
        issues.append(f"Wet fraction differs by {wet_diff_pct:.1f}% (threshold: 30%)")

    if len(issues) == 0:
        print("Status: GOOD - Synthesized samples match real data statistics")
    else:
        print("Status: ISSUES FOUND")
        for issue in issues:
            print(f"  - {issue}")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize real vs synthesized precipitation")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--synth_dir", type=str,
                        default="outputs/cvae_simplified/synth",
                        help="Directory containing synthesized samples")
    parser.add_argument("--output", type=str,
                        default="outputs/cvae_simplified/visualizations",
                        help="Output directory for plots")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to visualize (0 for all)")

    args = parser.parse_args()
    main(args)
