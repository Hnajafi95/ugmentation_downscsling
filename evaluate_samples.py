"""
Evaluate quality of generated precipitation samples from cVAE.

Metrics:
1. Spatial pattern correlation
2. Extreme value distribution matching
3. Mass conservation
4. Sample diversity
5. Physical realism checks
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import ks_2samp, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC
import matplotlib.pyplot as plt

from model_cvae import CVAE
from data_io import CvaeDataset, load_thresholds


def load_model(checkpoint_path, config):
    """Load trained cVAE model."""
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
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def generate_samples(model, dataset, num_days=50, samples_per_day=5, device='cuda'):
    """
    Generate samples from heavy precipitation days.

    Returns:
        real_samples: (N, H, W) real heavy precipitation days
        generated_samples: (N, K, H, W) generated samples (K per real day)
        X_lr_samples: (N, C, H_lr, W_lr) corresponding low-res inputs
    """
    model = model.to(device)

    # Get heavy day indices
    categories_path = Path(dataset.data_root) / "metadata" / "categories.json"
    with open(categories_path, 'r') as f:
        categories = json.load(f)

    # Categories format is {day_id: category_name}
    heavy_indices = []
    for day_id_str, category in categories.items():
        if category in ["heavy_coast", "heavy_interior"]:
            heavy_indices.append(int(day_id_str))

    heavy_indices.sort()
    print(f"Found {len(heavy_indices)} heavy precipitation days")

    # Convert day_ids to dataset indices
    # The dataset contains a subset of days (e.g., test split), so we need to find
    # which heavy days are actually in this dataset
    day_id_to_idx = {day_id: idx for idx, day_id in enumerate(dataset.day_ids)}

    valid_indices = []
    for day_id in heavy_indices:
        if day_id in day_id_to_idx:
            valid_indices.append(day_id_to_idx[day_id])

    print(f"Found {len(valid_indices)} heavy days in current dataset split")

    if len(valid_indices) == 0:
        raise ValueError("No heavy precipitation days found in the current dataset split. "
                        "Try using train split instead of test split.")

    # Limit to num_days
    valid_indices = valid_indices[:num_days]

    real_samples = []
    generated_samples = []
    X_lr_samples = []

    with torch.no_grad():
        for idx in valid_indices:
            # Get real data
            sample = dataset[idx]
            X_lr = sample['X_lr'].unsqueeze(0).to(device)  # (1, C, H_lr, W_lr)
            Y_hr = sample['Y_hr'].unsqueeze(0).to(device)  # (1, 1, H, W)
            S = sample['S'].unsqueeze(0).to(device)        # (1, S_ch, H, W)

            # Store real sample
            real_samples.append(Y_hr.squeeze().cpu().numpy())
            X_lr_samples.append(X_lr.squeeze().cpu().numpy())

            # Generate K samples
            batch_generated = []
            for k in range(samples_per_day):
                # Sample from prior
                Y_gen = model.sample(X_lr, S, use_prior=True)
                batch_generated.append(Y_gen.squeeze().cpu().numpy())

            generated_samples.append(np.stack(batch_generated))

    real_samples = np.stack(real_samples)           # (N, H, W)
    generated_samples = np.stack(generated_samples)  # (N, K, H, W)
    X_lr_samples = np.stack(X_lr_samples)           # (N, C, H_lr, W_lr)

    return real_samples, generated_samples, X_lr_samples


def compute_spatial_correlation(real, generated):
    """
    Compute spatial pattern correlation between real and generated samples.

    Args:
        real: (N, H, W)
        generated: (N, K, H, W)

    Returns:
        correlations: (N, K) correlation for each sample pair
    """
    N, K, H, W = generated.shape
    correlations = np.zeros((N, K))

    for i in range(N):
        real_flat = real[i].flatten()
        for k in range(K):
            gen_flat = generated[i, k].flatten()
            # Pearson correlation
            corr, _ = pearsonr(real_flat, gen_flat)
            correlations[i, k] = corr

    return correlations


def compute_extreme_value_distribution(real, generated, land_mask, p95):
    """
    Check if generated samples match the extreme value distribution.

    Uses Kolmogorov-Smirnov test on tail values.

    Returns:
        ks_statistic: KS test statistic (lower is better)
        p_value: p-value (>0.05 means distributions are similar)
    """
    # Extract extreme values
    real_extremes = real[real >= p95]

    # Flatten all generated samples
    N, K, H, W = generated.shape
    gen_flat = generated.reshape(-1)
    gen_extremes = gen_flat[gen_flat >= p95]

    # KS test
    ks_stat, p_value = ks_2samp(real_extremes, gen_extremes)

    return ks_stat, p_value


def compute_mass_conservation(real, generated, land_mask):
    """
    Check if total precipitation mass is conserved.

    Returns:
        relative_bias: Mean relative difference in total mass
    """
    N, K, H, W = generated.shape

    biases = []
    for i in range(N):
        real_mass = np.sum(real[i] * land_mask)
        for k in range(K):
            gen_mass = np.sum(generated[i, k] * land_mask)
            rel_bias = (gen_mass - real_mass) / (real_mass + 1e-8)
            biases.append(rel_bias)

    return np.mean(biases), np.std(biases)


def compute_diversity_score(generated):
    """
    Measure diversity between generated samples for the same conditioning.

    Higher diversity = better (samples are different from each other)

    Returns:
        diversity: Mean pairwise cosine distance
    """
    N, K, H, W = generated.shape

    diversities = []
    for i in range(N):
        # Flatten K samples
        samples_flat = generated[i].reshape(K, -1)  # (K, H*W)

        # Compute pairwise cosine similarity
        sim_matrix = cosine_similarity(samples_flat)

        # Get upper triangle (excluding diagonal)
        upper_tri = sim_matrix[np.triu_indices(K, k=1)]

        # Diversity = 1 - similarity
        diversity = 1 - np.mean(upper_tri)
        diversities.append(diversity)

    return np.mean(diversities), np.std(diversities)


def evaluate_model(model, dataset, device='cuda', num_days=50, samples_per_day=5):
    """
    Comprehensive evaluation of cVAE model.

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("Generating samples for evaluation...")
    real, generated, X_lr = generate_samples(
        model, dataset, num_days, samples_per_day, device
    )

    # Load land mask and thresholds
    land_mask_path = Path(dataset.data_root) / "statics" / "land_sea_mask.npy"
    land_mask = np.load(land_mask_path)

    thresholds_path = Path(dataset.data_root) / "metadata" / "thresholds.json"
    thresholds = load_thresholds(thresholds_path)
    p95 = thresholds['P95']

    print("\nComputing metrics...")

    # 1. Spatial correlation
    print("  - Spatial correlation")
    corrs = compute_spatial_correlation(real, generated)

    # 2. Extreme value distribution
    print("  - Extreme value distribution")
    ks_stat, p_value = compute_extreme_value_distribution(real, generated, land_mask, p95)

    # 3. Mass conservation
    print("  - Mass conservation")
    mass_bias_mean, mass_bias_std = compute_mass_conservation(real, generated, land_mask)

    # 4. Diversity
    print("  - Sample diversity")
    diversity_mean, diversity_std = compute_diversity_score(generated)

    # 5. Reconstruction quality (MAE on extremes)
    print("  - Reconstruction quality")
    gen_mean = generated.mean(axis=1)  # Average over K samples
    mae_extreme = np.mean(np.abs(real - gen_mean))

    metrics = {
        'spatial_correlation': {
            'mean': float(np.mean(corrs)),
            'std': float(np.std(corrs)),
            'min': float(np.min(corrs)),
            'max': float(np.max(corrs))
        },
        'extreme_distribution': {
            'ks_statistic': float(ks_stat),
            'p_value': float(p_value),
            'match': bool(p_value > 0.05)  # True if distributions match
        },
        'mass_conservation': {
            'bias_mean': float(mass_bias_mean),
            'bias_std': float(mass_bias_std),
            'abs_bias_mean': float(np.abs(mass_bias_mean))
        },
        'diversity': {
            'mean': float(diversity_mean),
            'std': float(diversity_std)
        },
        'reconstruction': {
            'mae_extreme': float(mae_extreme)
        }
    }

    return metrics, real, generated


def print_evaluation_summary(metrics):
    """Print human-readable evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print("\n1. SPATIAL PATTERN CORRELATION")
    print(f"   Mean correlation: {metrics['spatial_correlation']['mean']:.4f}")
    print(f"   Std dev:          {metrics['spatial_correlation']['std']:.4f}")
    print(f"   Range:            [{metrics['spatial_correlation']['min']:.4f}, {metrics['spatial_correlation']['max']:.4f}]")
    print(f"   Status: {'✓ GOOD' if metrics['spatial_correlation']['mean'] > 0.6 else '✗ POOR'}")

    print("\n2. EXTREME VALUE DISTRIBUTION")
    print(f"   KS statistic:     {metrics['extreme_distribution']['ks_statistic']:.4f}")
    print(f"   p-value:          {metrics['extreme_distribution']['p_value']:.4f}")
    print(f"   Distributions match: {'✓ YES' if metrics['extreme_distribution']['match'] else '✗ NO'}")

    print("\n3. MASS CONSERVATION")
    print(f"   Mean bias:        {metrics['mass_conservation']['bias_mean']:.4f}")
    print(f"   Abs bias:         {metrics['mass_conservation']['abs_bias_mean']:.4f}")
    print(f"   Status: {'✓ GOOD' if metrics['mass_conservation']['abs_bias_mean'] < 0.1 else '✗ POOR'}")

    print("\n4. SAMPLE DIVERSITY")
    print(f"   Mean diversity:   {metrics['diversity']['mean']:.4f}")
    print(f"   Std dev:          {metrics['diversity']['std']:.4f}")
    print(f"   Status: {'✓ GOOD' if metrics['diversity']['mean'] > 0.3 else '✗ POOR'}")

    print("\n5. RECONSTRUCTION QUALITY")
    print(f"   MAE (extremes):   {metrics['reconstruction']['mae_extreme']:.4f}")

    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)

    score = 0
    if metrics['spatial_correlation']['mean'] > 0.6:
        score += 1
    if metrics['extreme_distribution']['match']:
        score += 1
    if metrics['mass_conservation']['abs_bias_mean'] < 0.1:
        score += 1
    if metrics['diversity']['mean'] > 0.3:
        score += 1

    print(f"Quality score: {score}/4")
    if score >= 3:
        print("Status: ✓ GOOD - Model generates realistic samples")
    elif score >= 2:
        print("Status: ⚠ MODERATE - Model needs improvement")
    else:
        print("Status: ✗ POOR - Model generates unrealistic samples")

    print("=" * 80)


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset - use train split since most heavy days are there
    dataset = CvaeDataset(
        data_root=config['data_root'],
        split='train',  # Use train set where heavy days are
        use_land_sea=config['statics']['use_land_sea'],
        use_dist_coast=config['statics']['use_dist_coast'],
        use_elevation=config['statics']['use_elevation'],
        normalize_statics=config['statics']['normalize_statics']
    )

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, config)

    # Evaluate
    metrics, real, generated = evaluate_model(
        model, dataset, device,
        num_days=args.num_days,
        samples_per_day=args.samples_per_day
    )

    # Print summary
    print_evaluation_summary(metrics)

    # Save metrics
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cVAE generated samples")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Path to save evaluation metrics")
    parser.add_argument("--num_days", type=int, default=50,
                        help="Number of heavy days to evaluate")
    parser.add_argument("--samples_per_day", type=int, default=5,
                        help="Number of samples to generate per day")

    args = parser.parse_args()
    main(args)
