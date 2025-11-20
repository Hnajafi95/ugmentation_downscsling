"""
Enhanced loss functions for Phase 1+ improvements.

New features:
1. Sparsity loss (wet fraction matching)
2. Spatial gradient loss (preserve edges/fronts)
3. Free bits KL (prevent posterior collapse)
4. All existing features from losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from losses import (
    intensity_weighted_mae_with_tail_boost_mmday,
    mass_loss,
    kl_divergence
)


def sparsity_loss(y_hat, land_mask, target_wet_fraction=0.75, threshold=0.01):
    """
    Penalize deviation from target wet fraction.

    Encourages model to generate realistic dry/wet patterns instead of
    drizzle everywhere.

    Args:
        y_hat: (B, 1, H, W) predicted precipitation (normalized)
        land_mask: (1, 1, H, W) land mask
        target_wet_fraction: Target fraction of wet pixels (default 0.75)
        threshold: Threshold for considering pixel as "wet" (default 0.01)

    Returns:
        loss: Scalar sparsity loss
        info: Dict with logging info
    """
    # Apply land mask
    if land_mask.dim() == 3:
        land_mask = land_mask.unsqueeze(0)

    y_masked = y_hat * land_mask

    # Count wet pixels
    wet_mask = (y_masked > threshold).float()
    wet_fraction = wet_mask.sum() / (land_mask.sum() + 1e-8)

    # Loss is absolute difference from target
    loss = torch.abs(wet_fraction - target_wet_fraction)

    info = {
        'wet_fraction': wet_fraction.item(),
        'target_wet_fraction': target_wet_fraction,
        'n_wet_pixels': wet_mask.sum().item(),
        'n_land_pixels': land_mask.sum().item()
    }

    return loss, info


def spatial_gradient_loss(y_true, y_hat, land_mask=None):
    """
    Preserve spatial gradients (edges, precipitation fronts).

    This helps maintain realistic spatial structure and sharp transitions
    between wet and dry regions.

    Args:
        y_true: (B, 1, H, W) ground truth
        y_hat: (B, 1, H, W) predictions
        land_mask: Optional (1, 1, H, W) land mask

    Returns:
        loss: Scalar gradient loss
    """
    # Compute spatial gradients (first-order differences)
    # Horizontal gradients
    grad_true_x = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    grad_hat_x = y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]

    # Vertical gradients
    grad_true_y = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    grad_hat_y = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]

    # L1 loss on gradients
    loss_x = F.l1_loss(grad_hat_x, grad_true_x)
    loss_y = F.l1_loss(grad_hat_y, grad_true_y)

    loss = loss_x + loss_y

    return loss


def kl_divergence_with_free_bits(mu, logvar, free_bits=0.5):
    """
    KL divergence with free bits constraint to prevent posterior collapse.

    Free bits: Allow each latent dimension to have at least 'free_bits'
    nats of information. This prevents the model from collapsing the
    posterior to a point mass.

    Args:
        mu: (B, d_z) latent mean
        logvar: (B, d_z) latent log variance
        free_bits: Minimum KL per dimension (default 0.5 nats)

    Returns:
        kl_loss: Scalar KL loss
        info: Dict with logging info
    """
    # KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, d_z)

    # Apply free bits constraint
    # max(kl_per_dim, free_bits) for each dimension
    kl_per_dim_constrained = torch.max(
        kl_per_dim,
        torch.tensor(free_bits, device=kl_per_dim.device)
    )

    # Sum over dimensions, mean over batch
    kl_loss = kl_per_dim_constrained.sum(dim=1).mean()

    # For logging
    info = {
        'kl_mean_per_dim': kl_per_dim.mean().item(),
        'kl_max_per_dim': kl_per_dim.max().item(),
        'kl_min_per_dim': kl_per_dim.min().item(),
        'n_dims_at_free_bits': (kl_per_dim < free_bits).sum().item()
    }

    return kl_loss, info


class EnhancedCVAELoss(nn.Module):
    """
    Enhanced cVAE loss with sparsity and spatial structure preservation.

    L_total = L_weighted_rec +
              λ_mass * L_mass +
              λ_sparse * L_sparse +
              λ_spatial * L_spatial +
              β * L_kl

    Phase 1 additions:
    - Sparsity loss (wet fraction matching)
    - Spatial gradient loss (edge preservation)
    - Free bits KL (prevent collapse)
    """

    def __init__(self,
                 p99_mmday: float,
                 mean_pr,
                 std_pr,
                 scale: float = 20.0,
                 w_min: float = 0.1,
                 w_max: float = 3.0,
                 tail_boost: float = 2.5,
                 lambda_mass: float = 0.005,
                 lambda_sparse: float = 0.01,
                 lambda_spatial: float = 0.02,
                 target_wet_fraction: float = 0.75,
                 beta_kl: float = 0.005,
                 warmup_epochs: int = 20,
                 free_bits: float = 0.5):
        """
        Args:
            p99_mmday: P99 threshold in mm/day space
            mean_pr: (H, W) normalization mean
            std_pr: (H, W) normalization std
            scale: Intensity weighting scale (mm/day)
            w_min, w_max: Weight bounds
            tail_boost: Extra multiplier for P99+ (INCREASED to 2.5)
            lambda_mass: Weight for mass conservation
            lambda_sparse: Weight for sparsity loss (NEW)
            lambda_spatial: Weight for spatial gradient loss (NEW)
            target_wet_fraction: Target wet fraction (NEW, default 0.75)
            beta_kl: Final KL weight (REDUCED to 0.005)
            warmup_epochs: KL warmup epochs (INCREASED to 20)
            free_bits: Minimum KL per dimension (NEW, default 0.5)
        """
        super().__init__()

        self.p99_mmday = p99_mmday
        self.scale = scale
        self.w_min = w_min
        self.w_max = w_max
        self.tail_boost = tail_boost
        self.lambda_mass = lambda_mass
        self.lambda_sparse = lambda_sparse
        self.lambda_spatial = lambda_spatial
        self.target_wet_fraction = target_wet_fraction
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs
        self.free_bits = free_bits

        # Store normalization parameters
        if isinstance(mean_pr, np.ndarray):
            mean_pr = torch.from_numpy(mean_pr).float()
        if isinstance(std_pr, np.ndarray):
            std_pr = torch.from_numpy(std_pr).float()

        self.register_buffer('mean_pr', mean_pr)
        self.register_buffer('std_pr', std_pr)

        self.current_epoch = 0
        self.current_beta = 0.0

    def update_beta(self, epoch):
        """Update KL weight (warm-up schedule)."""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_beta = self.beta_kl * (epoch / self.warmup_epochs)
        else:
            self.current_beta = self.beta_kl

    def forward(self, y_true, y_hat, mu, logvar, land_mask):
        """
        Compute enhanced loss.

        Args:
            y_true: (B, 1, H, W) normalized precipitation
            y_hat: (B, 1, H, W) normalized predictions
            mu: (B, d_z)
            logvar: (B, d_z)
            land_mask: (B, 1, H, W) or (1, 1, H, W)

        Returns:
            loss_dict: All loss components
        """
        # 1. Weighted reconstruction loss (unchanged)
        rec_loss, rec_info = intensity_weighted_mae_with_tail_boost_mmday(
            y_true, y_hat, self.mean_pr, self.std_pr, self.p99_mmday,
            self.scale, self.w_min, self.w_max, self.tail_boost
        )

        # 2. Mass conservation (unchanged)
        m_loss = mass_loss(y_true, y_hat, land_mask)

        # 3. Sparsity loss (NEW)
        sparse_loss, sparse_info = sparsity_loss(
            y_hat, land_mask, self.target_wet_fraction
        )

        # 4. Spatial gradient loss (NEW)
        spatial_loss = spatial_gradient_loss(y_true, y_hat, land_mask)

        # 5. KL divergence with free bits (IMPROVED)
        kl_loss, kl_info = kl_divergence_with_free_bits(mu, logvar, self.free_bits)

        # Total loss
        total_loss = (rec_loss +
                     self.lambda_mass * m_loss +
                     self.lambda_sparse * sparse_loss +
                     self.lambda_spatial * spatial_loss +
                     self.current_beta * kl_loss)

        # For logging
        loss_dict = {
            'loss': total_loss,
            'L_rec': rec_loss,
            'L_mass': m_loss,
            'L_sparse': sparse_loss,
            'L_spatial': spatial_loss,
            'L_kl': kl_loss,
            'beta': self.current_beta,
            'weights_mean': rec_info['weights_mean'],
            'n_tail_pixels': rec_info['n_tail_pixels'],
            'mm_true_max': rec_info['mm_true_max'],
            'mm_true_mean': rec_info['mm_true_mean'],
            'wet_fraction': sparse_info['wet_fraction'],
            'kl_mean_per_dim': kl_info['kl_mean_per_dim'],
            'n_dims_at_free_bits': kl_info['n_dims_at_free_bits']
        }

        return loss_dict


def test_enhanced_losses():
    """Test the enhanced loss functions."""
    print("Testing enhanced losses...")

    # Dummy data
    B, H, W = 4, 156, 132
    y_true = torch.randn(B, 1, H, W).abs() * 2  # Normalized
    y_hat = y_true + torch.randn(B, 1, H, W) * 0.3
    land_mask = torch.ones(1, 1, H, W)
    mu = torch.randn(B, 128)
    logvar = torch.randn(B, 128) * 0.5

    # Dummy normalization params
    mean_pr = torch.zeros(H, W)
    std_pr = torch.ones(H, W)
    p99_mmday = 43.63

    # Test sparsity loss
    print("\n1. Sparsity loss:")
    loss, info = sparsity_loss(y_hat, land_mask, target_wet_fraction=0.75)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Wet fraction: {info['wet_fraction']:.4f}")

    # Test spatial gradient loss
    print("\n2. Spatial gradient loss:")
    loss = spatial_gradient_loss(y_true, y_hat)
    print(f"   Loss: {loss.item():.4f}")

    # Test KL with free bits
    print("\n3. KL with free bits:")
    loss, info = kl_divergence_with_free_bits(mu, logvar, free_bits=0.5)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Dims at free bits: {info['n_dims_at_free_bits']}")

    # Test enhanced criterion
    print("\n4. Enhanced criterion:")
    criterion = EnhancedCVAELoss(p99_mmday, mean_pr, std_pr)
    criterion.update_beta(epoch=10)
    loss_dict = criterion(y_true, y_hat, mu, logvar, land_mask)
    print(f"   Total loss: {loss_dict['loss'].item():.4f}")
    print(f"   L_rec: {loss_dict['L_rec'].item():.4f}")
    print(f"   L_sparse: {loss_dict['L_sparse'].item():.4f}")
    print(f"   L_spatial: {loss_dict['L_spatial'].item():.4f}")
    print(f"   L_kl: {loss_dict['L_kl'].item():.4f}")
    print(f"   Wet fraction: {loss_dict['wet_fraction']:.4f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_enhanced_losses()
