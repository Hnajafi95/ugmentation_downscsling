"""
Simplified loss functions for cVAE training.

Philosophy: ONE unified reconstruction objective (like SRDRN)
Instead of separate MAE_all + MAE_extreme, use intensity-weighted MAE
that naturally emphasizes heavier precipitation.

UPDATED: Now works in mm/day space like SRDRN for correct weighting!

L_total = L_weighted_rec + β * L_kl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def intensity_weighted_mae_mmday(y_true, y_hat, mean_pr, std_pr,
                                  scale=36.6, w_min=0.1, w_max=2.0):
    """
    Intensity-weighted MAE in mm/day space (like SRDRN).

    Works with normalized data by:
    1. Denormalizing to log1p space: logp = y * std + mean
    2. Converting to mm/day: mm = expm1(logp)
    3. Computing weights in mm/day space: w = clip(mm_true / scale, w_min, w_max)
    4. Computing weighted MAE

    This ensures heavy rain (100 mm/day) gets weight ~2.0,
    while light rain (1 mm/day) gets weight ~0.1.

    Args:
        y_true: (B, 1, H, W) ground truth (normalized)
        y_hat: (B, 1, H, W) predictions (normalized)
        mean_pr: (H, W) or (1, H, W) mean for denormalization
        std_pr: (H, W) or (1, H, W) std for denormalization
        scale: Scaling factor in mm/day (default 36.6, same as SRDRN)
        w_min: Minimum weight (default 0.1)
        w_max: Maximum weight (default 2.0)

    Returns:
        loss: Scalar weighted MAE
        weights_mean: Mean weight (for logging)
    """
    # Ensure mean_pr and std_pr are tensors with correct shape
    if isinstance(mean_pr, np.ndarray):
        mean_pr = torch.from_numpy(mean_pr).float().to(y_true.device)
    if isinstance(std_pr, np.ndarray):
        std_pr = torch.from_numpy(std_pr).float().to(y_true.device)

    # Reshape to (1, 1, H, W) for broadcasting
    if mean_pr.dim() == 2:
        mean_pr = mean_pr.unsqueeze(0).unsqueeze(0)
    elif mean_pr.dim() == 3:
        mean_pr = mean_pr.unsqueeze(0)

    if std_pr.dim() == 2:
        std_pr = std_pr.unsqueeze(0).unsqueeze(0)
    elif std_pr.dim() == 3:
        std_pr = std_pr.unsqueeze(0)

    # 1. Denormalize to log1p space
    logp_true = y_true * std_pr + mean_pr
    logp_hat = y_hat * std_pr + mean_pr

    # 2. Convert to mm/day space
    mm_true = torch.expm1(logp_true)  # expm1(x) = exp(x) - 1, inverse of log1p
    mm_hat = torch.expm1(logp_hat)

    # 3. Compute intensity-based weights from TRUE mm/day values
    weights = torch.clamp(mm_true / scale, min=w_min, max=w_max)

    # 4. Compute weighted MAE in mm/day space
    error_mm = torch.abs(mm_hat - mm_true)
    weighted_error = weights * error_mm

    # Normalize by sum of weights
    loss = weighted_error.sum() / (weights.sum() + 1e-8)

    # For logging
    weights_mean = weights.mean()

    return loss, weights_mean


def intensity_weighted_mae_with_tail_boost_mmday(y_true, y_hat, mean_pr, std_pr, p99_mmday,
                                                   scale=36.6, w_min=0.1, w_max=2.0,
                                                   tail_boost=1.5):
    """
    Enhanced version with extra boost for P99+ pixels (in mm/day space).

    L = sum(w * boost * |mm_hat - mm_true|) / sum(w * boost)

    where:
        mm = expm1(y * std + mean)  [convert to mm/day]
        w = clip(mm_true / scale, w_min, w_max)
        boost = tail_boost if mm_true >= p99_mmday, else 1.0

    This gives even MORE emphasis to extreme tail events (P99+).

    Args:
        y_true: (B, 1, H, W) ground truth (normalized)
        y_hat: (B, 1, H, W) predictions (normalized)
        mean_pr: (H, W) normalization mean
        std_pr: (H, W) normalization std
        p99_mmday: P99 threshold in mm/day space (not normalized!)
        scale: Scaling factor
        w_min, w_max: Weight bounds
        tail_boost: Extra multiplier for P99+ pixels (default 1.5)

    Returns:
        loss: Scalar weighted MAE
        info: Dict with logging info
    """
    # Ensure mean_pr and std_pr are tensors
    if isinstance(mean_pr, np.ndarray):
        mean_pr = torch.from_numpy(mean_pr).float().to(y_true.device)
    if isinstance(std_pr, np.ndarray):
        std_pr = torch.from_numpy(std_pr).float().to(y_true.device)

    # Reshape for broadcasting
    if mean_pr.dim() == 2:
        mean_pr = mean_pr.unsqueeze(0).unsqueeze(0)
    elif mean_pr.dim() == 3:
        mean_pr = mean_pr.unsqueeze(0)

    if std_pr.dim() == 2:
        std_pr = std_pr.unsqueeze(0).unsqueeze(0)
    elif std_pr.dim() == 3:
        std_pr = std_pr.unsqueeze(0)

    # Denormalize to mm/day
    logp_true = y_true * std_pr + mean_pr
    logp_hat = y_hat * std_pr + mean_pr
    mm_true = torch.expm1(logp_true)
    mm_hat = torch.expm1(logp_hat)

    # Base intensity weights
    weights = torch.clamp(mm_true / scale, min=w_min, max=w_max)

    # Extra boost for P99+ pixels (in mm/day space)
    tail_mask = (mm_true >= p99_mmday).float()
    boost = 1.0 + (tail_boost - 1.0) * tail_mask

    # Combined weighting
    final_weights = weights * boost

    # Weighted error in mm/day space
    error_mm = torch.abs(mm_hat - mm_true)
    weighted_error = final_weights * error_mm

    # Normalize
    loss = weighted_error.sum() / (final_weights.sum() + 1e-8)

    # Info for logging
    info = {
        'weights_mean': weights.mean().item(),
        'boost_mean': boost.mean().item(),
        'n_tail_pixels': tail_mask.sum().item(),
        'mm_true_max': mm_true.max().item(),
        'mm_true_mean': mm_true.mean().item()
    }

    return loss, info


def mass_loss(y_true, y_hat, land_mask):
    """
    Mass conservation loss (keep this for physical consistency).

    L_mass = mean(|sum_land(y_true) - sum_land(y_hat)|)
    """
    if land_mask.dim() == 3:
        land_mask = land_mask.unsqueeze(0)

    mass_true = (y_true * land_mask).sum(dim=(1, 2, 3))
    mass_hat = (y_hat * land_mask).sum(dim=(1, 2, 3))

    mass_diff = torch.abs(mass_true - mass_hat)
    loss = mass_diff.mean()

    return loss


def kl_divergence(mu, logvar):
    """
    KL divergence (required for VAE).

    KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl.mean()
    return kl_loss


def gradient_loss(y_true, y_hat):
    """
    Gradient (edge) loss - forces sharp transitions between wet and dry regions.

    Computes L1 difference between spatial gradients (horizontal and vertical edges).
    This penalizes smooth/blurry predictions and encourages sharp boundaries.

    CRITICAL for fixing "drizzle everywhere" - forces model to create
    sharp transitions: "this pixel is wet, neighbor is dry".

    Args:
        y_true: (B, 1, H, W) ground truth
        y_hat: (B, 1, H, W) predictions

    Returns:
        loss: Scalar gradient loss
    """
    # Horizontal gradients (left-right edges)
    # dy[..., :, i] = abs(y[..., :, i+1] - y[..., :, i])
    dy_true = torch.abs(y_true[:, :, :, 1:] - y_true[:, :, :, :-1])
    dy_hat = torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1])

    # Vertical gradients (top-bottom edges)
    # dx[..., i, :] = abs(y[..., i+1, :] - y[..., i, :])
    dx_true = torch.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :])
    dx_hat = torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :])

    # L1 loss on gradient differences
    loss_horizontal = torch.mean(torch.abs(dy_true - dy_hat))
    loss_vertical = torch.mean(torch.abs(dx_true - dx_hat))

    return loss_horizontal + loss_vertical


class SimplifiedCVAELoss(nn.Module):
    """
    Simplified cVAE loss with edge preservation for sharp predictions.

    L_total = L_weighted_rec + λ_mass * L_mass + λ_grad * L_grad + β * L_kl

    Components:
    - L_weighted_rec: Intensity-weighted MAE (emphasizes heavy precipitation)
    - L_mass: Mass conservation (physical consistency)
    - L_grad: Gradient loss (CRITICAL - forces sharp edges, prevents drizzle everywhere)
    - L_kl: KL divergence (VAE regularization)

    UPDATED: Added gradient loss to fix "smooth background" problem!

    This creates sharp transitions between wet/dry regions.
    """

    def __init__(self,
                 p99_mmday: float,
                 mean_pr,
                 std_pr,
                 scale: float = 36.6,
                 w_min: float = 0.1,
                 w_max: float = 2.0,
                 tail_boost: float = 1.5,
                 lambda_mass: float = 0.01,
                 lambda_grad: float = 1.0,
                 beta_kl: float = 0.01,
                 warmup_epochs: int = 15):
        """
        Args:
            p99_mmday: P99 threshold in mm/day space (not normalized!)
            mean_pr: (H, W) normalization mean for denormalization
            std_pr: (H, W) normalization std for denormalization
            scale: Intensity weighting scale in mm/day (default 36.6, same as SRDRN)
                   - Same scale works because we denormalize to mm/day first!
            w_min, w_max: Weight bounds
            tail_boost: Extra multiplier for P99+ (default 1.5)
            lambda_mass: Weight for mass conservation (keep small)
            lambda_grad: Weight for gradient (edge) loss (CRITICAL for sharp edges)
            beta_kl: Final KL weight (required for VAE)
            warmup_epochs: KL warmup epochs
        """
        super().__init__()

        self.p99_mmday = p99_mmday
        self.scale = scale
        self.w_min = w_min
        self.w_max = w_max
        self.tail_boost = tail_boost
        self.lambda_mass = lambda_mass
        self.lambda_grad = lambda_grad
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs

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
            self.current_beta = self.beta_kl * (epoch / self.warmup_epochs)
        else:
            self.current_beta = self.beta_kl

    def forward(self, y_true, y_hat, mu, logvar, land_mask):
        """
        Compute loss with edge preservation.

        Args:
            y_true: (B, 1, H, W) normalized precipitation
            y_hat: (B, 1, H, W) normalized predictions
            mu: (B, d_z)
            logvar: (B, d_z)
            land_mask: (B, 1, H, W) or (1, 1, H, W)

        Returns:
            loss_dict: All loss components
        """
        # 1. Weighted reconstruction loss in mm/day space
        rec_loss, rec_info = intensity_weighted_mae_with_tail_boost_mmday(
            y_true, y_hat, self.mean_pr, self.std_pr, self.p99_mmday,
            self.scale, self.w_min, self.w_max, self.tail_boost
        )

        # 2. Mass conservation (physical consistency)
        m_loss = mass_loss(y_true, y_hat, land_mask)

        # 3. Gradient (edge) loss - CRITICAL for sharp predictions!
        # Forces model to create sharp wet/dry transitions, not smooth drizzle
        grad_loss = gradient_loss(y_true, y_hat)

        # 4. KL divergence (VAE regularization)
        kl_loss = kl_divergence(mu, logvar)

        # Total loss with edge preservation
        total_loss = (rec_loss +
                     self.lambda_mass * m_loss +
                     self.lambda_grad * grad_loss +
                     self.current_beta * kl_loss)

        # For logging
        loss_dict = {
            'loss': total_loss,
            'L_rec': rec_loss,
            'L_mass': m_loss,
            'L_grad': grad_loss,
            'L_kl': kl_loss,
            'beta': self.current_beta,
            'weights_mean': rec_info['weights_mean'],
            'n_tail_pixels': rec_info['n_tail_pixels'],
            'mm_true_max': rec_info['mm_true_max'],
            'mm_true_mean': rec_info['mm_true_mean']
        }

        return loss_dict


class MinimalCVAELoss(nn.Module):
    """
    MINIMAL loss: Just weighted reconstruction + KL.

    L_total = L_weighted_rec + β * L_kl

    UPDATED: Now works in mm/day space like SRDRN!

    Most similar to standard supervised learning.
    Remove mass conservation if you want absolute minimum complexity.
    """

    def __init__(self,
                 p99_mmday: float,
                 mean_pr,
                 std_pr,
                 scale: float = 36.6,
                 tail_boost: float = 1.5,
                 beta_kl: float = 0.01,
                 warmup_epochs: int = 15):
        super().__init__()

        self.p99_mmday = p99_mmday
        self.scale = scale
        self.tail_boost = tail_boost
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs

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
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            self.current_beta = self.beta_kl * (epoch / self.warmup_epochs)
        else:
            self.current_beta = self.beta_kl

    def forward(self, y_true, y_hat, mu, logvar, land_mask=None):
        """
        Minimal loss computation in mm/day space.

        Only TWO terms: reconstruction + KL!
        """
        # Weighted reconstruction in mm/day space
        rec_loss, rec_info = intensity_weighted_mae_with_tail_boost_mmday(
            y_true, y_hat, self.mean_pr, self.std_pr, self.p99_mmday,
            self.scale, 0.1, 2.0, self.tail_boost
        )

        # KL divergence
        kl_loss = kl_divergence(mu, logvar)

        # Total loss (ONLY 2 TERMS!)
        total_loss = rec_loss + self.current_beta * kl_loss

        loss_dict = {
            'loss': total_loss,
            'L_rec': rec_loss,
            'L_kl': kl_loss,
            'beta': self.current_beta,
            'weights_mean': rec_info['weights_mean'],
            'mm_true_max': rec_info['mm_true_max']
        }

        return loss_dict


def test_losses():
    """Test the simplified losses."""
    print("Testing simplified losses...")

    # Dummy data
    B, H, W = 4, 156, 132
    y_true = torch.randn(B, 1, H, W).abs() * 10  # 0-10 mm/day
    y_hat = y_true + torch.randn(B, 1, H, W) * 0.5
    land_mask = torch.ones(1, 1, H, W)
    mu = torch.randn(B, 64)
    logvar = torch.randn(B, 64)

    p99 = float(torch.quantile(y_true, 0.99))

    # Test intensity-weighted MAE
    print("\n1. Intensity-weighted MAE:")
    loss, weights_mean = intensity_weighted_mae(y_true, y_hat)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Mean weight: {weights_mean.item():.4f}")

    # Test with tail boost
    print("\n2. With tail boost:")
    loss, info = intensity_weighted_mae_with_tail_boost(y_true, y_hat, p99)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Tail pixels: {info['n_tail_pixels']}")

    # Test simplified criterion
    print("\n3. Simplified criterion:")
    criterion = SimplifiedCVAELoss(p99=p99)
    criterion.update_beta(epoch=10)
    loss_dict = criterion(y_true, y_hat, mu, logvar, land_mask)
    print(f"   Total loss: {loss_dict['loss'].item():.4f}")
    print(f"   L_rec: {loss_dict['L_rec'].item():.4f}")
    print(f"   L_kl: {loss_dict['L_kl'].item():.4f}")

    # Test minimal criterion
    print("\n4. Minimal criterion (2 terms only):")
    criterion = MinimalCVAELoss(p99=p99)
    criterion.update_beta(epoch=10)
    loss_dict = criterion(y_true, y_hat, mu, logvar)
    print(f"   Total loss: {loss_dict['loss'].item():.4f}")
    print(f"   L_rec: {loss_dict['L_rec'].item():.4f}")
    print(f"   L_kl: {loss_dict['L_kl'].item():.4f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_losses()
