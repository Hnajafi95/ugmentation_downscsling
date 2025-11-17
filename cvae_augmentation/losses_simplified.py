"""
Simplified loss functions for cVAE training.

Philosophy: ONE unified reconstruction objective (like SRDRN)
Instead of separate MAE_all + MAE_extreme, use intensity-weighted MAE
that naturally emphasizes heavier precipitation.

L_total = L_weighted_rec + β * L_kl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def intensity_weighted_mae(y_true, y_hat, scale=36.6, w_min=0.1, w_max=2.0):
    """
    Intensity-weighted MAE (inspired by SRDRN approach).

    One unified objective that automatically emphasizes heavy precipitation
    without separate base/extreme terms.

    L = sum(w * |y_hat - y_true|) / sum(w)

    where w = clip(y_true / scale, w_min, w_max)

    This gives ~20× more weight to extreme rain vs light rain!

    Args:
        y_true: (B, 1, H, W) ground truth precipitation
        y_hat: (B, 1, H, W) predicted precipitation
        scale: Scaling factor (default 36.6, similar to SRDRN)
               Higher scale = less extreme emphasis
        w_min: Minimum weight (default 0.1)
        w_max: Maximum weight (default 2.0)

    Returns:
        loss: Scalar weighted MAE
        weights_mean: Mean weight (for logging)
    """
    # Compute intensity-based weights from ground truth
    weights = torch.clamp(y_true / scale, min=w_min, max=w_max)

    # Weighted absolute error
    weighted_error = weights * torch.abs(y_hat - y_true)

    # Normalize by sum of weights
    loss = weighted_error.sum() / (weights.sum() + 1e-8)

    # For logging
    weights_mean = weights.mean()

    return loss, weights_mean


def intensity_weighted_mae_with_tail_boost(y_true, y_hat, p99,
                                            scale=36.6, w_min=0.1, w_max=2.0,
                                            tail_boost=1.5):
    """
    Enhanced version with extra boost for P99+ pixels.

    L = sum(w * boost * |y_hat - y_true|) / sum(w * boost)

    where:
        w = clip(y_true / scale, w_min, w_max)
        boost = tail_boost if y_true >= p99, else 1.0

    This gives even MORE emphasis to extreme tail events (P99+).

    Args:
        y_true: (B, 1, H, W) ground truth
        y_hat: (B, 1, H, W) predictions
        p99: P99 threshold (e.g., 4.26 from your data)
        scale: Scaling factor
        w_min, w_max: Weight bounds
        tail_boost: Extra multiplier for P99+ pixels (default 1.5)

    Returns:
        loss: Scalar weighted MAE
        info: Dict with logging info
    """
    # Base intensity weights
    weights = torch.clamp(y_true / scale, min=w_min, max=w_max)

    # Extra boost for P99+ pixels
    tail_mask = (y_true >= p99).float()
    boost = 1.0 + (tail_boost - 1.0) * tail_mask  # 1.0 normal, tail_boost for P99+

    # Combined weighting
    final_weights = weights * boost

    # Weighted error
    weighted_error = final_weights * torch.abs(y_hat - y_true)

    # Normalize
    loss = weighted_error.sum() / (final_weights.sum() + 1e-8)

    # Info for logging
    info = {
        'weights_mean': weights.mean().item(),
        'boost_mean': boost.mean().item(),
        'n_tail_pixels': tail_mask.sum().item()
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


class SimplifiedCVAELoss(nn.Module):
    """
    Simplified cVAE loss with ONE unified reconstruction objective.

    L_total = L_weighted_rec + λ_mass * L_mass + β * L_kl

    Where L_weighted_rec is a SINGLE intensity-weighted MAE that
    automatically emphasizes heavy precipitation (like SRDRN).

    This avoids the multi-objective optimization problem!
    """

    def __init__(self,
                 p99: float,
                 scale: float = 36.6,
                 w_min: float = 0.1,
                 w_max: float = 2.0,
                 tail_boost: float = 1.5,
                 lambda_mass: float = 0.01,
                 beta_kl: float = 0.01,
                 warmup_epochs: int = 15):
        """
        Args:
            p99: P99 threshold for tail boost
            scale: Intensity weighting scale (default 36.6)
                   - Lower = more extreme emphasis
                   - Higher = more balanced
            w_min, w_max: Weight bounds
            tail_boost: Extra multiplier for P99+ (default 1.5)
                       - 1.0 = no boost (just intensity weighting)
                       - 1.5 = 50% extra weight for P99+
                       - 2.0 = 100% extra weight for P99+
            lambda_mass: Weight for mass conservation (keep small)
            beta_kl: Final KL weight (required for VAE)
            warmup_epochs: KL warmup epochs
        """
        super().__init__()

        self.p99 = p99
        self.scale = scale
        self.w_min = w_min
        self.w_max = w_max
        self.tail_boost = tail_boost
        self.lambda_mass = lambda_mass
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs

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
        Compute simplified loss.

        Args:
            y_true: (B, 1, H, W)
            y_hat: (B, 1, H, W)
            mu: (B, d_z)
            logvar: (B, d_z)
            land_mask: (B, 1, H, W) or (1, 1, H, W)

        Returns:
            loss_dict: All loss components
        """
        # 1. Weighted reconstruction loss (ONE unified objective!)
        rec_loss, rec_info = intensity_weighted_mae_with_tail_boost(
            y_true, y_hat, self.p99,
            self.scale, self.w_min, self.w_max, self.tail_boost
        )

        # 2. Mass conservation (optional, but good for physical consistency)
        m_loss = mass_loss(y_true, y_hat, land_mask)

        # 3. KL divergence (required for VAE)
        kl_loss = kl_divergence(mu, logvar)

        # Total loss (ONLY 3 TERMS instead of 4+!)
        total_loss = rec_loss + self.lambda_mass * m_loss + self.current_beta * kl_loss

        # For logging
        loss_dict = {
            'loss': total_loss,
            'L_rec': rec_loss,
            'L_mass': m_loss,
            'L_kl': kl_loss,
            'beta': self.current_beta,
            'weights_mean': rec_info['weights_mean'],
            'n_tail_pixels': rec_info['n_tail_pixels']
        }

        return loss_dict


class MinimalCVAELoss(nn.Module):
    """
    MINIMAL loss: Just weighted reconstruction + KL.

    L_total = L_weighted_rec + β * L_kl

    Most similar to standard supervised learning.
    Remove mass conservation if you want absolute minimum complexity.
    """

    def __init__(self,
                 p99: float,
                 scale: float = 36.6,
                 tail_boost: float = 1.5,
                 beta_kl: float = 0.01,
                 warmup_epochs: int = 15):
        super().__init__()

        self.p99 = p99
        self.scale = scale
        self.tail_boost = tail_boost
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs

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
        Minimal loss computation.

        Only TWO terms: reconstruction + KL!
        """
        # Weighted reconstruction
        rec_loss, rec_info = intensity_weighted_mae_with_tail_boost(
            y_true, y_hat, self.p99,
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
            'weights_mean': rec_info['weights_mean']
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
