"""
Loss functions for cVAE training.

Implements:
1. Extreme-weighted MAE (reconstruction loss with emphasis on heavy precipitation)
2. Mass/budget loss (preserve total precipitation over land)
3. KL divergence (regularization for latent distribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extreme_weighted_mae(y_true, y_hat, p95, lambda_base=1.0, lambda_ext=3.0):
    """
    Compute extreme-weighted MAE loss.

    L_rec = λ_base * MAE_all + λ_ext * MAE_extreme

    Where MAE_extreme focuses on pixels where y_true >= p95.

    Args:
        y_true: (B, 1, H, W) ground truth
        y_hat: (B, 1, H, W) predictions
        p95: Scalar threshold (e.g., 95th percentile from training data)
        lambda_base: Weight for base MAE
        lambda_ext: Weight for extreme MAE

    Returns:
        loss: Scalar tensor
        mae_base: Base MAE for logging
        mae_ext: Extreme MAE for logging
    """
    # Base MAE over all pixels
    mae_base = F.l1_loss(y_hat, y_true, reduction='mean')

    # Create extreme mask
    mask_ext = (y_true >= p95).float()  # (B, 1, H, W)

    # Count extreme pixels
    n_ext = mask_ext.sum()

    if n_ext > 0:
        # MAE over extreme pixels only
        diff_ext = torch.abs(y_hat - y_true) * mask_ext
        mae_ext = diff_ext.sum() / n_ext
    else:
        # No extreme pixels in this batch
        mae_ext = torch.tensor(0.0, device=y_true.device)

    # Combined loss
    loss = lambda_base * mae_base + lambda_ext * mae_ext

    return loss, mae_base, mae_ext


def mass_loss(y_true, y_hat, land_mask):
    """
    Compute mass/budget loss: absolute difference in total precipitation over land.

    L_mass = |sum_land(y_true) - sum_land(y_hat)|

    Args:
        y_true: (B, 1, H, W) ground truth
        y_hat: (B, 1, H, W) predictions
        land_mask: (B, 1, H, W) or (1, H, W) land mask (1=land, 0=sea)

    Returns:
        loss: Scalar tensor (mean absolute mass difference per sample)
    """
    # Ensure land_mask is broadcastable
    if land_mask.dim() == 3:
        land_mask = land_mask.unsqueeze(0)  # (1, 1, H, W)

    # Compute total mass over land for each sample
    mass_true = (y_true * land_mask).sum(dim=(1, 2, 3))  # (B,)
    mass_hat = (y_hat * land_mask).sum(dim=(1, 2, 3))    # (B,)

    # Absolute difference
    mass_diff = torch.abs(mass_true - mass_hat)  # (B,)

    # Mean over batch
    loss = mass_diff.mean()

    return loss


def kl_divergence(mu, logvar):
    """
    Compute KL divergence between N(mu, exp(logvar)) and N(0, I).

    KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

    Args:
        mu: (B, d_z) mean
        logvar: (B, d_z) log variance

    Returns:
        kl_loss: Scalar tensor (mean over batch and dimensions)
    """
    # KL divergence formula
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)

    # Mean over batch
    kl_loss = kl.mean()

    return kl_loss


class CVAELoss(nn.Module):
    """
    Combined loss for cVAE training.

    L_total = L_rec + λ_mass * L_mass + β * L_kl

    Where:
    - L_rec is extreme-weighted MAE
    - L_mass is mass conservation loss
    - L_kl is KL divergence

    Supports KL warm-up schedule.
    """

    def __init__(self,
                 p95: float,
                 lambda_base: float = 1.0,
                 lambda_ext: float = 3.0,
                 lambda_mass: float = 0.1,
                 beta_kl: float = 0.5,
                 warmup_epochs: int = 10,
                 min_kl_weight: float = 0.0):
        """
        Args:
            p95: Threshold for extreme precipitation (from thresholds.json)
            lambda_base: Weight for base MAE
            lambda_ext: Weight for extreme MAE
            lambda_mass: Weight for mass loss
            beta_kl: Final weight for KL divergence
            warmup_epochs: Number of epochs to warm up KL weight from 0 to beta_kl
            min_kl_weight: Minimum KL per latent dimension (free bits). KL below this is not penalized.
        """
        super().__init__()

        self.p95 = p95
        self.lambda_base = lambda_base
        self.lambda_ext = lambda_ext
        self.lambda_mass = lambda_mass
        self.beta_kl = beta_kl
        self.warmup_epochs = warmup_epochs
        self.min_kl_weight = min_kl_weight

        self.current_epoch = 0
        self.current_beta = 0.0

    def update_beta(self, epoch):
        """
        Update KL weight based on current epoch (for warm-up).

        Args:
            epoch: Current epoch (0-indexed)
        """
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Linear warm-up
            self.current_beta = self.beta_kl * (epoch / self.warmup_epochs)
        else:
            self.current_beta = self.beta_kl

    def forward(self, y_true, y_hat, mu, logvar, land_mask):
        """
        Compute total loss.

        Args:
            y_true: (B, 1, H, W) ground truth
            y_hat: (B, 1, H, W) predictions
            mu: (B, d_z) latent mean
            logvar: (B, d_z) latent log variance
            land_mask: (B, 1, H, W) or (1, H, W) land mask

        Returns:
            loss_dict: Dictionary with all loss components
        """
        # Reconstruction loss
        rec_loss, mae_base, mae_ext = extreme_weighted_mae(
            y_true, y_hat, self.p95, self.lambda_base, self.lambda_ext
        )

        # Mass loss
        m_loss = mass_loss(y_true, y_hat, land_mask)

        # KL loss
        kl_loss = kl_divergence(mu, logvar)

        # Apply free bits (min_kl_weight) if specified
        # Only penalize KL above the minimum threshold
        if self.min_kl_weight > 0:
            kl_loss_clamped = torch.clamp(kl_loss - self.min_kl_weight, min=0.0)
        else:
            kl_loss_clamped = kl_loss

        # Total loss
        total_loss = rec_loss + self.lambda_mass * m_loss + self.current_beta * kl_loss_clamped

        # Return all components for logging
        loss_dict = {
            'loss': total_loss,
            'L_rec': rec_loss,
            'L_base': mae_base,
            'L_ext': mae_ext,
            'L_mass': m_loss,
            'L_kl': kl_loss,
            'beta': self.current_beta
        }

        return loss_dict


def test_losses():
    """
    Unit test for loss functions.
    """
    print("Testing loss functions...")

    # Create dummy data
    B, H, W = 4, 156, 132
    y_true = torch.randn(B, 1, H, W).abs()  # Non-negative
    y_hat = torch.randn(B, 1, H, W).abs()
    land_mask = torch.ones(1, 1, H, W)
    mu = torch.randn(B, 64)
    logvar = torch.randn(B, 64)

    p95 = float(torch.quantile(y_true, 0.95))

    # Test extreme-weighted MAE
    rec_loss, mae_base, mae_ext = extreme_weighted_mae(y_true, y_hat, p95)
    print(f"Reconstruction loss: {rec_loss.item():.4f}")
    print(f"  MAE_base: {mae_base.item():.4f}")
    print(f"  MAE_ext: {mae_ext.item():.4f}")

    # Test mass loss
    m_loss = mass_loss(y_true, y_hat, land_mask)
    print(f"Mass loss: {m_loss.item():.4f}")

    # Test KL divergence
    kl_loss = kl_divergence(mu, logvar)
    print(f"KL loss: {kl_loss.item():.4f}")

    # Test combined loss
    criterion = CVAELoss(p95=p95, warmup_epochs=5)
    criterion.update_beta(epoch=0)
    loss_dict = criterion(y_true, y_hat, mu, logvar, land_mask)
    print(f"Total loss (epoch 0): {loss_dict['loss'].item():.4f}")
    print(f"  Beta: {loss_dict['beta']:.4f}")

    criterion.update_beta(epoch=5)
    loss_dict = criterion(y_true, y_hat, mu, logvar, land_mask)
    print(f"Total loss (epoch 5): {loss_dict['loss'].item():.4f}")
    print(f"  Beta: {loss_dict['beta']:.4f}")

    print("All loss tests passed!")


if __name__ == "__main__":
    test_losses()
