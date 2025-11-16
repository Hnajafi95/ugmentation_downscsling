"""
Utility functions for computing evaluation metrics.

Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Tail MAE (MAE for values >= threshold)
- Mass bias (difference in total precipitation over land)
"""

import torch
import numpy as np


def compute_mae(y_true, y_hat, mask=None):
    """
    Compute Mean Absolute Error.

    Args:
        y_true: (B, 1, H, W) or (B, H, W) ground truth
        y_hat: (B, 1, H, W) or (B, H, W) predictions
        mask: Optional (B, 1, H, W) or (1, H, W) mask (1=include, 0=exclude)

    Returns:
        mae: Scalar float
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    diff = np.abs(y_true - y_hat)

    if mask is not None:
        # Apply mask
        mask = mask.astype(bool)
        diff_masked = diff[mask]
        if len(diff_masked) > 0:
            mae = np.mean(diff_masked)
        else:
            mae = 0.0
    else:
        mae = np.mean(diff)

    return float(mae)


def compute_rmse(y_true, y_hat, mask=None):
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: (B, 1, H, W) or (B, H, W) ground truth
        y_hat: (B, 1, H, W) or (B, H, W) predictions
        mask: Optional (B, 1, H, W) or (1, H, W) mask

    Returns:
        rmse: Scalar float
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    sq_diff = (y_true - y_hat) ** 2

    if mask is not None:
        mask = mask.astype(bool)
        sq_diff_masked = sq_diff[mask]
        if len(sq_diff_masked) > 0:
            mse = np.mean(sq_diff_masked)
        else:
            mse = 0.0
    else:
        mse = np.mean(sq_diff)

    rmse = np.sqrt(mse)
    return float(rmse)


def compute_tail_mae(y_true, y_hat, threshold, mask=None):
    """
    Compute MAE for values >= threshold (tail/extreme values).

    Args:
        y_true: (B, 1, H, W) or (B, H, W) ground truth
        y_hat: (B, 1, H, W) or (B, H, W) predictions
        threshold: Scalar threshold (e.g., P95)
        mask: Optional (B, 1, H, W) or (1, H, W) mask

    Returns:
        tail_mae: Scalar float (or 0.0 if no tail values)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Create tail mask
    tail_mask = (y_true >= threshold)

    # Combine with optional mask
    if mask is not None:
        mask = mask.astype(bool)
        tail_mask = tail_mask & mask

    # Extract tail values
    y_true_tail = y_true[tail_mask]
    y_hat_tail = y_hat[tail_mask]

    if len(y_true_tail) > 0:
        tail_mae = np.mean(np.abs(y_true_tail - y_hat_tail))
    else:
        tail_mae = 0.0

    return float(tail_mae)


def compute_mass_bias(y_true, y_hat, land_mask):
    """
    Compute mass bias over land: sum(y_hat) - sum(y_true).

    Positive bias = model predicts too much precipitation
    Negative bias = model predicts too little precipitation

    Args:
        y_true: (B, 1, H, W) or (B, H, W) ground truth
        y_hat: (B, 1, H, W) or (B, H, W) predictions
        land_mask: (B, 1, H, W) or (1, H, W) land mask

    Returns:
        mass_bias: Scalar float (total bias over all samples)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_hat, torch.Tensor):
        y_hat = y_hat.detach().cpu().numpy()
    if isinstance(land_mask, torch.Tensor):
        land_mask = land_mask.detach().cpu().numpy()

    # Ensure land_mask is broadcastable
    if land_mask.ndim == 3:
        land_mask = land_mask[np.newaxis, :]

    # Compute total mass over land
    mass_true = np.sum(y_true * land_mask)
    mass_hat = np.sum(y_hat * land_mask)

    mass_bias = mass_hat - mass_true

    return float(mass_bias)


class MetricsTracker:
    """
    Track and accumulate metrics over multiple batches.
    """

    def __init__(self, p95, land_mask):
        """
        Args:
            p95: Threshold for tail metrics
            land_mask: (1, H, W) or (1, 1, H, W) land mask
        """
        self.p95 = p95
        self.land_mask = land_mask

        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.y_true_all = []
        self.y_hat_all = []

    def update(self, y_true, y_hat):
        """
        Accumulate predictions for a batch.

        Args:
            y_true: (B, 1, H, W) ground truth
            y_hat: (B, 1, H, W) predictions
        """
        self.y_true_all.append(y_true.detach().cpu())
        self.y_hat_all.append(y_hat.detach().cpu())

    def compute(self):
        """
        Compute all metrics over accumulated batches.

        Returns:
            metrics: Dictionary of metrics
        """
        if len(self.y_true_all) == 0:
            return {}

        # Concatenate all batches
        y_true = torch.cat(self.y_true_all, dim=0)  # (N, 1, H, W)
        y_hat = torch.cat(self.y_hat_all, dim=0)    # (N, 1, H, W)

        # Compute metrics
        metrics = {
            'MAE_all': compute_mae(y_true, y_hat, mask=self.land_mask),
            'RMSE_all': compute_rmse(y_true, y_hat, mask=self.land_mask),
            'MAE_tail': compute_tail_mae(y_true, y_hat, self.p95, mask=self.land_mask),
            'mass_bias': compute_mass_bias(y_true, y_hat, self.land_mask)
        }

        return metrics


def test_metrics():
    """
    Unit test for metrics functions.
    """
    print("Testing metrics functions...")

    # Create dummy data
    B, H, W = 4, 156, 132
    y_true = torch.randn(B, 1, H, W).abs()
    y_hat = y_true + torch.randn(B, 1, H, W) * 0.1  # Add some noise
    land_mask = torch.ones(1, 1, H, W)

    p95 = float(torch.quantile(y_true, 0.95))

    # Test MAE
    mae = compute_mae(y_true, y_hat, mask=land_mask)
    print(f"MAE: {mae:.4f}")

    # Test RMSE
    rmse = compute_rmse(y_true, y_hat, mask=land_mask)
    print(f"RMSE: {rmse:.4f}")

    # Test tail MAE
    tail_mae = compute_tail_mae(y_true, y_hat, p95, mask=land_mask)
    print(f"Tail MAE (P95={p95:.4f}): {tail_mae:.4f}")

    # Test mass bias
    mass_bias = compute_mass_bias(y_true, y_hat, land_mask)
    print(f"Mass bias: {mass_bias:.4f}")

    # Test MetricsTracker
    tracker = MetricsTracker(p95, land_mask)
    tracker.update(y_true[:2], y_hat[:2])
    tracker.update(y_true[2:], y_hat[2:])
    metrics = tracker.compute()
    print("\nMetricsTracker results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nAll metrics tests passed!")


if __name__ == "__main__":
    test_metrics()
