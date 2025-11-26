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
        mask: Optional (B, 1, H, W) or (1, 1, H, W) mask (1=include, 0=exclude)

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
        # Broadcast mask to match data shape if needed
        if mask.shape[0] == 1 and diff.shape[0] > 1:
            mask = np.broadcast_to(mask, diff.shape)
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
        mask: Optional (B, 1, H, W) or (1, 1, H, W) mask

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
        # Broadcast mask to match data shape if needed
        if mask.shape[0] == 1 and sq_diff.shape[0] > 1:
            mask = np.broadcast_to(mask, sq_diff.shape)
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
        mask: Optional (B, 1, H, W) or (1, 1, H, W) mask

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
        # Broadcast mask to match data shape if needed
        if mask.shape[0] == 1 and y_true.shape[0] > 1:
            mask = np.broadcast_to(mask, y_true.shape)
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
        land_mask: (B, 1, H, W) or (1, 1, H, W) land mask

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

    # Broadcast mask to match data shape if needed (for proper multiplication)
    if land_mask.shape[0] == 1 and y_true.shape[0] > 1:
        land_mask = np.broadcast_to(land_mask, y_true.shape)

    # Compute total mass over land
    mass_true = np.sum(y_true * land_mask)
    mass_hat = np.sum(y_hat * land_mask)

    mass_bias = mass_hat - mass_true

    return float(mass_bias)


def denormalize_to_mmday(y_normalized, mean_pr, std_pr):
    """
    Convert normalized precipitation (z-space) to mm/day.

    Inverse of: z = (log1p(precip) - mean) / std
    So: precip = expm1(z * std + mean)

    Args:
        y_normalized: (B, 1, H, W) normalized values
        mean_pr: (H, W) mean of log1p precipitation
        std_pr: (H, W) std of log1p precipitation

    Returns:
        y_mmday: (B, 1, H, W) in mm/day
    """
    if isinstance(y_normalized, torch.Tensor):
        y_normalized = y_normalized.detach().cpu().numpy()

    # Reshape mean/std for broadcasting: (H, W) -> (1, 1, H, W)
    if mean_pr.ndim == 2:
        mean_pr = mean_pr[np.newaxis, np.newaxis, :, :]
        std_pr = std_pr[np.newaxis, np.newaxis, :, :]

    # Denormalize: z -> log1p(precip) -> precip
    log1p_precip = y_normalized * std_pr + mean_pr
    y_mmday = np.expm1(log1p_precip)

    # Clip negative values (can happen due to prediction errors)
    y_mmday = np.maximum(y_mmday, 0.0)

    return y_mmday


class MetricsTracker:
    """
    Track and accumulate metrics over multiple batches.

    Can compute metrics in either z-space (normalized) or mm/day space.
    For alignment with the loss function, mm/day space is recommended.
    """

    def __init__(self, p95, land_mask, mean_pr=None, std_pr=None, p95_mmday=None):
        """
        Args:
            p95: Threshold for tail metrics (in z-space, for backward compatibility)
            land_mask: (1, H, W) or (1, 1, H, W) land mask
            mean_pr: Optional (H, W) mean for denormalization to mm/day
            std_pr: Optional (H, W) std for denormalization to mm/day
            p95_mmday: Optional P95 threshold in mm/day space
        """
        self.p95 = p95  # z-space threshold (for backward compatibility)
        self.land_mask = land_mask

        # For mm/day computation
        self.mean_pr = mean_pr
        self.std_pr = std_pr
        self.p95_mmday = p95_mmday
        self.use_mmday = mean_pr is not None and std_pr is not None and p95_mmday is not None

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

        If mm/day normalization params are provided, computes metrics in mm/day space
        for proper alignment with the loss function. Otherwise uses z-space.

        Returns:
            metrics: Dictionary of metrics
        """
        if len(self.y_true_all) == 0:
            return {}

        # Concatenate all batches
        y_true = torch.cat(self.y_true_all, dim=0)  # (N, 1, H, W)
        y_hat = torch.cat(self.y_hat_all, dim=0)    # (N, 1, H, W)

        if self.use_mmday:
            # Convert to mm/day space for metrics aligned with loss function
            y_true_mmday = denormalize_to_mmday(y_true, self.mean_pr, self.std_pr)
            y_hat_mmday = denormalize_to_mmday(y_hat, self.mean_pr, self.std_pr)

            # Compute metrics in mm/day space
            metrics = {
                'MAE_all': compute_mae(y_true_mmday, y_hat_mmday, mask=self.land_mask),
                'RMSE_all': compute_rmse(y_true_mmday, y_hat_mmday, mask=self.land_mask),
                'MAE_tail': compute_tail_mae(y_true_mmday, y_hat_mmday, self.p95_mmday, mask=self.land_mask),
                'mass_bias': compute_mass_bias(y_true_mmday, y_hat_mmday, self.land_mask)
            }

            # Also compute z-space metrics for comparison
            metrics['MAE_all_zspace'] = compute_mae(y_true, y_hat, mask=self.land_mask)
            metrics['MAE_tail_zspace'] = compute_tail_mae(y_true, y_hat, self.p95, mask=self.land_mask)
        else:
            # Compute metrics in z-space (original behavior)
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
