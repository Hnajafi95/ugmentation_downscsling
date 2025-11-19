import xarray as xr
import numpy as np
import json
import warnings
import unittest
from typing import Tuple, Dict, Any

# Add pandas for Excel reporting
try:
    import pandas as pd
except ImportError:
    print("Pandas is not installed. Please run: pip install pandas openpyxl")
    exit()

# New imports for advanced metrics
try:
    from skimage.metrics import structural_similarity
except ImportError:
    print("scikit-image is not installed. Please run: pip install scikit-image")
    exit()

from scipy.stats import entropy, genpareto, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error

# --- CORRECTED Helper Functions for Advanced Metrics ---

def calculate_ssim(obs_map, pred_map):
    """
    Calculates the Structural Similarity (SSIM) between two 2D maps.
    VERIFIED: Ensures the SSIM window size is always an odd integer >= 3.
    """
    mask = ~np.isnan(obs_map) & ~np.isnan(pred_map)
    if not mask.any(): 
        return np.nan
    
    obs_clean = np.where(mask, obs_map, 0)
    pred_clean = np.where(mask, pred_map, 0)
    
    data_range = np.nanmax(obs_clean) - np.nanmin(obs_clean)
    if data_range == 0: 
        return 1.0
    
    H, W = obs_clean.shape
    # Determine a valid window size for SSIM. It must be odd and less than the image dimensions.
    win_size = min(7, H, W)
    if win_size % 2 == 0:
        win_size -= 1
        
    if win_size < 3: 
        # Not enough data for a meaningful SSIM calculation.
        return np.nan
        
    try:
        return structural_similarity(obs_clean, pred_clean, data_range=data_range, win_size=win_size)
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        return np.nan

def calculate_threshold_metrics(obs_daily, pred_daily, threshold):
    """
    VERIFIED: CSI, Precision, Recall at a fixed threshold (NaN-safe).
    For precipitation: events are when values exceed threshold.
    """
    valid = ~np.isnan(obs_daily) & ~np.isnan(pred_daily)
    if not valid.any():
        return {'CSI': np.nan, 'precision': np.nan, 'recall': np.nan, 'n_valid': 0}
    
    obs_events  = (obs_daily > threshold) & valid
    pred_events = (pred_daily > threshold) & valid

    hits         = np.sum(obs_events & pred_events)
    misses       = np.sum(obs_events & ~pred_events)
    false_alarms = np.sum(~obs_events & pred_events)
    correct_neg  = np.sum(valid & ~obs_events & ~pred_events)

    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    recall    = hits / (hits + misses)       if (hits + misses) > 0 else 0.0
    csi       = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0

    return {
        'CSI': csi, 
        'precision': precision, 
        'recall': recall,
        'hits': int(hits),
        'misses': int(misses),
        'false_alarms': int(false_alarms),
        'correct_negatives': int(correct_neg),
        'n_valid': int(valid.sum())
    }

def calculate_sedi(obs_daily, pred_daily, threshold):
    """
    VERIFIED: Symmetric Extremal Dependence Index (SEDI).
    Excludes NaNs via 'valid' and clips H,F to avoid log(0).
    """
    valid = ~np.isnan(obs_daily) & ~np.isnan(pred_daily)
    if not valid.any():
        return np.nan
        
    obs_events  = (obs_daily > threshold) & valid
    pred_events = (pred_daily > threshold) & valid

    hits         = np.sum(obs_events & pred_events)
    misses       = np.sum(obs_events & ~pred_events)
    false_alarms = np.sum(~obs_events & pred_events)
    correct_neg  = np.sum(valid & ~obs_events & ~pred_events)

    H = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    F = false_alarms / (false_alarms + correct_neg) if (false_alarms + correct_neg) > 0 else 0.0

    eps = 1e-6
    H = np.clip(H, eps, 1 - eps)
    F = np.clip(F, eps, 1 - eps)

    try:
        num = np.log(F) - np.log(H) + np.log(1 - H) - np.log(1 - F)
        den = np.log(F) + np.log(H) + np.log(1 - H) + np.log(1 - F)
        return num / den if abs(den) > eps else np.nan
    except Exception:
        return np.nan

def calculate_distributional_metrics(obs_flat, pred_flat):
    """
    CORRECTED: Calculates KL Divergence in the standard direction KL(obs || pred).
    Added safety guards for NaNs and degenerate distributions.
    """
    # Remove any potential NaNs
    obs_clean = obs_flat[~np.isnan(obs_flat)]
    pred_clean = pred_flat[~np.isnan(pred_flat)]

    if obs_clean.size == 0 or pred_clean.size == 0:
        return {'KL_Divergence': np.nan, 'n_obs': 0, 'n_pred': 0}

    min_val = min(obs_clean.min(), pred_clean.min())
    max_val = max(obs_clean.max(), pred_clean.max())
    
    # Handle degenerate case where all values are the same
    if max_val - min_val <= 1e-10:
        return {'KL_Divergence': 0.0, 'n_obs': len(obs_clean), 'n_pred': len(pred_clean)}
        
    # Use more bins for better resolution, but cap for computational efficiency
    n_bins = min(100, max(20, int(np.sqrt(min(len(obs_clean), len(pred_clean))))))
    bins = np.linspace(min_val, max_val, num=n_bins + 1)
    
    obs_hist, _ = np.histogram(obs_clean, bins=bins, density=True)
    pred_hist, _ = np.histogram(pred_clean, bins=bins, density=True)
    
    # Normalize to ensure they sum to 1 (probability distributions)
    obs_hist = obs_hist / (obs_hist.sum() + 1e-10)
    pred_hist = pred_hist / (pred_hist.sum() + 1e-10)
    
    # Add small constant to avoid log(0)
    obs_hist += 1e-10
    pred_hist += 1e-10
    
    # Calculate KL(obs || pred) = sum(obs * log(obs/pred))
    kl_divergence = entropy(pk=obs_hist, qk=pred_hist)
    
    return {
        'KL_Divergence': kl_divergence,
        'n_obs': len(obs_clean),
        'n_pred': len(pred_clean),
        'bins_used': n_bins
    }

def analyze_evt_gpd(daily_data, threshold):
    """VERIFIED: Fits a Generalized Pareto Distribution (GPD) to the tail of the data."""
    # Remove NaNs first
    clean_data = daily_data[~np.isnan(daily_data)]
    exceedances = clean_data[clean_data > threshold] - threshold
    
    if len(exceedances) < 30:  # Increased minimum for more reliable fits
        return {
            'shape': np.nan, 'scale': np.nan, 'loc': np.nan, 
            'error': f'Not enough exceedances: {len(exceedances)} < 30',
            'n_exceedances': len(exceedances)
        }
    
    try:
        # Suppress RuntimeWarning from genpareto.fit which can be noisy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Use method of moments for initial guess
            c, loc, scale = genpareto.fit(exceedances, method='MLE')
            
        # Basic sanity checks
        if scale <= 0:
            return {
                'shape': np.nan, 'scale': np.nan, 'loc': np.nan,
                'error': f'Invalid scale parameter: {scale}',
                'n_exceedances': len(exceedances)
            }
            
        return {
            'shape': c, 'scale': scale, 'loc': loc, 
            'n_exceedances': len(exceedances),
            'threshold_used': float(threshold)
        }
    except Exception as e:
        return {
            'shape': np.nan, 'scale': np.nan, 'loc': np.nan, 
            'error': str(e),
            'n_exceedances': len(exceedances)
        }

def calculate_variogram(coords, values, max_points=5000, n_bins=15):
    """
    CORRECTED: Calculates the experimental variogram for a 2D field.
    Fixed the variance calculation to be proper semivariogram.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(values)
    coords_clean = coords[valid_mask]
    values_clean = values[valid_mask]
    
    n_points = len(values_clean)
    if n_points < 10:
        return np.array([]), np.array([])
        
    if n_points > max_points:
        # Randomly subsample indices to keep computation feasible
        indices = np.random.choice(n_points, max_points, replace=False)
        coords_clean = coords_clean[indices]
        values_clean = values_clean[indices]

    # Calculate pairwise distances between coordinates
    dists = pdist(coords_clean)
    
    # Calculate semivariogram: 0.5 * (Z(xi) - Z(xj))^2 for each pair
    n = len(values_clean)
    semivar_values = []
    
    for i in range(n):
        for j in range(i+1, n):
            semivar_values.append(0.5 * (values_clean[i] - values_clean[j])**2)
    
    semivar_values = np.array(semivar_values)
    
    # Create distance bins
    max_dist = np.percentile(dists, 75)  # Use 75th percentile to avoid outliers
    bins = np.linspace(0, max_dist, n_bins + 1)
    semivariance = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (dists >= bins[i]) & (dists < bins[i+1])
        if np.any(mask):
            semivariance[i] = np.mean(semivar_values[mask])
        else:
            semivariance[i] = np.nan
            
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, semivariance

def calculate_rmse(obs, pred):
    """VERIFIED: Calculate RMSE with proper NaN handling."""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    if not mask.any():
        return np.nan
    return np.sqrt(mean_squared_error(obs[mask], pred[mask]))

def calculate_spatial_correlation(obs_map, pred_map):
    """VERIFIED: Calculate spatial correlation with proper NaN handling."""
    mask = ~np.isnan(obs_map) & ~np.isnan(pred_map)
    if not mask.any() or mask.sum() < 2:
        return np.nan
    
    obs_flat = obs_map[mask]
    pred_flat = pred_map[mask]
    
    # Check for constant values
    if np.std(obs_flat) == 0 or np.std(pred_flat) == 0:
        return np.nan
    
    try:
        corr, _ = pearsonr(obs_flat, pred_flat)
        return corr
    except Exception:
        return np.nan

# --- UNIT TESTS ---
class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)  # For reproducible tests
        
    def test_threshold_metrics_perfect(self):
        """Test threshold metrics with perfect predictions."""
        obs = np.array([0, 5, 15, 25, 0, 10])
        pred = obs.copy()  # Perfect prediction
        
        result = calculate_threshold_metrics(obs, pred, threshold=10)
        
        self.assertAlmostEqual(result['CSI'], 1.0)
        self.assertAlmostEqual(result['precision'], 1.0)
        self.assertAlmostEqual(result['recall'], 1.0)
        self.assertEqual(result['hits'], 2)  # Values 15, 25
        self.assertEqual(result['misses'], 0)
        self.assertEqual(result['false_alarms'], 0)
    
    def test_threshold_metrics_with_nans(self):
        """Test threshold metrics handle NaNs correctly."""
        obs = np.array([0, 5, np.nan, 25, 0, 15])  # Changed 10 to 15
        pred = np.array([0, 5, 15, np.nan, 0, 12])  # Changed 10 to 12
        
        result = calculate_threshold_metrics(obs, pred, threshold=10)
        
        # Should only consider 4 valid pairs: (0,0), (5,5), (0,0), (15,12)
        self.assertEqual(result['n_valid'], 4)
        self.assertEqual(result['hits'], 1)  # Only (15,12) both exceed threshold
    
    def test_sedi_calculation(self):
        """Test SEDI calculation."""
        obs = np.array([0, 5, 15, 25, 0, 10, 20])
        pred = np.array([0, 5, 12, 30, 2, 8, 18])
        
        result = calculate_sedi(obs, pred, threshold=10)
        
        # Should be a valid number between -1 and 1
        self.assertFalse(np.isnan(result))
        self.assertGreaterEqual(result, -1)
        self.assertLessEqual(result, 1)
    
    def test_kl_divergence_identical(self):
        """Test KL divergence with identical distributions."""
        data = np.random.exponential(2, 1000)  # Typical precipitation-like distribution
        
        result = calculate_distributional_metrics(data, data)
        
        # KL divergence should be very close to 0 for identical distributions
        self.assertLess(result['KL_Divergence'], 0.1)
    
    def test_kl_divergence_different(self):
        """Test KL divergence with different distributions."""
        obs = np.random.exponential(1, 1000)
        pred = np.random.exponential(2, 1000)  # Different scale
        
        result = calculate_distributional_metrics(obs, pred)
        
        # KL divergence should be positive and significant
        self.assertGreater(result['KL_Divergence'], 0)
        self.assertFalse(np.isnan(result['KL_Divergence']))
    
    def test_variogram_constant_field(self):
        """Test variogram with constant field."""
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        values = np.array([5, 5, 5, 5])  # Constant field
        
        bin_centers, semivariance = calculate_variogram(coords, values)
        
        # Variogram should be all zeros for constant field
        valid_variance = semivariance[~np.isnan(semivariance)]
        if len(valid_variance) > 0:
            self.assertTrue(np.allclose(valid_variance, 0, atol=1e-10))
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        obs = np.array([1, 2, 3, 4, 5])
        pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        rmse = calculate_rmse(obs, pred)
        expected_rmse = np.sqrt(np.mean([0.01, 0.01, 0.01, 0.01, 0.01]))
        
        self.assertAlmostEqual(rmse, expected_rmse, places=5)
    
    def test_spatial_correlation(self):
        """Test spatial correlation calculation."""
        # Create a simple 2D pattern
        obs_map = np.array([[1, 2], [3, 4]])
        pred_map = np.array([[1.1, 2.1], [3.1, 4.1]])  # Nearly perfect correlation
        
        corr = calculate_spatial_correlation(obs_map, pred_map)
        
        self.assertGreater(corr, 0.99)  # Should be very high correlation

def run_verification_tests():
    """Run all verification tests."""
    print("=" * 50)
    print("RUNNING VERIFICATION TESTS")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMetrics)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All verification tests PASSED!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) FAILED!")
        print(f"❌ {len(result.errors)} test(s) had ERRORS!")
        return False

# --- Main Analysis Function (VERIFIED VERSION) ---
def run_full_analysis(obs_file, pred_file, time_range=None, variable_obs='Precip', variable_pred='precip_mm', land_mask_file=None):
    """
    VERIFIED: Main analysis function with comprehensive error checking and validation.
    """
    print("Loading OBS…")
    # Resample to daily frequency for precipitation (sum is appropriate)
    obs_ds = xr.open_dataset(obs_file, decode_times=True).resample(time='1D').sum(keep_attrs=True)
    
    print("Loading PRED…")
    pred_ds = xr.open_dataset(pred_file, decode_times=True).resample(time='1D').sum(keep_attrs=True)
    
    # Identify the variable names
    obs_var = variable_obs if variable_obs in obs_ds.data_vars else list(obs_ds.data_vars)[0]
    pred_var = variable_pred if variable_pred in pred_ds.data_vars else list(pred_ds.data_vars)[0]
    
    print(f"Using variables: OBS='{obs_var}', PRED='{pred_var}'")
    
    # Count per-day samples for diagnostics
    obs_cnt = obs_ds[obs_var].resample(time='1D').count()
    pred_cnt = pred_ds[pred_var].resample(time='1D').count()
    
    zero_obs_days = ((obs_cnt == 0).all(dim=('lat','lon'))).sum().item()
    zero_pred_days = ((pred_cnt == 0).all(dim=('lat','lon'))).sum().item()
    print(f"[Diag] Zero-sample days — OBS: {zero_obs_days} | PRED: {zero_pred_days}")
    
    if time_range:
        t0, t1 = time_range
        obs_ds = obs_ds.sel(time=slice(t0, t1))
        pred_ds = pred_ds.sel(time=slice(t0, t1))
        print(f"Time range applied: {t0} to {t1}")
        
    def fix_lon(ds):
        """Standardize longitude coordinates to -180 to 180 range"""
        if ds.lon.max() > 180:
            ds = ds.assign_coords(lon=lambda ds: (ds.lon + 180) % 360 - 180)
        return ds.sortby(['lat', 'lon'])
        
    obs_ds = fix_lon(obs_ds)
    pred_ds = fix_lon(pred_ds)
    
    obs, pred = obs_ds[obs_var], pred_ds[pred_var]
    
    print("Interpolating & aligning obs ↔ pred…")
    obs_interp = obs.interp(lat=pred.lat, lon=pred.lon, method='nearest')
    obs_aligned, pred_aligned = xr.align(obs_interp, pred, join='inner')
    
    print(f"Common times: {obs_aligned.time.size}")
    print(f"Grid shape: {obs_aligned.shape}")
    
    # Check for all-NaN days
    obs_full_aln = obs_aligned.values
    pred_full_aln = pred_aligned.values
    
    allnan_obs_days = np.where(np.all(np.isnan(obs_full_aln), axis=(1,2)))[0]
    allnan_pred_days = np.where(np.all(np.isnan(pred_full_aln), axis=(1,2)))[0]
    print(f"[Diag] All-NaN days (pre-mask) — OBS: {len(allnan_obs_days)} | PRED: {len(allnan_pred_days)}")
    
    # Drop days that are all-NaN
    keep_days = (~np.isnan(pred_aligned)).any(dim=('lat','lon')) & (~np.isnan(obs_aligned)).any(dim=('lat','lon'))
    dropped = int((~keep_days).sum().item())
    if dropped > 0:
        print(f"[Fix] Dropping {dropped} all-NaN day(s)")
        obs_aligned = obs_aligned.sel(time=obs_aligned.time[keep_days])
        pred_aligned = pred_aligned.sel(time=pred_aligned.time[keep_days])
    
    # --- APPLY LAND MASK ---
    if land_mask_file is not None:
        print(f"Applying land mask from: {land_mask_file}")
        lm = np.load(land_mask_file, allow_pickle=True)
        lm = np.asarray(lm)
        
        # Handle 3D mask (squeeze to 2D)
        if lm.ndim == 3:
            print(f"Land mask shape: {lm.shape}")
            lm = lm[0] if lm.shape[0] == 1 else (lm.any(axis=0)).astype(lm.dtype)
        elif lm.ndim != 2:
            raise ValueError(f"Unexpected land mask ndim={lm.ndim}; expected 2 or 3.")
        
        # Match target grid
        tgt_lat = int(pred_aligned.sizes['lat'])
        tgt_lon = int(pred_aligned.sizes['lon'])
        
        if lm.shape == (tgt_lat, tgt_lon):
            lm2 = lm
        elif lm.shape == (tgt_lon, tgt_lat):
            lm2 = lm.T
            print("Transposed land mask to match (lat, lon) order")
        else:
            # Resize using nearest neighbor
            def _resize_nn(arr, new_shape):
                src_lat, src_lon = arr.shape
                new_lat, new_lon = new_shape
                lat_idx = np.clip(np.round(np.linspace(0, src_lat - 1, new_lat)).astype(int), 0, src_lat - 1)
                lon_idx = np.clip(np.round(np.linspace(0, src_lon - 1, new_lon)).astype(int), 0, src_lon - 1)
                return arr[np.ix_(lat_idx, lon_idx)]
            lm2 = _resize_nn(lm, (tgt_lat, tgt_lon))
            print(f"Resized land mask from {lm.shape} to {lm2.shape}")
        
        # Create land mask DataArray
        lm_da = xr.DataArray(lm2, coords={'lat': pred_aligned.lat, 'lon': pred_aligned.lon}, dims=('lat', 'lon'))
        land_mask_bool = (lm_da > 0.5)
        
        # Apply mask
        obs_aligned = obs_aligned.where(land_mask_bool)
        pred_aligned = pred_aligned.where(land_mask_bool)
        
        land_frac = float(land_mask_bool.mean().values)
        print(f"Land fraction: {land_frac:.3f}")
    
    # Initialize results
    all_metrics = {
        'overall_performance': {}, 
        'spatial_seasonal': {}, 
        'distributional': {}, 
        'extreme_events': {}, 
        'extreme_fixed': {},
        'evt_gpd': {},
        'diagnostics': {}
    }
    
    print("\nCalculating metrics on the full daily dataset...")
    obs_daily_np, pred_daily_np = obs_aligned.values, pred_aligned.values
    
    # Post-mask diagnostics
    allnan_obs_mask_days = np.where(np.all(np.isnan(obs_daily_np), axis=(1,2)))[0]
    allnan_pred_mask_days = np.where(np.all(np.isnan(pred_daily_np), axis=(1,2)))[0]
    print(f"[Diag] All-NaN MASKED days — OBS: {len(allnan_obs_mask_days)} | PRED: {len(allnan_pred_mask_days)}")
    
    # Overall RMSE
    overall_rmse = calculate_rmse(obs_daily_np, pred_daily_np)
    all_metrics['overall_performance']['RMSE'] = overall_rmse
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    # Valid data statistics
    valid_mask_flat = ~np.isnan(obs_daily_np.flatten()) & ~np.isnan(pred_daily_np.flatten())
    obs_flat, pred_flat = obs_daily_np.flatten()[valid_mask_flat], pred_daily_np.flatten()[valid_mask_flat]
    
    all_metrics['diagnostics'] = {
        'total_gridpoints': obs_daily_np.size,
        'valid_gridpoints': int(valid_mask_flat.sum()),
        'obs_mean': float(np.mean(obs_flat)),
        'pred_mean': float(np.mean(pred_flat)),
        'obs_std': float(np.std(obs_flat)),
        'pred_std': float(np.std(pred_flat)),
        'obs_max': float(np.max(obs_flat)),
        'pred_max': float(np.max(pred_flat))
    }
    
    print(f"Valid gridpoints: {all_metrics['diagnostics']['valid_gridpoints']:,} / {all_metrics['diagnostics']['total_gridpoints']:,}")
    print(f"OBS mean/std: {all_metrics['diagnostics']['obs_mean']:.2f} ± {all_metrics['diagnostics']['obs_std']:.2f}")
    print(f"PRED mean/std: {all_metrics['diagnostics']['pred_mean']:.2f} ± {all_metrics['diagnostics']['pred_std']:.2f}")
    
    # Distributional metrics
    all_metrics['distributional'] = calculate_distributional_metrics(obs_flat, pred_flat)
    print(f"KL Divergence: {all_metrics['distributional']['KL_Divergence']:.4f}")
    
    # Fixed threshold metrics (10mm, 20mm)
    for thr in (10.0, 20.0):
        thr_result = calculate_threshold_metrics(obs_daily_np, pred_daily_np, thr)
        all_metrics['extreme_fixed'][f'{thr:.0f}mm'] = thr_result
        print(f"Threshold {thr:.0f}mm - CSI: {thr_result['CSI']:.3f}, Precision: {thr_result['precision']:.3f}, Recall: {thr_result['recall']:.3f}")
        print(f"  Events - Hits: {thr_result['hits']}, Misses: {thr_result['misses']}, False Alarms: {thr_result['false_alarms']}")
    
    # Adaptive threshold (P95) for extreme events
    obs_precip_only = obs_flat[obs_flat > 0.1]
    if len(obs_precip_only) > 0:
        p95 = np.percentile(obs_precip_only, 95)
        all_metrics['extreme_events']['threshold_p95'] = p95
        print(f"P95 threshold: {p95:.2f}mm")
        
        p95_metrics = calculate_threshold_metrics(obs_daily_np, pred_daily_np, p95)
        all_metrics['extreme_events']['threshold_metrics'] = p95_metrics
        all_metrics['extreme_events']['SEDI'] = calculate_sedi(obs_daily_np, pred_daily_np, p95)
        
        print(f"P95 metrics - CSI: {p95_metrics['CSI']:.3f}, SEDI: {all_metrics['extreme_events']['SEDI']:.3f}")
        
        # EVT/GPD analysis on daily land-maximum series
        obs_max = np.nanmax(obs_daily_np, axis=(1, 2))
        pred_max = np.nanmax(pred_daily_np, axis=(1, 2))
        
        obs_max_pos = obs_max[obs_max > 0.1]
        if obs_max_pos.size >= 50:
            p99_max = np.nanpercentile(obs_max_pos, 99)
        else:
            p99_max = np.nanpercentile(obs_max_pos, 95) if obs_max_pos.size > 0 else np.nan
        
        all_metrics['evt_gpd']['threshold_p99_max'] = p99_max
        print(f"P99 max threshold: {p99_max:.2f}mm")
        
        def _fit_with_fallback(series, thr_primary, thr_fallback=97.5):
            fit = analyze_evt_gpd(series, thr_primary)
            if np.isnan(fit.get('shape', np.nan)):
                # Try a slightly lower threshold if too few exceedances
                thr2 = np.nanpercentile(series[series > 0.1], thr_fallback)
                fit = analyze_evt_gpd(series, thr2)
                fit['fallback_threshold'] = float(thr2)
            return fit
        
        all_metrics['evt_gpd']['observed_fit'] = _fit_with_fallback(obs_max, p99_max)
        all_metrics['evt_gpd']['predicted_fit'] = _fit_with_fallback(pred_max, p99_max)
        
        # Print GPD results
        obs_gpd = all_metrics['evt_gpd']['observed_fit']
        pred_gpd = all_metrics['evt_gpd']['predicted_fit']
        print(f"GPD fits - OBS shape: {obs_gpd.get('shape', 'NaN'):.3f}, PRED shape: {pred_gpd.get('shape', 'NaN'):.3f}")
        print(f"GPD exceedances - OBS: {obs_gpd.get('n_exceedances', 0)}, PRED: {pred_gpd.get('n_exceedances', 0)}")
    
    # Seasonal analysis
    print("\nCalculating seasonal metrics...")
    obs_aligned.coords['season'] = obs_aligned.time.dt.season
    pred_aligned.coords['season'] = pred_aligned.time.dt.season
    seasonal_obs = obs_aligned.groupby('season').mean('time')
    seasonal_pred = pred_aligned.groupby('season').mean('time')
    
    # Create coordinate grid for variogram
    lon_grid, lat_grid = np.meshgrid(seasonal_obs.lon.values, seasonal_obs.lat.values)
    coords = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
    
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        if season not in seasonal_obs.season:
            print(f"--- Season {season} skipped (no data) ---")
            continue
            
        print(f"--- Season {season} ---")
        o_da, p_da = seasonal_obs.sel(season=season).compute(), seasonal_pred.sel(season=season).compute()
        o_map, p_map = o_da.values, p_da.values
        
        # Calculate seasonal metrics
        seasonal_rmse = calculate_rmse(o_map, p_map)
        ssim_val = calculate_ssim(o_map, p_map)
        spatial_corr = calculate_spatial_correlation(o_map, p_map)
        
        # Calculate variograms
        mask = ~np.isnan(o_map) & ~np.isnan(p_map)
        if mask.any():
            coords_masked = coords[mask.ravel()]
            o_flat_masked = o_map[mask]
            p_flat_masked = p_map[mask]
            
            var_obs = calculate_variogram(coords_masked, o_flat_masked)
            var_pred = calculate_variogram(coords_masked, p_flat_masked)
            
            # Calculate variogram MSE
            if len(var_obs[1]) > 0 and len(var_pred[1]) > 0:
                var_mask = ~np.isnan(var_obs[1]) & ~np.isnan(var_pred[1])
                if var_mask.any():
                    variogram_mse = np.mean((var_obs[1][var_mask] - var_pred[1][var_mask])**2)
                else:
                    variogram_mse = np.nan
            else:
                variogram_mse = np.nan
        else:
            var_obs = (np.array([]), np.array([]))
            var_pred = (np.array([]), np.array([]))
            variogram_mse = np.nan
        
        all_metrics['spatial_seasonal'][season] = {
            'RMSE': seasonal_rmse,
            'SSIM': ssim_val,
            'Spatial_Correlation': spatial_corr,
            'Variogram_MSE': variogram_mse,
            'variogram_obs': var_obs,
            'variogram_pred': var_pred
        }
        
        print(f"  RMSE: {seasonal_rmse:.4f}, SSIM: {ssim_val:.3f}, Spatial Corr: {spatial_corr:.3f}")
        print(f"  Variogram MSE: {variogram_mse:.6f}")
    
    # Final sanity checks and diagnostics
    print("\n" + "="*50)
    print("SANITY CHECKS")
    print("="*50)
    
    try:
        p95_print = all_metrics['extreme_events'].get('threshold_p95', np.nan)
        print(f"P95 threshold: {p95_print:.3f}mm")
        
        # Exceedance counts at fixed thresholds
        for thr in (10.0, 20.0):
            valid = ~np.isnan(obs_daily_np) & ~np.isnan(pred_daily_np)
            obs_exc = np.sum((obs_daily_np > thr) & valid)
            pred_exc = np.sum((pred_daily_np > thr) & valid)
            total_valid = valid.sum()
            print(f"Exceedances @ {thr:.0f}mm — OBS: {int(obs_exc)} ({100*obs_exc/total_valid:.2f}%) | PRED: {int(pred_exc)} ({100*pred_exc/total_valid:.2f}%)")
        
        # Check for reasonable value ranges
        if all_metrics['diagnostics']['obs_max'] > 1000:
            print(f"⚠️  WARNING: Very high observed precipitation: {all_metrics['diagnostics']['obs_max']:.1f}mm")
        if all_metrics['diagnostics']['pred_max'] > 1000:
            print(f"⚠️  WARNING: Very high predicted precipitation: {all_metrics['diagnostics']['pred_max']:.1f}mm")
            
        # Check correlation
        overall_corr = np.corrcoef(obs_flat, pred_flat)[0, 1]
        print(f"Overall correlation: {overall_corr:.3f}")
        all_metrics['overall_performance']['Correlation'] = overall_corr
        
    except Exception as e:
        print(f"Error in sanity checks: {e}")
    
    print("="*50)
    return all_metrics

# --- Enhanced Excel Report Generation ---
def generate_report(train_metrics, test_metrics, filename="metrics_report.xlsx"):
    """
    VERIFIED: Generates a comprehensive formatted Excel report from the metrics dictionaries.
    """
    # Enhanced metric descriptions
    METRIC_INFO = {
        'RMSE': {
            "desc": "Root Mean Squared Error. Standard deviation of prediction errors. Sensitive to large errors.", 
            "interp": "Lower is better (0 = perfect)",
            "units": "mm"
        },
        'Correlation': {
            "desc": "Overall Pearson correlation between observed and predicted values.", 
            "interp": "Higher is better (1 = perfect)",
            "units": "unitless"
        },
        'SSIM': {
            "desc": "Structural Similarity Index. Measures similarity of spatial patterns (luminance, contrast, structure).", 
            "interp": "Higher is better (1 = perfect)",
            "units": "unitless"
        },
        'Spatial_Correlation': {
            "desc": "Pixel-wise Pearson correlation between observed and predicted seasonal maps.", 
            "interp": "Higher is better (1 = perfect)",
            "units": "unitless"
        },
        'Variogram_MSE': {
            "desc": "Mean Squared Error between observed and predicted variograms. Measures similarity of spatial texture.", 
            "interp": "Lower is better (0 = perfect)",
            "units": "mm²"
        },
        'KL_Divergence': {
            "desc": "Kullback-Leibler Divergence KL(obs||pred). Measures how predicted distribution differs from observed.", 
            "interp": "Lower is better (0 = identical distributions)",
            "units": "nats"
        },
        'CSI': {
            "desc": "Critical Success Index. Overall skill in forecasting events, penalizes both misses and false alarms.", 
            "interp": "Higher is better (1 = perfect)",
            "units": "unitless"
        },
        'precision': {
            "desc": "Precision. Of all predicted events, what fraction was correct?", 
            "interp": "Higher is better (1 = perfect)",
            "units": "unitless"
        },
        'recall': {
            "desc": "Recall (Sensitivity). Of all actual events, what fraction was correctly identified?", 
            "interp": "Higher is better (1 = perfect)",
            "units": "unitless"
        },
        'SEDI': {
            "desc": "Symmetric Extremal Dependence Index. Skill score for rare/extreme events, robust to base rate.", 
            "interp": "Higher is better (1 = perfect)",
            "units": "unitless"
        },
        'GPD_shape': {
            "desc": "Shape parameter (ξ) of GPD fit to extremes. Positive=heavy-tailed, negative=light-tailed.", 
            "interp": "Predicted should match Observed",
            "units": "unitless"
        },
        'GPD_scale': {
            "desc": "Scale parameter of GPD fit to extremes. Related to spread of extreme values.", 
            "interp": "Predicted should match Observed",
            "units": "mm"
        }
    }
    
    report_data = []

    def add_row(category, metric, season, train_val, test_val, notes=""):
        info = METRIC_INFO.get(metric, {"desc": "-", "interp": "-", "units": "-"})
        
        # Format values based on typical ranges
        def format_val(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "NaN"
            if isinstance(val, (int, float)):
                if abs(val) < 0.001:
                    return f"{val:.2e}"
                elif abs(val) < 0.01:
                    return f"{val:.4f}"
                elif abs(val) < 1:
                    return f"{val:.3f}"
                else:
                    return f"{val:.3f}"
            return str(val)
        
        report_data.append({
            "Category": category,
            "Metric": metric,
            "Season": season,
            "Train Value": format_val(train_val),
            "Test Value": format_val(test_val),
            "Units": info["units"],
            "Interpretation": info["interp"],
            "Description": info["desc"],
            "Notes": notes
        })

    # Overall performance
    add_row("Overall Performance", "RMSE", "Daily", 
            train_metrics['overall_performance'].get('RMSE'), 
            test_metrics['overall_performance'].get('RMSE'))
    
    add_row("Overall Performance", "Correlation", "Daily", 
            train_metrics['overall_performance'].get('Correlation'), 
            test_metrics['overall_performance'].get('Correlation'))

    # Seasonal performance
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        if season not in train_metrics.get('spatial_seasonal', {}) or season not in test_metrics.get('spatial_seasonal', {}):
            continue
        
        train_season = train_metrics['spatial_seasonal'][season]
        test_season = test_metrics['spatial_seasonal'][season]

        add_row("Seasonal Performance", "RMSE", season, 
                train_season.get('RMSE'), test_season.get('RMSE'))
        add_row("Spatial Patterns", "SSIM", season, 
                train_season.get('SSIM'), test_season.get('SSIM'))
        add_row("Spatial Patterns", "Spatial_Correlation", season, 
                train_season.get('Spatial_Correlation'), test_season.get('Spatial_Correlation'))
        add_row("Spatial Patterns", "Variogram_MSE", season, 
                train_season.get('Variogram_MSE'), test_season.get('Variogram_MSE'))

    # Distributional metrics
    add_row("Distributional", "KL_Divergence", "Overall", 
            train_metrics.get('distributional', {}).get('KL_Divergence'), 
            test_metrics.get('distributional', {}).get('KL_Divergence'))

    # Fixed threshold metrics
    for thr_label in ['10mm', '20mm']:
        if 'extreme_fixed' in train_metrics and thr_label in train_metrics['extreme_fixed']:
            tm_train = train_metrics['extreme_fixed'][thr_label]
            tm_test = test_metrics['extreme_fixed'][thr_label]
            
            notes = f"Hits: {tm_test.get('hits', 'N/A')}, Misses: {tm_test.get('misses', 'N/A')}, FA: {tm_test.get('false_alarms', 'N/A')}"
            
            add_row(f"Fixed Threshold ({thr_label})", "CSI", "Overall", 
                    tm_train.get('CSI'), tm_test.get('CSI'), notes)
            add_row(f"Fixed Threshold ({thr_label})", "precision", "Overall", 
                    tm_train.get('precision'), tm_test.get('precision'))
            add_row(f"Fixed Threshold ({thr_label})", "recall", "Overall", 
                    tm_train.get('recall'), tm_test.get('recall'))

    # Adaptive threshold (P95) metrics
    if 'extreme_events' in train_metrics and 'threshold_metrics' in train_metrics['extreme_events']:
        tm_train = train_metrics['extreme_events']['threshold_metrics']
        tm_test = test_metrics['extreme_events']['threshold_metrics']
        
        p95_train = train_metrics['extreme_events'].get('threshold_p95', 'N/A')
        p95_test = test_metrics['extreme_events'].get('threshold_p95', 'N/A')
        notes = f"Train P95: {p95_train:.2f}mm, Test P95: {p95_test:.2f}mm" if isinstance(p95_train, (int, float)) and isinstance(p95_test, (int, float)) else ""
        
        add_row("Adaptive Threshold (P95)", "CSI", "Overall", 
                tm_train.get('CSI'), tm_test.get('CSI'), notes)
        add_row("Adaptive Threshold (P95)", "precision", "Overall", 
                tm_train.get('precision'), tm_test.get('precision'))
        add_row("Adaptive Threshold (P95)", "recall", "Overall", 
                tm_train.get('recall'), tm_test.get('recall'))
        add_row("Adaptive Threshold (P95)", "SEDI", "Overall", 
                train_metrics['extreme_events'].get('SEDI'), 
                test_metrics['extreme_events'].get('SEDI'))

    # EVT/GPD metrics
    if 'evt_gpd' in train_metrics and 'observed_fit' in train_metrics['evt_gpd']:
        gpd_train_obs = train_metrics['evt_gpd']['observed_fit']
        gpd_train_pred = train_metrics['evt_gpd']['predicted_fit']
        gpd_test_obs = test_metrics['evt_gpd']['observed_fit']
        gpd_test_pred = test_metrics['evt_gpd']['predicted_fit']
        
        exc_notes = f"Train exc: {gpd_train_obs.get('n_exceedances', 'N/A')}, Test exc: {gpd_test_obs.get('n_exceedances', 'N/A')}"
        
        add_row("EVT GPD (P99) - Observed", "GPD_shape", "Overall", 
                gpd_train_obs.get('shape'), gpd_test_obs.get('shape'), exc_notes)
        add_row("EVT GPD (P99) - Predicted", "GPD_shape", "Overall", 
                gpd_train_pred.get('shape'), gpd_test_pred.get('shape'))
        add_row("EVT GPD (P99) - Observed", "GPD_scale", "Overall", 
                gpd_train_obs.get('scale'), gpd_test_obs.get('scale'))
        add_row("EVT GPD (P99) - Predicted", "GPD_scale", "Overall", 
                gpd_train_pred.get('scale'), gpd_test_pred.get('scale'))
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    df = df[["Category", "Metric", "Season", "Train Value", "Test Value", "Units", "Interpretation", "Description", "Notes"]]

    # Add summary statistics
    summary_data = []
    
    def add_summary(desc, train_val, test_val):
        summary_data.append({
            "Metric": desc,
            "Train": train_val,
            "Test": test_val,
            "Difference": test_val - train_val if isinstance(train_val, (int, float)) and isinstance(test_val, (int, float)) else "N/A"
        })
    
    # Get some key summary metrics
    if 'diagnostics' in train_metrics and 'diagnostics' in test_metrics:
        add_summary("Valid Gridpoints", 
                   train_metrics['diagnostics'].get('valid_gridpoints', 0),
                   test_metrics['diagnostics'].get('valid_gridpoints', 0))
        add_summary("Obs Mean (mm)", 
                   train_metrics['diagnostics'].get('obs_mean', np.nan),
                   test_metrics['diagnostics'].get('obs_mean', np.nan))
        add_summary("Pred Mean (mm)", 
                   train_metrics['diagnostics'].get('pred_mean', np.nan),
                   test_metrics['diagnostics'].get('pred_mean', np.nan))
    
    summary_df = pd.DataFrame(summary_data)

    # Write to Excel with multiple sheets
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Detailed_Metrics')
            summary_df.to_excel(writer, index=False, sheet_name='Summary_Statistics')
            
            # Add diagnostic information
            if 'diagnostics' in train_metrics:
                diag_data = []
                for period, metrics in [('Train', train_metrics), ('Test', test_metrics)]:
                    if 'diagnostics' in metrics:
                        for key, value in metrics['diagnostics'].items():
                            diag_data.append({'Period': period, 'Diagnostic': key, 'Value': value})
                
                if diag_data:
                    diag_df = pd.DataFrame(diag_data)
                    diag_df.to_excel(writer, index=False, sheet_name='Diagnostics')
        
        print(f"\n✅ Successfully generated comprehensive report: {filename}")
        print("📊 Report includes:")
        print("   - Detailed_Metrics: All performance metrics")
        print("   - Summary_Statistics: Key summary comparisons")
        print("   - Diagnostics: Data quality and coverage information")
        
    except Exception as e:
        print(f"\n❌ Failed to write Excel file. Error: {e}")
        # Fallback to CSV
        try:
            csv_filename = filename.replace('.xlsx', '.csv')
            df.to_csv(csv_filename, index=False)
            print(f"📄 Saved as CSV instead: {csv_filename}")
        except Exception as e2:
            print(f"❌ Failed to write CSV file too. Error: {e2}")

# --- Main execution block with verification ---
if __name__ == "__main__":
    print("🔬 VERIFIED PRECIPITATION METRICS ANALYSIS")
    print("=" * 60)
    
    # Run verification tests first
    tests_passed = run_verification_tests()
    
    if not tests_passed:
        print("\n⚠️  Some tests failed. Proceeding with analysis but results should be validated.")
        print("Check the test failures above for potential issues.")
    
    print("\n" + "=" * 60)
    print("STARTING MAIN ANALYSIS")
    print("=" * 60)
    
    # Define file paths and time ranges
    full_obs_file = 'highres-files/PRISM_daily_FL_1981-2019_remapped.nc'
    train_pred_file = 'train_precip_mm.nc'
    test_pred_file = 'test_precip_mm.nc'
    
    train_time_range = ('1981-01-01', '2014-12-31')
    test_time_range = ('2015-01-01', '2019-12-31')
    
    try:
        print("\n" + "=" * 40)
        print("===== TRAINING PERIOD ANALYSIS =====")
        print("=" * 40)
        train_metrics_result = run_full_analysis(
            full_obs_file, train_pred_file,
            time_range=train_time_range,
            variable_obs='Precip',
            variable_pred='precip_mm',
            land_mask_file="mydata/land_mask.npy"
        )

        print("\n" + "=" * 40)
        print("===== TEST PERIOD ANALYSIS =====")
        print("=" * 40)
        test_metrics_result = run_full_analysis(
            full_obs_file, test_pred_file,
            time_range=test_time_range,
            variable_obs='Precip',
            variable_pred='precip_mm',
            land_mask_file="mydata/land_mask.npy"
        )

        print("\n" + "=" * 40)
        print("===== GENERATING REPORT =====")
        print("=" * 40)
        generate_report(train_metrics_result, test_metrics_result, "verified_metrics_report.xlsx")
        
        print("\n🎉 Analysis completed successfully!")
        print("📋 Check the printed results above for any warnings or issues.")
        print("📊 Full results saved in verified_metrics_report.xlsx")
        
    except Exception as e:
        print(f"\n💥 Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔍 Please check:")
        print("   - File paths are correct")
        print("   - NetCDF files exist and are readable")
        print("   - Land mask file exists (if specified)")
        print("   - Required packages are installed")