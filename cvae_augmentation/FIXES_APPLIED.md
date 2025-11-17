# cVAE Training Fixes - Summary

## Problems Identified

### 1. **Extreme Loss Volatility** ⚠️ CRITICAL
- **Issue**: `lambda_ext: 10.0` caused validation loss to spike 3× between epochs
- **Root Cause**: Heavy precipitation is rare (1.3% of samples), so batches with different amounts of extreme pixels had wildly different losses
- **Evidence**: Epoch 5 val_L_rec=2.1301 → Epoch 22 val_L_rec=6.5455

### 2. **KL Warm-up Never Completed**
- **Issue**: `warmup_epochs: 30` but training stopped at epoch 25
- **Impact**: Best model (epoch 5) had beta≈0.017, meaning KL regularization was essentially disabled
- **Consequence**: Poor latent space structure, potential posterior collapse

### 3. **Unstable Early Stopping Metric**
- **Issue**: Monitoring `val_L_rec` (includes volatile extreme loss)
- **Impact**: Early stopping triggered by loss spikes rather than true lack of improvement

### 4. **Model Overfitting Risk**
- **Stats**: 152M parameters / 9,928 samples = 15,400:1 ratio
- **Heavy samples**: Only 186 extreme events (1.3%)
- **Risk**: Severe overfitting, especially for rare extreme events

### 5. **Learning Rate Too Conservative**
- **Issue**: lr=0.00005 was very slow
- **Impact**: Model not converging fast enough before early stopping

---

## Fixes Applied ✅

### 1. **Stabilized Extreme Loss Weighting**

**File**: `losses.py:15-67`

**Changes**:
- Reduced `lambda_ext` from **10.0 → 2.0** to reduce volatility
- Added **per-sample normalization** for extreme MAE
- Added **extreme_sample_ratio scaling** to prevent dominance when few samples have extremes

**Impact**:
- Loss will be much more stable across batches
- Extreme events still emphasized (2× weight) without causing instability

```python
# Old: Simple averaging caused huge variance
mae_ext = diff_ext.sum() / n_ext

# New: Per-sample averaging + ratio scaling
extreme_sample_ratio = len(mae_ext_per_sample) / batch_size
loss = lambda_base * mae_base + lambda_ext * mae_ext * extreme_sample_ratio
```

### 2. **Fixed KL Divergence Parameters**

**File**: `config.yaml:43-44`

**Changes**:
- Reduced `beta_kl` from **0.1 → 0.01** to prevent posterior collapse
- Reduced `warmup_epochs` from **30 → 15** to complete before early stopping
- Increased `lambda_mass` from **0.001 → 0.01** for better mass conservation

**Impact**:
- KL warm-up will complete by epoch 15
- Lighter KL weight prevents the model from ignoring latent space
- Better physical constraint (mass conservation)

### 3. **Stable Early Stopping Metric**

**File**: `train_cvae.py:342-343, 391-402`

**Changes**:
- Added configurable `early_stopping_metric` in config
- Set to **`MAE_all`** (stable) instead of `L_rec` (volatile)
- Metric is printed and tracked clearly

**Impact**:
- Early stopping based on reconstruction quality, not loss spikes
- More reliable convergence detection

### 4. **Improved Training Configuration**

**File**: `config.yaml:50-59`

**Changes**:
- Increased `lr` from **0.00005 → 0.0001** (2× faster learning)
- Increased `epochs` from **80 → 100** (more time to converge)
- Increased `early_stopping_patience` from **20 → 25** (more tolerance)

**Impact**:
- Faster convergence
- More opportunity to find good solution
- Less likely to stop due to temporary plateaus

---

## Expected Improvements 📈

### Training Stability
- **Before**: val_L_rec varied 2.1 → 6.5 (3× range)
- **After**: Should see <20% variation between epochs

### Convergence
- **Before**: Best model at epoch 5, stopped at 25
- **After**: Should converge around epoch 20-30 with stable improvement

### Latent Space Quality
- **Before**: beta=0.017 at best model (almost no regularization)
- **After**: beta will reach 0.01 by epoch 15 (proper regularization)

### Extreme Event Modeling
- **Before**: Loss dominated by extreme pixels → underfitting moderate rain
- **After**: Balanced focus on all precipitation levels + emphasis on extremes

---

## Additional Recommendations (Not Yet Applied)

### 1. **Reduce Model Complexity** 🔧
Currently: 152M parameters for ~10k samples is excessive

**Suggested change in `config.yaml`**:
```yaml
model:
  base_filters: 64  # Change from 96 → 64
  d_y: 256          # Change from 384 → 256
  d_z: 64           # Change from 96 → 64
```

**Impact**:
- Reduce parameters to ~60M (60% reduction)
- Less overfitting
- Faster training
- Lower memory usage

### 2. **Add Dropout for Regularization** 🔧

**Suggested change in `model_cvae.py`**:
- Increase dropout from 0.1 → 0.2 in encoders
- Add dropout in decoder as well

### 3. **Data Augmentation** 🔧
Since extreme events are rare (186 samples), consider:
- Random spatial flips/rotations
- Add small noise to inputs
- Mixup between samples

### 4. **Monitor More Metrics** 📊
Add to validation logging:
- Extreme pixels RMSE (not just MAE)
- Correlation coefficient
- Frequency of extreme predictions vs. ground truth

---

## How to Retrain 🚀

1. **Backup your old model**:
   ```bash
   cp -r outputs/cvae outputs/cvae_old
   ```

2. **Clear old checkpoints** (optional):
   ```bash
   rm -rf outputs/cvae/checkpoints/*
   rm -rf outputs/cvae/logs/*
   ```

3. **Retrain with new configuration**:
   ```bash
   cd cvae_augmentation
   python train_cvae.py --config config.yaml
   ```

4. **Monitor training**:
   ```bash
   tail -f outputs/cvae/logs/train_log.csv
   ```

5. **Check for improvements**:
   - val_MAE_all should decrease steadily
   - val_MAE_tail should improve (this is your extreme rain skill)
   - Training should run 30-50 epochs before early stopping

---

## Troubleshooting 🔍

### If training still stops early (<30 epochs):
- Increase `early_stopping_patience` to 30-40
- Check if MAE_all is actually improving (plot the log file)

### If loss is still volatile:
- Reduce `lambda_ext` further to 1.0
- Increase batch_size to 128 (if GPU memory allows)

### If extreme events are still poorly modeled:
- Try `early_stopping_metric: "MAE_tail"` to directly optimize for extremes
- Oversample heavy precipitation days during training

### If model overfits:
- Apply the model complexity reduction (base_filters: 64)
- Increase weight_decay to 0.001
- Add more dropout (0.2 or 0.3)

---

## Summary of Changes

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|---------|
| `lambda_ext` | 10.0 | 2.0 | Reduce volatility |
| `beta_kl` | 0.1 | 0.01 | Prevent collapse |
| `warmup_epochs` | 30 | 15 | Complete before stopping |
| `lambda_mass` | 0.001 | 0.01 | Better conservation |
| `lr` | 0.00005 | 0.0001 | Faster convergence |
| `epochs` | 80 | 100 | More training time |
| `early_stopping_patience` | 20 | 25 | More tolerance |
| `early_stopping_metric` | L_rec | MAE_all | Stable metric |
| Loss computation | Batch-level | Per-sample + ratio | Reduce variance |

**Expected Outcome**: Training should run 30-50 epochs, show stable improvement, and produce a model with much better extreme precipitation skill!
