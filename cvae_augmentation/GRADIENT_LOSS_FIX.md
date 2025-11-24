# cVAE "Smooth Background" Fix - Adding Gradient Loss

## 🔴 New Problem After Mean Collapse Fix

After fixing the mean collapse problem, we got **partial improvement** but still have issues:

### Results After Mean Collapse Fix:
```
✓ Spatial correlation: 0.19 (positive, but still LOW)
✗ Wet fraction: 100% (STILL 100%!)
✗ Mass conservation: -0.82 (poor)
✗ Sample diversity: 0.014 (very low)
```

**Visual**: Instead of "average rain everywhere" (mean collapse), now we have **"light drizzle everywhere"** (smooth background).

---

## 🔍 Root Cause: Missing Gradient Loss

### The Problem:

The model is predicting a **smooth layer of 0.6-1.5 mm/day everywhere** instead of concentrated storms.

**Why pixel threshold didn't work**:
```
Model prediction: 0.8 mm/day everywhere (smooth background)
Pixel threshold:  0.5 mm/day (from config)
Result:          0.8 > 0.5 → KEEP EVERYTHING
Wet fraction:    100% (all pixels > 0.1 counted as "wet")
```

### Why This Happens:

Looking at `losses.py`, we found:
- **`gradient_loss` function doesn't exist!**
- Config has `lambda_grad: 1.0` but it's not connected to anything
- Loss function only has: reconstruction + mass + KL
- **No penalty for smooth predictions**

**Result**: Model spreads rain out like "thin layer of jam" across entire map instead of concentrated "peaks and valleys".

---

## ✅ Complete Fix: Add Gradient (Edge) Loss

### What is Gradient Loss?

Gradient loss penalizes **smooth predictions** and rewards **sharp edges**:

```python
def gradient_loss(y_true, y_hat):
    """
    Computes L1 difference between spatial gradients.

    Forces model to create SHARP transitions:
    - "This pixel is WET (50mm), neighbor is DRY (0mm)" ✅
    - "This pixel is 0.8mm, neighbor is 0.7mm" ❌ (smooth, gets penalized)
    """
    # Horizontal edges (left-right)
    dy_true = abs(y_true[:, :, :, 1:] - y_true[:, :, :, :-1])
    dy_hat = abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1])

    # Vertical edges (top-bottom)
    dx_true = abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :])
    dx_hat = abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :])

    return mean(abs(dy_true - dy_hat)) + mean(abs(dx_true - dx_hat))
```

**Effect**: Model learns to create **concentrated storms with dry areas**, not smooth drizzle.

---

## 📝 Changes Made

### 1. `losses.py` - Added Gradient Loss

**Added** (lines 197-228):
```python
def gradient_loss(y_true, y_hat):
    """Forces sharp wet/dry transitions."""
    # Horizontal gradients
    dy_true = torch.abs(y_true[:, :, :, 1:] - y_true[:, :, :, :-1])
    dy_hat = torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1])

    # Vertical gradients
    dx_true = torch.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :])
    dx_hat = torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :])

    return torch.mean(torch.abs(dy_true - dy_hat)) + torch.mean(torch.abs(dx_true - dx_hat))
```

**Updated** `SimplifiedCVAELoss.__init__`:
```python
def __init__(self, ..., lambda_grad=1.0, ...):  # Added lambda_grad parameter
    self.lambda_grad = lambda_grad
```

**Updated** `SimplifiedCVAELoss.forward`:
```python
# 3. Gradient (edge) loss - NEW!
grad_loss = gradient_loss(y_true, y_hat)

# Total loss
total_loss = rec_loss + lambda_mass * m_loss + lambda_grad * grad_loss + beta_kl * kl_loss
```

### 2. `config.yaml` - Aggressive Edge Preservation

**Updated** loss weights:
```yaml
loss:
  w_min: 0.5          # INCREASED from 0.2 (penalize background noise more)
  lambda_grad: 2.0    # INCREASED from 1.0 (strong edge preservation)
```

**Updated** pixel threshold:
```yaml
sampling:
  pixel_threshold: 2.0  # INCREASED from 0.5 → 2.0 mm/day
                        # Meteorological standard: < 1-2 mm/day is "trace"
```

---

## 📊 Expected Improvements

| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| **Spatial Correlation** | 0.19 | **>0.60** | ✅ Sharp storm structure |
| **Wet Fraction** | 100% | **~70-80%** | ✅ Realistic dry areas |
| **Mass Conservation** | -0.82 | **±0.20** | ✅ Better preservation |
| **Sample Diversity** | 0.014 | **0.15-0.25** | ✅ Better variation |

### Visual Changes:

**Before (smooth background)**:
- Uniform 0.8 mm/day everywhere
- No dry areas
- Blurry, no structure

**After (sharp edges)**:
- Concentrated storms 50-200 mm/day
- Clear dry regions (0 mm/day)
- Sharp wet/dry transitions

---

## 🚀 How to Apply

### Step 1: Pull Latest Code

```bash
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation
git pull origin claude/review-srdrn-preprocessing-01EAAjCDyGDVyYztrGttJP1d
```

### Step 2: RETRAIN (Required!)

```bash
# Backup old model
cp outputs/cvae_final/checkpoints/cvae_best.pt outputs/cvae_final/checkpoints/cvae_best_pre_gradient_loss.pt

# Retrain with gradient loss
python train_cvae.py --config config.yaml
```

**Why retrain?**: Loss function changed - model needs to learn with edge preservation.

**Expected time**: ~1-2 hours

**What to watch**: Training logs should show `L_grad` decreasing over epochs.

### Step 3: Sample & Evaluate

```bash
# Clean old samples
rm -rf outputs/cvae_final/synth/*

# Generate new samples (threshold=2.0 will be applied)
python sample_cvae.py --config config.yaml --mode posterior

# Evaluate
python evaluate_samples.py --config config.yaml

# Visualize
python visualize_samples.py --config config.yaml --num_samples 20
```

---

## 🎯 Verification Checklist

After retraining and sampling:

### Must Pass:
- [ ] **Training logs show `L_grad`**: Gradient loss should be logged and decreasing
- [ ] **Wet fraction < 85%**: If still 100%, gradient loss isn't working
- [ ] **Spatial correlation > 0.50**: Sharp structures should improve correlation
- [ ] **Visual check**: See concentrated storms with clear dry areas

### Check These:
```bash
# Training log should show L_grad
tail -20 outputs/cvae_final/logs/train_log.csv | grep L_grad

# Wet fraction in statistics
cat outputs/cvae_final/visualizations/statistics.json | grep wet_fraction
# Should be: "wet_fraction": 0.70-0.80

# Look at visualizations
ls outputs/cvae_final/visualizations/*.png
```

---

## 🔧 Troubleshooting

### If wet fraction still > 90%:

**Possible causes**:
1. Gradient loss not being computed (check logs for `L_grad`)
2. lambda_grad too low
3. Pixel threshold too low

**Fixes**:
```yaml
# Increase gradient loss weight
loss:
  lambda_grad: 3.0  # Increase from 2.0

# Increase pixel threshold
sampling:
  pixel_threshold: 3.0  # Increase from 2.0
```

### If spatial correlation still low (< 0.4):

**Possible cause**: Gradient loss too aggressive, destroying structure

**Fix**:
```yaml
loss:
  lambda_grad: 1.0  # Reduce from 2.0
  w_min: 0.3        # Reduce from 0.5
```

### If training is unstable (NaN losses):

**Possible cause**: Gradient loss too strong

**Fix**:
```yaml
loss:
  lambda_grad: 0.5  # Start lower
```

Then gradually increase if stable.

---

## 📚 Technical Notes

### Why Gradient Loss Works:

Real precipitation has **sharp boundaries**:
- Inside storm: 50-200 mm/day
- Outside storm: 0-1 mm/day
- **Transition**: Very sharp (few kilometers)

Gradient loss measures these transitions:
- Large gradient in real data → Model should also have large gradient
- Small gradient everywhere (smooth) → Large penalty

### Why Pixel Threshold of 2.0?

- **Meteorological standard**: < 1-2 mm/day often classified as "trace precipitation"
- **Measurement noise**: Rain gauges unreliable below this
- **Visual clarity**: 2.0 mm/day threshold creates clear dry regions

### Comparison with Other Methods:

| Method | Effect | Pros | Cons |
|--------|--------|------|------|
| **Sparsity Loss (L1)** | Encourages zeros | Simple | Doesn't preserve edges |
| **Adversarial Loss (GAN)** | Very sharp | Best quality | Unstable, mode collapse |
| **Gradient Loss (Ours)** | Sharp edges | Stable, preserves structure | Needs tuning |

---

## ✨ Summary

**Problem**: Model predicted smooth 0.8mm drizzle everywhere (wet fraction 100%)

**Root cause**: No gradient_loss function - no penalty for smooth predictions

**Fix**:
1. **Added** `gradient_loss()` function to `losses.py`
2. **Integrated** into `SimplifiedCVAELoss` class
3. **Increased** `lambda_grad` from 1.0 → 2.0
4. **Increased** `w_min` from 0.2 → 0.5
5. **Increased** `pixel_threshold` from 0.5 → 2.0 mm/day

**Expected result**: Sharp storms with clear dry areas, wet fraction ~70-80%

**Action required**: RETRAIN model with new loss function

**Time**: ~2-3 hours total

---

**Files Modified**:
- `losses.py` - Added gradient_loss (+35 lines)
- `config.yaml` - Updated loss weights (3 changes)

**Ready to run!** 🚀
