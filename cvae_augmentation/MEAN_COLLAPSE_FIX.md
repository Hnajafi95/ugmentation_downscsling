# cVAE Mean Collapse Problem - Root Cause & Complete Fix

## 🔴 Problem: Mean Collapse

After implementing pixel thresholding, the model showed a new failure mode called **"Mean Collapse"**:

### Symptoms:
```
Spatial correlation: -0.1462 (negative/random patterns)
Wet fraction:         1.000   (100% - still drizzle everywhere!)
Mass conservation:    -1.0575 (worse than before)
Sample diversity:     0.0559  (very low)
Mean precipitation:   19.14   (almost identical to real: 19.84)
```

### The Behavior:
The model stopped trying to predict **where** the rain is. Instead, it painted the **entire map** with ~19 mm/day (the average rainfall), creating a flat, uniform precipitation field with no spatial structure.

---

## 🔍 Root Cause Analysis

### Problem 1: Normalization Space Mismatch (PRIMARY)

**Issue**: Pixel threshold was applied in the WRONG space.

- **Model output**: log1p-transformed + z-scored (normalized space)
- **Pixel threshold**: 0.1 mm/day (physical space)
- **Result**: Threshold of 0.1 in normalized space is meaningless

**Example**:
```
Normalized value: 0.05  (could be 15 mm/day after denormalization!)
Threshold check:  0.05 < 0.1 → REMOVE
Reality:          Should NOT have been removed!
```

**Why wet fraction stayed 100%**:
- In z-scored space (mean≈0, std≈1), most values are between -2 and +2
- A threshold of 0.1 only removes values very close to the mean
- After denormalization, even 0.05 in normalized space can become 10+ mm/day
- Result: No pixels were actually being zeroed out

### Problem 2: Conditioning Dropout (SECONDARY)

**Issue**: Decoder had `nn.Dropout(p=0.5)` on `h_X` (low-res input encoding).

```python
# OLD CODE (model_cvae.py line 264):
self.cond_dropout = nn.Dropout(p=0.5)

# In forward pass:
h_X_dropped = self.cond_dropout(h_X)  # 50% chance of zeroing h_X
h = torch.cat([z, h_X_dropped], dim=1)
```

**Why this caused mean collapse**:
1. 50% of training batches: `h_X` is zeroed out
2. Model loses spatial guidance from low-res input
3. Combined with aggressive loss penalties (w_min=1.0, tail_boost=3.0)
4. Model learns: "safest bet is to predict average rain everywhere"
5. This minimizes error across all conflicting objectives

**The mathematical trap**:
- Predict 0 and it rains heavy → HUGE penalty from tail_boost=3.0
- Predict heavy and it's dry → HUGE penalty from w_min=1.0
- Predict average (~19mm) everywhere → Small penalty on average
- **Model chose the "safe" middle ground**

### Problem 3: Aggressive Loss Weights

**Issue**: Multiple competing objectives with extreme penalties.

```yaml
# OLD CONFIG:
w_min: 1.0        # Equal weight for dry regions (too aggressive)
tail_boost: 3.0   # 3x boost for extremes (too aggressive)
```

**Result**: "Loss function civil war"
- Tail boost: "Predict HUGE values for heavy rain!"
- w_min=1.0: "Predict EXACTLY zero for dry areas!"
- KL divergence: "Keep latent simple!"
- **Pareto optimality problem**: Model found a "safe" compromise that satisfies none well

---

## ✅ Complete Fix - Three Parts

### Fix 1: Denormalize Before Pixel Thresholding

**Changes to `sample_cvae.py`**:

```python
# Added denormalization function (lines 21-53):
def denormalize_to_mmday(y_normalized, mean_pr, std_pr):
    """Convert normalized precipitation to mm/day."""
    # Reverse z-score
    log1p_precip = y_normalized * std_pr + mean_pr
    # Reverse log1p
    y_mmday = np.expm1(log1p_precip)
    return np.maximum(y_mmday, 0.0)

# In sampling loop (lines 299-323):
# 1. Model outputs in normalized space
Y_hat_normalized = Y_hat.cpu().numpy()[0]

# 2. Denormalize to mm/day
Y_hat_mmday = denormalize_to_mmday(Y_hat_normalized, mean_pr, std_pr)

# 3. Apply pixel threshold in PHYSICAL space
Y_hat_mmday[Y_hat_mmday < pixel_threshold] = 0.0  # Now meaningful!

# 4. Re-normalize before saving (to match training data format)
log1p_precip = np.log1p(Y_hat_mmday)
Y_hat_renormalized = (log1p_precip - mean_pr) / (std_pr + 1e-8)
np.save(Y_hr_syn_path, Y_hat_renormalized)
```

**Why this works**:
- Threshold now applied to actual mm/day values
- 0.5 mm/day is climatologically meaningful (trace precipitation)
- Will correctly zero out light drizzle
- Saved data still in normalized format (consistent with training)

### Fix 2: Remove Conditioning Dropout

**Changes to `model_cvae.py`**:

```python
# REMOVED (line 264):
# self.cond_dropout = nn.Dropout(p=0.5)

# Updated forward pass (line 309):
# OLD: h = torch.cat([z, h_X_dropped], dim=1)
# NEW: h = torch.cat([z, h_X], dim=1)  # No dropout!
```

**Why this works**:
- Model always has access to spatial information from `h_X`
- Can learn where to put rain based on low-res input
- Still uses `z` for diversity (controlled by beta_kl)
- Prevents "hedging" behavior

**CRITICAL**: Must retrain model after this change!
- Old weights were trained WITH dropout
- Architecture has changed - old checkpoint incompatible

### Fix 3: Relax Loss Weights

**Changes to `config.yaml`**:

```yaml
# UPDATED loss weights:
loss:
  w_min: 0.2        # REDUCED from 1.0 (stop panicking about dry spots)
  w_max: 4.0        # Keep emphasis on heavy rain
  tail_boost: 1.5   # REDUCED from 3.0 (stop hedging behavior)
  lambda_grad: 1.0  # Added: edge sharpness loss

# UPDATED sampling:
sampling:
  pixel_threshold: 0.5  # INCREASED from 0.1 (more aggressive cleanup)
```

**Why this works**:
- Lower w_min: Model comfortable predicting zeros
- Lower tail_boost: Less panic about missing extremes
- lambda_grad: Encourages sharp edges (structural loss)
- Higher pixel_threshold: More aggressive drizzle removal

---

## 📊 Expected Improvements

| Metric | Before Fix | After Fix (Expected) | Status |
|--------|------------|----------------------|--------|
| **Spatial Correlation** | -0.15 | **>0.60** | ✅ Storm locations match |
| **Wet Fraction** | 100% | **~75-80%** | ✅ Realistic dry areas |
| **Mass Conservation** | -1.06 | **±0.20** | ✅ Total mass preserved |
| **Sample Diversity** | 0.056 | **0.15-0.25** | ✅ Good variation |
| **Mean Precipitation** | 19.14 | **~19.84** | ✅ Matches real data |

---

## 🚀 How to Apply the Fix

### Step 1: Update Code (Already Done in Git)

All changes have been committed to the repository:
- `sample_cvae.py`: Denormalization added
- `model_cvae.py`: Conditioning dropout removed
- `config.yaml`: Loss weights relaxed

### Step 2: RETRAIN the Model

**CRITICAL**: You must retrain because the architecture changed!

```bash
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation

# Option 1: Use automated script
bash fix_mean_collapse.sh

# Option 2: Manual retraining
python train_cvae.py --config config.yaml
```

**Expected training time**: ~1-2 hours on GPU
**Expected early stopping**: Around epoch 100-150

### Step 3: Generate New Samples

```bash
# Clean old samples
rm -rf outputs/cvae_final/synth/*

# Generate with new model
python sample_cvae.py --config config.yaml --mode posterior
```

### Step 4: Evaluate

```bash
# Evaluate samples
python evaluate_samples.py --config config.yaml

# Visualize
python visualize_samples.py --config config.yaml --num_samples 20
```

---

## 🎯 Success Criteria

After retraining and sampling, verify:

### Must Have (Critical):
- ✅ **Spatial correlation > 0.50**: Generated rain in correct locations
- ✅ **Wet fraction 65-85%**: Not 100% (realistic dry areas)
- ✅ **Mass conservation ±0.30**: Preserving total precipitation

### Should Have (Important):
- ✅ **Visual inspection**: See concentrated storms, not uniform drizzle
- ✅ **Diversity 0.10-0.30**: Samples are different but similar
- ✅ **No "flat gray planes"**: Clear spatial structure visible

### Nice to Have (Optional):
- ✅ **KS statistic < 0.15**: Distribution matches well
- ✅ **P99 values > 100 mm/day**: Capturing extreme events

---

## 🔧 Troubleshooting

### If wet fraction is still > 95%:

**Diagnosis**: Pixel threshold still too low or denormalization broken

**Fix**:
```yaml
# In config.yaml:
sampling:
  pixel_threshold: 1.0  # Increase from 0.5
```

**Verify denormalization**:
```python
# Quick test in Python:
import numpy as np
mean_pr = np.load('data/metadata/ERA5_mean_train.npy')
std_pr = np.load('data/metadata/ERA5_std_train.npy')

# Check range
print(f"Mean range: {mean_pr.min():.3f} to {mean_pr.max():.3f}")
print(f"Std range: {std_pr.min():.3f} to {std_pr.max():.3f}")

# Should be reasonable log-space values, not huge numbers
```

### If spatial correlation is still low (< 0.3):

**Diagnosis**: Model didn't train properly without dropout

**Fix**:
1. Check training logs for convergence
2. Verify KL divergence didn't explode (should be ~100-200)
3. Try reducing beta_kl further:
   ```yaml
   loss:
     beta_kl: 0.0001  # Even lower
   ```

### If training is unstable:

**Diagnosis**: Loss weights still too aggressive

**Fix**:
```yaml
loss:
  w_min: 0.1       # Even more relaxed
  tail_boost: 1.0  # No boost
  lambda_grad: 0.5 # Reduce sharpness penalty
```

---

## 📚 Technical Notes

### Why Log1p + Z-Score?

Precipitation has extreme dynamic range (0.1 to 500+ mm/day):
1. **log1p**: Compresses range, handles zeros gracefully
2. **z-score**: Normalizes per-pixel statistics for training stability

### Why Re-Normalize After Thresholding?

The downstream evaluation and mixing scripts expect normalized format:
- `evaluate_samples.py`: Loads dataset (normalized)
- `visualize_samples.py`: Has its own denormalization
- `prepare_augmented_data.py`: Expects same format as training data

### Why Posterior Sampling?

For data augmentation of specific storms:
- **Posterior** `z ~ q(z|x,y)`: Variations of the same storm ✅
- **Prior** `z ~ p(z)`: Completely different random storm ❌

### Why Not Just Train with Sparsity Loss?

Could work, but:
- Post-processing is faster (no retraining for tuning)
- More flexible (can adjust threshold per use case)
- Reversible (can regenerate without threshold if needed)

---

## 📖 Related Reading

- **Posterior Collapse**: [Bowman et al., 2016]
- **Drizzle Problem**: [Stephens et al., 2010 - GCM precipitation]
- **Loss Function Design**: [Leinonen et al., 2020 - Precipitation downscaling]
- **Pareto Optimality**: [Multi-objective optimization in deep learning]

---

## ✨ Summary

**Three bugs, three fixes**:

1. **Normalization bug** → Denormalize before thresholding
2. **Architecture bug** → Remove conditioning dropout, RETRAIN
3. **Loss design bug** → Relax aggressive weights

**Critical step**: Must retrain model after architecture change!

**Expected result**: Realistic spatial patterns with proper wet/dry areas

**Files modified**:
- `sample_cvae.py` - Added denormalization (40 lines)
- `model_cvae.py` - Removed dropout (2 lines)
- `config.yaml` - Relaxed losses (5 lines)
- `fix_mean_collapse.sh` - Automated pipeline (NEW)

**Time to fix**: ~2-3 hours (mostly retraining)

**Ready to run!** 🚀
