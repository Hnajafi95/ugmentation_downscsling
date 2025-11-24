# 🚨 QUICK FIX GUIDE - cVAE Mean Collapse

## ⚡ TL;DR - What You Need to Know

Your cVAE had **"mean collapse"** - it painted everything with average rain (~19mm/day) instead of realistic storms.

**Root cause**: Three bugs working together:
1. ❌ Pixel threshold applied in wrong space (normalized vs mm/day)
2. ❌ Conditioning dropout removed spatial guidance
3. ❌ Aggressive loss weights caused "hedging" behavior

**Fix**: Updated code + **MUST RETRAIN MODEL**

---

## 🔥 CRITICAL: You Must Retrain!

**The model architecture changed** (removed conditioning dropout).
Your old checkpoint won't work correctly with the new code.

```bash
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation

# Run the automated fix script:
bash fix_mean_collapse.sh
```

**This will**:
1. Backup your old model
2. Retrain with new architecture (~1-2 hours)
3. Generate new samples with proper denormalization
4. Evaluate and visualize

---

## 📋 What Changed?

### 1. sample_cvae.py - Denormalization Fix

**Problem**: Applying 0.1 mm/day threshold to normalized data (meaningless!)

**Fix**: Denormalize → Threshold → Re-normalize

```python
# NEW: Proper workflow
Y_hat_normalized = model.decode(z, h_X).cpu().numpy()
Y_hat_mmday = denormalize_to_mmday(Y_hat_normalized, mean_pr, std_pr)  # To mm/day
Y_hat_mmday[Y_hat_mmday < 0.5] = 0.0  # Threshold in physical space
Y_hat_renormalized = normalize_back(Y_hat_mmday, mean_pr, std_pr)  # Save normalized
```

### 2. model_cvae.py - Remove Dropout

**Problem**: 50% of time, model couldn't see where rain should be (h_X zeroed)

**Fix**: Removed `self.cond_dropout = nn.Dropout(p=0.5)`

```python
# OLD (line 264):
self.cond_dropout = nn.Dropout(p=0.5)
h_X_dropped = self.cond_dropout(h_X)

# NEW (line 309):
# No dropout - model always sees spatial structure
h = torch.cat([z, h_X], dim=1)
```

### 3. config.yaml - Relax Loss Weights

**Problem**: Too aggressive penalties → model predicted safe average everywhere

**Fix**:
```yaml
# OLD:
w_min: 1.0        # Too strict
tail_boost: 3.0   # Too aggressive

# NEW:
w_min: 0.2        # Relaxed
tail_boost: 1.5   # Moderate
pixel_threshold: 0.5  # Increased from 0.1
```

---

## 📊 Expected Results

### Before Fix:
```
✗ Spatial correlation:  -0.15  (random)
✗ Wet fraction:          100%  (drizzle everywhere)
✗ Mass conservation:    -1.06  (losing rain)
✗ Visual: Flat gray plane with ~19mm everywhere
```

### After Fix:
```
✓ Spatial correlation:   >0.60  (realistic patterns)
✓ Wet fraction:        75-80%  (realistic dry areas)
✓ Mass conservation:     ~0.0  (preserves total)
✓ Visual: Concentrated storms with dry regions
```

---

## 🎯 Verification Checklist

After running `fix_mean_collapse.sh`, check:

### Must Pass:
- [ ] **Wet fraction < 90%**: If still 100%, threshold didn't work
- [ ] **Spatial correlation > 0.4**: If negative, model didn't train well
- [ ] **Visual check**: See storms, not uniform drizzle

### Should Pass:
- [ ] **Mass conservation < ±0.3**: Total rain preserved
- [ ] **Diversity 0.10-0.30**: Samples have variation

### Files to Check:
```bash
# View evaluation metrics
cat outputs/cvae_final/visualizations/statistics.json

# Look at visualizations
ls outputs/cvae_final/visualizations/*.png

# Check training converged
tail -20 outputs/cvae_final/logs/train_log.csv
```

---

## 🔧 Troubleshooting

### Problem: Wet fraction still > 95%

**Likely cause**: Denormalization not working

**Quick check**:
```python
import numpy as np
mean_pr = np.load('data/metadata/ERA5_mean_train.npy')
print(f"Mean range: {mean_pr.min():.2f} to {mean_pr.max():.2f}")
# Should be around -2 to 4 (log-space values)
```

**Fix**: Increase threshold
```yaml
# In config.yaml:
sampling:
  pixel_threshold: 1.0  # Increase from 0.5
```

### Problem: Spatial correlation still low (< 0.3)

**Likely cause**: Model didn't train properly

**Checks**:
1. Look at training log - did it converge?
2. Check KL divergence - should be ~100-200, not 1000+
3. Look at validation MAE_tail - should decrease over epochs

**Fix**: Retrain with even lower beta_kl
```yaml
loss:
  beta_kl: 0.0001  # Reduce from 0.0005
```

### Problem: Training unstable / NaN losses

**Likely cause**: Loss weights still too aggressive

**Fix**:
```yaml
loss:
  w_min: 0.1       # Even more relaxed
  tail_boost: 1.0  # Remove boost entirely
```

---

## 📁 Files Modified

```
cvae_augmentation/
├── sample_cvae.py          ← Added denormalization (40 lines)
├── model_cvae.py           ← Removed dropout (2 lines)
├── config.yaml             ← Relaxed losses (5 lines)
├── fix_mean_collapse.sh    ← Automated pipeline (NEW)
├── MEAN_COLLAPSE_FIX.md    ← Detailed explanation (NEW)
└── QUICK_FIX_GUIDE.md      ← This file (NEW)
```

---

## ⏱️ Time Required

- **Code review**: 5 minutes (read this file)
- **Retraining**: 1-2 hours (GPU)
- **Sampling**: 10-20 minutes
- **Evaluation**: 5-10 minutes
- **Total**: ~2-3 hours

---

## 🆘 If Still Having Issues

1. **Check denormalization**:
   ```bash
   python -c "from sample_cvae import denormalize_to_mmday; print('Import OK')"
   ```

2. **Verify model loaded**:
   ```bash
   python -c "from model_cvae import CVAE; print('Model OK')"
   ```

3. **Check config**:
   ```bash
   grep -A 3 "pixel_threshold" config.yaml
   # Should show: pixel_threshold: 0.5
   ```

4. **Review detailed docs**: `MEAN_COLLAPSE_FIX.md`

---

## ✨ Summary

**What happened**: Model painted everything with average rain (mean collapse)

**Why**: Three bugs (normalization mismatch + dropout + aggressive losses)

**Fix**: Updated code + **MUST RETRAIN**

**Command**: `bash fix_mean_collapse.sh`

**Time**: ~2-3 hours total

**Expected**: Realistic storms with 75-80% wet fraction, correlation >0.6

---

**Ready to run!** 🚀

Just execute `bash fix_mean_collapse.sh` and let it handle everything.
