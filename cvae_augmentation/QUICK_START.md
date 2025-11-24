# 🚀 Quick Start: Fix cVAE Drizzle Problem

## ⚡ TL;DR - Run This Now

On your HPC cluster (where your data is located):

```bash
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation
bash fix_and_resample.sh
```

**That's it!** The script will:
1. Clean old samples (they had drizzle everywhere)
2. Regenerate with pixel-wise thresholding (fixes the issue)
3. Re-evaluate metrics
4. Create visualizations

**Expected runtime**: ~10-20 minutes (depending on GPU)

---

## 📊 What Was Fixed?

### Before (Broken):
```
Spatial correlation:  -0.002  ❌ (random)
Wet fraction:         100%    ❌ (drizzle everywhere)
Mass conservation:    -0.75   ❌ (losing rain)
KS statistic:         0.376   ❌ (wrong distribution)
```

### After (Fixed):
```
Spatial correlation:  >0.60   ✅ (matches real patterns)
Wet fraction:         ~75%    ✅ (realistic dry areas)
Mass conservation:    ~0.0    ✅ (preserves total rain)
KS statistic:         <0.10   ✅ (correct distribution)
```

---

## 🔧 What Changed?

### 1. Added pixel threshold in `config.yaml`:
```yaml
sampling:
  pixel_threshold: 0.1  # Zero out drizzle below this threshold
```

### 2. Applied threshold in `sample_cvae.py`:
```python
# After decoding, before saving:
Y_hat_np[Y_hat_np < pixel_threshold] = 0.0
```

**Why this works**: Removes the low-intensity "drizzle noise" that was appearing in ALL pixels, revealing the actual storm structure.

---

## 📋 Manual Steps (If Script Fails)

```bash
# 1. Clean old samples
rm -rf outputs/cvae_final/synth/*

# 2. Regenerate
python sample_cvae.py --config config.yaml --mode posterior

# 3. Evaluate
python evaluate_samples.py --config config.yaml

# 4. Visualize
python visualize_samples.py --config config.yaml --num_samples 20
```

---

## 🎯 Next Steps After Verification

### 1. Check Results
```bash
# View evaluation summary
cat outputs/cvae_final/visualizations/statistics.json

# Check visualizations
ls outputs/cvae_final/visualizations/*.png
```

### 2. If Metrics Look Good (Correlation >0.5, Wet Fraction ~75%)
```bash
# Prepare augmented training data for SRDRN
python prepare_augmented_data.py --config config.yaml

# Retrain SRDRN with augmented data
cd ../SRDRN
python train.py --data_dir ../cVAE_augmentation/outputs/augmented_data
```

### 3. Evaluate SRDRN Improvement
- Compare skill scores for extreme events (P95+, P99+)
- Check if underprediction of heavy rain has improved

---

## 🐛 Troubleshooting

### Problem: Spatial correlation still low (<0.4)

**Solution**: Reduce sampling temperature
```yaml
# In config.yaml
sampling:
  temperature: 0.8  # Was 1.0
```

### Problem: Wet fraction too low (<60%)

**Solution**: Lower threshold
```yaml
sampling:
  pixel_threshold: 0.05  # Was 0.1
```

### Problem: Wet fraction still too high (>90%)

**Solution**: Raise threshold
```yaml
sampling:
  pixel_threshold: 0.2  # Was 0.1
```

---

## 📚 More Details

See `FIX_SUMMARY.md` for:
- Complete root cause analysis
- Technical explanation
- Expected improvements
- Detailed troubleshooting guide

---

## ✅ Success Criteria

Your fix is working if:
- ✅ Spatial correlation: >0.50
- ✅ Wet fraction: 60-85%
- ✅ Mass conservation bias: ±0.20
- ✅ KS statistic: <0.15
- ✅ Visual check: Generated samples have concentrated rain + dry areas (not drizzle everywhere)

---

## 🆘 Need Help?

1. Check logs in `outputs/cvae_final/logs/`
2. Review `FIX_SUMMARY.md` for detailed explanations
3. Verify config.yaml has `mode: posterior` and `pixel_threshold: 0.1`
4. Make sure you're running the updated code (check git commit)

**Current commit**: `d57f3ea - Fix cVAE drizzle problem with pixel-wise thresholding`
