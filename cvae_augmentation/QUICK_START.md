# Quick Start Guide - Using Simplified Loss

## ✅ Changes Pushed to GitHub

The following files are now on GitHub (branch: `claude/augment-extreme-rainfall-data-01Y76Ny6HSLsgQLYcHRBWQph`):

1. **`train_cvae.py`** - Updated to support simplified loss ✅
2. **`losses_simplified.py`** - New loss implementation ✅
3. **`config_simplified.yaml`** - Config using simplified loss ✅
4. **`LOSS_COMPARISON.md`** - Detailed explanation ✅

---

## 🚀 How to Use on Your HPC

### **Step 1: Download Updated Files**

On your HPC, pull the latest changes:

```bash
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation

# Download the 3 new/updated files
# From GitHub branch: claude/augment-extreme-rainfall-data-01Y76Ny6HSLsgQLYcHRBWQph

# You need:
# - train_cvae.py (updated)
# - losses_simplified.py (new)
# - config_simplified.yaml (new)
```

### **Step 2: Train with Simplified Loss**

```bash
# Train with simplified loss (recommended)
python train_cvae.py --config config_simplified.yaml
```

**What will happen:**
```
Creating loss criterion: simplified
  Intensity weighting: scale=36.6, tail_boost=1.5

Starting training...
Epoch 1/150
  Train - Loss: X.XX, Rec: X.XX, KL: X.XX
  Val   - Loss: X.XX, Rec: X.XX, MAE: X.XX, Tail: X.XX
```

---

## 📋 Three Loss Options

### **Option 1: Simplified Loss** ⭐ **RECOMMENDED**

**Config**: `config_simplified.yaml`

```yaml
loss:
  type: "simplified"
  scale: 36.6         # Intensity weighting scale
  tail_boost: 1.5     # Extra boost for P99+ pixels
  lambda_mass: 0.005  # Small mass conservation
  beta_kl: 0.01
  warmup_epochs: 15
```

**What it does:**
- ONE unified reconstruction objective (no trade-off!)
- Intensity weighting: light rain w=0.1, heavy rain w=2.0-3.0
- P99+ pixels get extra 1.5× boost
- Small mass conservation term
- Total: 3 terms (rec + mass + KL)

**Expected:** Clean convergence, good at both overall AND extremes

---

### **Option 2: Minimal Loss** (Ultra-Simple)

**Config**: Create `config_minimal.yaml` by copying `config_simplified.yaml` and changing:

```yaml
loss:
  type: "minimal"     # ← Change this
  scale: 36.6
  tail_boost: 1.5
  beta_kl: 0.01
  warmup_epochs: 15
```

**What it does:**
- ONLY weighted reconstruction + KL (2 terms total)
- No mass conservation
- Simplest possible

**Expected:** Fastest convergence, but no physical constraint

---

### **Option 3: Original Loss** (Current)

**Config**: `config.yaml` (your current config)

```yaml
loss:
  type: "original"    # or omit this field
  lambda_base: 1.0
  lambda_ext: 5.0
  lambda_mass: 0.01
  beta_kl: 0.01
```

**What it does:**
- Multi-objective loss (base + extreme MAE)
- Has the trade-off problem

**Expected:** Same behavior as your current training

---

## 🎯 Recommended Training Strategy

### **Quick Test (Verify It Works)**

```bash
# Test with 10 epochs to make sure everything loads correctly
python train_cvae.py --config config_simplified.yaml
```

Watch for:
```
Creating loss criterion: simplified  ← Should see this
  Intensity weighting: scale=36.6, tail_boost=1.5

Epoch 1/150
  Train - Rec: X.XX  ← This is your weighted reconstruction loss
  Val   - Tail: X.XX  ← Should steadily decrease
```

**If it works**: Kill it (Ctrl+C) and start full training

### **Full Training**

```bash
# Run full training (will take 12-24 hours)
python train_cvae.py --config config_simplified.yaml
```

**Expected behavior:**
- Training runs 60-120 epochs (not 19!)
- Tail MAE decreases steadily (no trade-off!)
- Early stopping on MAE_tail directly
- Best model good at extremes

---

## 📊 What to Watch For

### **Good Signs ✅**

```
Epoch 1:   Tail MAE: 2.0
Epoch 10:  Tail MAE: 1.5
Epoch 20:  Tail MAE: 1.0
Epoch 30:  Tail MAE: 0.7
Epoch 40:  Tail MAE: 0.6
Epoch 50:  Tail MAE: 0.55  ← Stabilizing
Epoch 60:  Tail MAE: 0.54  ← Best model
```

**This is convergence!** Model steadily improves on extremes.

### **Bad Signs ❌**

```
Epoch 1:   Tail MAE: 2.0
Epoch 10:  Tail MAE: 1.0
Epoch 20:  Tail MAE: 1.5   ← Went back up!
Epoch 30:  Tail MAE: 0.9
Epoch 40:  Tail MAE: 1.2   ← Oscillating!
```

**This is the trade-off problem.** If you see this with simplified loss, something is wrong.

---

## 🔧 Troubleshooting

### **ImportError: cannot import SimplifiedCVAELoss**

**Problem:** `losses_simplified.py` not in directory

**Fix:**
```bash
# Make sure file is in cvae_augmentation/
ls losses_simplified.py

# If not there, download from GitHub
```

### **KeyError: 'scale'**

**Problem:** Using old config with simplified loss

**Fix:** Use `config_simplified.yaml`, not `config.yaml`

### **Loss not decreasing**

**Possible causes:**
1. Learning rate too high/low
2. Model too large (overfitting)
3. Tail boost too aggressive

**Try:**
```yaml
# Reduce tail_boost
tail_boost: 1.2  # instead of 1.5

# Or adjust scale
scale: 50.0      # less extreme emphasis
```

---

## 📈 Comparing to Current Model

After training completes, compare:

| Model | Checkpoint | Tail MAE | Notes |
|-------|-----------|----------|-------|
| **Current** | epoch_025.pt | 0.52 | Best from old training |
| **Simplified** | cvae_best.pt | ??? | Should be <0.6 |

**If simplified Tail MAE < current:**
→ Use simplified model for augmentation ✅

**If simplified Tail MAE > current:**
→ Tune hyperparameters or use current model

---

## 💡 Hyperparameter Tuning

If results not good enough, try:

### **More Extreme Emphasis**
```yaml
scale: 25.0       # Lower scale = more extreme focus
tail_boost: 2.0   # Higher boost = more P99+ emphasis
```

### **Less Extreme Emphasis**
```yaml
scale: 50.0       # Higher scale = more balanced
tail_boost: 1.2   # Lower boost = less P99+ emphasis
```

### **Remove Mass Constraint**
```yaml
type: "minimal"   # Switch to minimal loss
# Removes lambda_mass completely
```

---

## ✅ Summary

**What you need to do:**

1. ✅ Download 3 files from GitHub:
   - `train_cvae.py` (updated)
   - `losses_simplified.py` (new)
   - `config_simplified.yaml` (new)

2. ✅ Run training:
   ```bash
   python train_cvae.py --config config_simplified.yaml
   ```

3. ✅ Wait 12-24 hours for training

4. ✅ Check if Tail MAE < 0.6 (better than current 0.52)

5. ✅ Use best model for augmentation

**Expected improvement:**
- No trade-off (both MAE_all AND Tail MAE improve!)
- Convergence at 60-100 epochs
- Best model good at extremes

Your HPC code will now be consistent with GitHub! 🎉
