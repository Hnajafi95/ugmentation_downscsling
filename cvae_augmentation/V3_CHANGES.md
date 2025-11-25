# V3 Configuration Changes - Summary

## Problem Statement

V2 suffered from **catastrophic mode collapse**:
- **Diversity: 0.0001** (should be >0.15) - All samples nearly identical!
- **Spatial correlation: 0.8674** - Too high, suggests pixel-perfect copying
- **Poor SRDRN test generalization** compared to V1

V1 had better diversity and test generalization, but lacked:
- Proper denormalization (bug in sample_cvae.py)
- Gradient loss for sharp boundaries
- Correct pixel thresholding

## Root Cause Analysis

### V2's Mode Collapse

**Primary cause: beta_kl too low**
- V2: `beta_kl = 0.0005` (20x lower than V1)
- Effect: KL regularization too weak → model learns to ignore latent space `z`
- Result: All samples become deterministic copies of conditioning input

**Secondary cause: Gradient loss**
- V2: `lambda_grad = 2.0` (strong penalty on spatial gradients)
- Effect: Forces model to match exact edge patterns from training data
- Combined with low beta_kl → perfect storm for mode collapse

### V1's Success

V1 had better diversity because:
- `beta_kl = 0.01` - Strong enough to maintain useful latent space
- `tail_boost = 2.0` - Stronger emphasis on P99+ extremes
- Better balance: `w_min=0.1, w_max=3.0, lambda_mass=0.005`
- Result: Lower train scores but **better test generalization**

## V3 Solution

**Philosophy**: Best of both worlds
- V1's proven loss weights (diversity + generalization)
- V2's technical fixes (denormalization + thresholding)
- Remove redundant gradient loss

## Configuration Changes

| Parameter | V1 | V2 | V3 | Reason |
|-----------|----|----|----|---------|
| **beta_kl** | 0.01 | 0.0005 | **0.01** ✓ | **PRIMARY FIX** - Restore latent space diversity |
| **lambda_grad** | ❌ | 2.0 | **0.0** ✓ | Remove pixel-copying tendency |
| **w_min** | 0.1 | 0.5 | **0.1** ✓ | Better dynamic range |
| **w_max** | 3.0 | 4.0 | **3.0** ✓ | Reduce overfitting |
| **tail_boost** | 2.0 | 1.5 | **2.0** ✓ | Stronger extreme emphasis |
| **lambda_mass** | 0.005 | 0.01 | **0.005** ✓ | More flexibility |
| **pixel_threshold** | ❌ | 1.0 | **1.0** ✓ | Keep V2 fix (no NaN) |
| **denormalization** | ❌ (bug) | ✅ | ✅ | Keep V2 fix |
| **d_z** | 64 | 128 | **128** ✓ | Keep higher capacity |

## Expected Results

### cVAE Evaluation Metrics

| Metric | V2 (Actual) | V3 (Expected) | Target |
|--------|-------------|---------------|---------|
| **Diversity** | 0.0001 ❌ | **>0.15** ✓ | >0.15 |
| **Spatial Correlation** | 0.8674 ❌ | **0.5-0.6** ✓ | 0.5-0.6 |
| **Mass Conservation** | -16.2% ✓ | **±10-15%** ✓ | <20% |
| **KS p-value** | 0.0000 ❌ | **>0.05** ✓ | >0.05 |

### SRDRN Test Performance

V3 should **match or exceed V1** on test metrics:
- Test CSI (P95): V1=0.086, V2=0.078 → V3 target ≥0.086
- Test Correlation: V1=0.256, V2=0.223 → V3 target ≥0.256
- Test Recall (P95): V1=0.102, V2=0.091 → V3 target ≥0.102

## Technical Rationale

### Why Remove Gradient Loss?

**Gradient loss is redundant** because:

1. **Original problem already solved**:
   - V1 issue: "drizzle everywhere" due to normalization bug
   - V2 fix: Proper denormalization + pixel thresholding
   - No longer need gradient loss for sharp boundaries

2. **Gradient loss harms diversity**:
   - Penalizes spatial variations
   - Forces exact pattern matching
   - Combined with low beta_kl → mode collapse

3. **V1 proved it's unnecessary**:
   - V1 had no gradient loss
   - V1 achieved better test generalization
   - V1 had better diversity (implied by SRDRN results)

### Why Increase beta_kl 20x?

**KL divergence is critical for VAE diversity**:

**Low beta_kl (0.0005):**
- Posterior q(z|x,y) ≈ Prior p(z) constraint too weak
- Model learns to ignore z, uses only conditioning h_X
- All samples collapse to deterministic output
- Diversity → 0

**Proper beta_kl (0.01):**
- Forces posterior to stay close to prior
- Model must use z to encode variations
- Samples explore latent space
- Diversity >0.15

## Training Instructions

```bash
# On HPC
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation

# Train with V3 config
python train_cvae.py --config config_v3.yaml

# After training, generate samples
python sample_cvae.py --config config_v3.yaml

# Evaluate diversity
python evaluate_samples.py --config config_v3.yaml --num_days 100

# Expected output:
# - Diversity: >0.15 ✓
# - Spatial correlation: 0.5-0.6 ✓
# - No more mode collapse!
```

## Success Criteria

V3 is successful if:

1. ✅ **Diversity >0.15** (V2 had 0.0001)
2. ✅ **Spatial correlation 0.5-0.6** (V2 had 0.8674 - too high)
3. ✅ **KS p-value >0.05** (distributions match)
4. ✅ **SRDRN test metrics ≥ V1** (prove augmentation helps)
5. ✅ **No NaN correlations** (pixel_threshold=1.0 prevents this)

## Fallback Plan

If V3 has sharp boundary issues (unlikely):
- Try `lambda_grad: 0.5` (compromise, much weaker than V2's 2.0)
- But expect this won't be needed - V1 didn't need it!

## Files Changed

1. **config_v3.yaml** - New V3 configuration
2. **V3_CHANGES.md** - This document (tracking changes)

## Files Unchanged (Already Fixed in V2)

1. **sample_cvae.py** - Denormalization fix (lines 21-54)
2. **losses.py** - Gradient loss function (lines 197-228)
3. **evaluate_samples.py** - Load pre-generated samples
4. **config.yaml** - Keep V2 as reference

---

**Bottom Line**: V3 = V1's diversity + V2's technical fixes - redundant gradient loss
