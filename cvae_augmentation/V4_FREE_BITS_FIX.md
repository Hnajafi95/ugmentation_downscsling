# V4: Free Bits Solution for Posterior Collapse

## Problem Diagnosis

**V3 still had mode collapse despite beta_kl=0.01:**
```
Diversity: 0.0001 (target: >0.15)
Spatial correlation: 0.8911 (too high - pixel-copying)
```

**Test Results:**
- **Temperature=3**: Improved diversity to 0.0517 (still <0.15)
- **Prior mode**: Improved diversity to only 0.0057 (barely better)

**Root Cause Identified:**

The reconstruction loss **overwhelms** the KL loss:
```
Reconstruction Loss: ~8.0
KL Loss contribution: 0.01 × 12.0 = 0.12
Ratio: 67:1
```

The model learns to **collapse posterior variance** (logvar → -∞) to achieve perfect reconstruction:
```python
z = mu + eps * exp(logvar/2)
# If logvar → -∞, then exp(logvar/2) → 0
# Result: z ≈ mu (deterministic, no randomness)
```

All samples become identical because there's no variation in z.

---

## Free Bits Solution

### What are Free Bits?

**Free bits** enforce a **minimum KL divergence per latent dimension**:

```python
# Standard KL
kl_per_dim = -0.5 * (1 + logvar - mu^2 - exp(logvar))

# Free bits: each dimension must have at least 0.5 nats
kl_per_dim_constrained = max(kl_per_dim, 0.5)
```

**Effect**: The model **cannot** collapse variance below the threshold. It's forced to maintain stochasticity.

### Why It Works

With 128 latent dimensions and free_bits=0.5:
- **Minimum KL**: 128 × 0.5 = 64 nats
- **KL contribution**: 0.05 × 64 = 3.2
- **New ratio**: 8.0:3.2 = 2.5:1 (much more balanced!)

The model can no longer cheat by making logvar extremely small.

---

## V4 Configuration Changes

| Parameter | V3 | V4 | Effect |
|-----------|----|----|---------|
| **use_free_bits** | false | **true** ✓ | **PRIMARY FIX** - Prevents variance collapse |
| **free_bits** | - | **0.5** ✓ | Minimum KL per dimension |
| **beta_kl** | 0.01 | **0.05** ✓ | 5x increase for better balance |
| **lambda_grad** | 0.0 | **0.0** ✓ | Keep removed (prevents pixel-copying) |
| **w_min** | 0.1 | **0.1** ✓ | Keep V1 value |
| **w_max** | 3.0 | **3.0** ✓ | Keep V1 value |
| **tail_boost** | 2.0 | **2.0** ✓ | Keep V1 value |
| **lambda_mass** | 0.005 | **0.005** ✓ | Keep V1 value |

---

## Expected Results

### Loss Balance During Training

**V3 (broken):**
```
Rec: 2.23, KL: 12.48 × 0.01 = 0.12 → Total: 2.35
Ratio: 2.23:0.12 = 18.6:1 (KL irrelevant)
```

**V4 (fixed):**
```
Rec: ~2.20, KL: 64.0 × 0.05 = 3.2 → Total: 5.40
Ratio: 2.20:3.2 = 0.69:1 (KL actually matters!)
```

### Evaluation Metrics

| Metric | V3 (Actual) | V4 (Expected) |
|--------|-------------|---------------|
| **Diversity** | 0.0001 ❌ | **>0.15** ✓ |
| **Spatial Correlation** | 0.8911 ❌ | **0.5-0.6** ✓ |
| **Mass Conservation** | -15% ✓ | **±10-15%** ✓ |
| **KS p-value** | 0.0000 ❌ | **>0.05** ✓ |

---

## Training Instructions

```bash
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation

# Pull V4 updates
git pull origin claude/review-srdrn-preprocessing-01EAAjCDyGDVyYztrGttJP1d

# Train with V4 config
python train_cvae.py --config config_v4.yaml

# Monitor training - you should see:
# - KL loss MUCH higher (~64 instead of ~12)
# - Total loss higher but more balanced
# - Training might take slightly longer

# Generate samples
python sample_cvae.py --config config_v4.yaml

# Evaluate - diversity should be >0.15!
python evaluate_samples.py --config config_v4.yaml --num_days 100
```

---

## Technical Details

### Free Bits Implementation

**Added to `losses.py`:**
```python
def kl_divergence_with_free_bits(mu, logvar, free_bits=0.5):
    """Enforce minimum KL per dimension."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Constraint: each dim must have at least free_bits nats
    kl_per_dim_constrained = torch.max(
        kl_per_dim,
        torch.tensor(free_bits, device=kl_per_dim.device)
    )

    return kl_per_dim_constrained.sum(dim=1).mean()
```

**Modified `SimplifiedCVAELoss`:**
```python
class SimplifiedCVAELoss(nn.Module):
    def __init__(self, ..., use_free_bits=False, free_bits=0.5):
        self.use_free_bits = use_free_bits
        self.free_bits = free_bits

    def forward(self, ...):
        if self.use_free_bits:
            kl_loss, kl_info = kl_divergence_with_free_bits(
                mu, logvar, self.free_bits
            )
        else:
            kl_loss = kl_divergence(mu, logvar)
```

### Why Temperature=3 Helped Slightly

With temperature=3 in V3:
```python
z = mu + 3.0 * eps * exp(logvar/2)
```

This artificially **amplified** the (tiny) variance. Diversity increased to 0.0517, proving:
1. The model CAN generate variations
2. But only when we force it by scaling noise

Free bits is the **proper fix** - forces the model to learn appropriate variance during training, not just at sampling time.

---

## Success Criteria

V4 succeeds if:

1. ✅ **Diversity >0.15** without temperature scaling
2. ✅ **Spatial correlation 0.5-0.6** (not 0.89)
3. ✅ **KS p-value >0.05** (distributions match)
4. ✅ **Training KL loss ~64** (enforced by free bits)
5. ✅ **SRDRN test metrics ≥ V1** (prove augmentation works)

---

## Fallback Plan

**If V4 still fails (diversity <0.15):**

1. **Increase free_bits**: Try 1.0 or 2.0 (more aggressive)
2. **Increase beta_kl**: Try 0.1 or 0.2 (stronger KL weight)
3. **Reduce reconstruction weight**: Scale down loss.scale from 20→10

**If V4 succeeds but sharp boundaries are needed:**
- Try `lambda_grad: 0.5` (weak gradient loss)
- But this likely won't be needed

---

## Files Changed

1. **losses.py** - Added `kl_divergence_with_free_bits()` function
2. **losses.py** - Updated `SimplifiedCVAELoss` to support free bits
3. **config_v4.yaml** - New configuration with free bits enabled
4. **V4_FREE_BITS_FIX.md** - This documentation

---

## References

**Free Bits Paper:**
- "Generating Sentences from a Continuous Space" (Bowman et al., 2016)
- Shows free bits prevents VAE posterior collapse in NLP

**Applied to Computer Vision:**
- Used successfully in pixel-level VAEs (VQ-VAE2, etc.)
- Standard technique for preventing mode collapse

---

**Bottom Line**: V4 = V3 + free bits constraint = forced diversity through minimum variance
