# cVAE Improvement Plan for Extreme Precipitation Downscaling

## Executive Summary

Current cVAE results show **significant improvement** in extreme event prediction:
- CSI (P95) test: 0.0032 → 0.089 (28x improvement!)
- SEDI (P95) test: 0.047 → 0.427 (9x improvement)
- Correlation test: 0.114 → 0.250 (2.2x improvement)

However, **sample quality metrics are poor**:
- Diversity: 0.0438 (very low)
- Spatial correlation: 0.24 (poor)
- Wet fraction: 100% vs 75.8% (spurious drizzle everywhere)

This plan addresses 5 critical issues to improve sample quality while maintaining the extreme event improvements.

---

## Issue #1: Posterior Collapse → Low Diversity

### Problem
- Latent dimension d_z=64 is too small
- KL weight β=0.01 is too aggressive early in training
- Diversity score of 0.0438 indicates samples are nearly identical

### Solution
**A. Increase Latent Dimension**
```yaml
model:
  d_z: 128  # Double from 64 → 128 (or try 256)
```

**B. Cyclical KL Annealing**
Instead of linear warmup, use cyclical schedule:
- Start: β = 0
- Warmup to β_max over 20 epochs
- Then cycle: reduce β every 30 epochs for 10 epochs to allow exploration
- This prevents premature posterior collapse

**C. Free Bits Constraint**
Add minimum KL threshold to prevent collapse:
```python
kl_loss = torch.max(kl_divergence(mu, logvar), torch.tensor(0.5))
```

### Expected Impact
- Diversity: 0.04 → 0.15-0.25
- Sample variety: 2-3x improvement

---

## Issue #2: Poor Spatial Structure

### Problem
- Spatial correlation: 0.24 (should be > 0.7)
- Decoder has no skip connections to preserve spatial information
- No attention mechanism to focus on heavy rainfall regions

### Solution
**A. Add Skip Connections (U-Net style)**
Modify decoder to receive encoder features at matching resolutions:
```
EncoderY → conv4 (19×16) ──┐
                             ├→ Decoder conv1 (concat)
                             │
EncoderY → conv3 (39×33) ──┼──→ Decoder conv2 (concat)
                             │
EncoderY → conv2 (78×66) ──┴──→ Decoder conv3 (concat)
```

**B. Add Spatial Attention**
Before final output layer:
```python
class SpatialAttention(nn.Module):
    """Focus on regions with heavy precipitation"""
    def forward(self, x):
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attention = sigmoid(conv(concat([avg_pool, max_pool])))
        return x * attention
```

**C. Multi-Scale Loss**
Add reconstruction losses at intermediate resolutions:
```python
L_total = L_rec(156×132) + 0.3*L_rec(78×66) + 0.1*L_rec(39×33)
```

### Expected Impact
- Spatial correlation: 0.24 → 0.60-0.75
- Better preservation of coastal vs interior patterns

---

## Issue #3: Spurious Wetness (Wet Fraction Issue)

### Problem
- Generated samples show wetness everywhere (100%) vs real (75.8%)
- Model generates low-intensity "drizzle" to minimize reconstruction loss
- Requires post-hoc threshold filtering

### Solution
**A. Sparsity-Inducing Loss**
Add penalty for generating too many wet pixels:
```python
def sparsity_loss(y_hat, threshold=0.01):
    """Penalize low-intensity drizzle"""
    wet_mask = (y_hat > threshold).float()
    wet_fraction = wet_mask.mean()
    target_wet_fraction = 0.75  # From data analysis
    return torch.abs(wet_fraction - target_wet_fraction)
```

**B. Threshold-Aware Training**
Apply soft thresholding during training:
```python
def soft_threshold(x, threshold=0.01, temperature=0.1):
    """Smooth threshold function"""
    return x * torch.sigmoid((x - threshold) / temperature)
```

**C. Focal Loss for Dry Regions**
Increase weight on correctly predicting dry pixels:
```python
dry_mask = (y_true < 0.01)
dry_weight = 2.0  # Higher weight for dry pixels
weights = torch.where(dry_mask, dry_weight, base_weights)
```

### Expected Impact
- Wet fraction: 100% → 75-80%
- Eliminate need for post-hoc threshold filtering
- More realistic spatial patterns

---

## Issue #4: Training Stopped Too Early

### Problem
- Best model at epoch 66, stopped at 106
- MAE_tail still slowly improving (19.0 → 18.4)
- Early stopping patience=40 too conservative

### Solution
**A. Increase Early Stopping Patience**
```yaml
train:
  early_stopping_patience: 60  # Increase from 40
  early_stopping_min_delta: 0.01  # Require larger improvement
```

**B. Use ReduceLROnPlateau**
Instead of cosine annealing:
```yaml
scheduler:
  type: "plateau"
  factor: 0.5
  patience: 15
  min_lr: 1e-6
```

**C. Train Longer**
```yaml
train:
  epochs: 250  # Increase from 150
```

### Expected Impact
- Better convergence
- MAE_tail: 18.4 → 16-17 (estimated)

---

## Issue #5: Limited Model Capacity

### Problem
- base_filters=64 reduced from 96 to avoid overfitting
- May be too small for 156×132 high-res generation
- 110M parameters may be insufficient

### Solution
**A. Increase Model Capacity**
```yaml
model:
  base_filters: 96  # Restore from 64
  d_y: 384  # Increase from 256
  d_x: 96   # Increase from 64
```

**B. Add Residual Blocks**
Replace single conv layers with residual blocks:
```python
class ResBlock(nn.Module):
    def __init__(self, channels):
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)
```

**C. Increase Dropout**
To prevent overfitting with larger model:
```python
self.dropout = nn.Dropout(0.2)  # Increase from 0.1
```

### Expected Impact
- Better extreme value capture
- Improved spatial detail
- Parameters: 110M → ~180M

---

## Loss Function Improvements

### Current Loss
```python
L_total = L_weighted_rec + λ_mass * L_mass + β * L_kl
```

### Enhanced Loss
```python
L_total = L_weighted_rec +           # Intensity-weighted reconstruction
          λ_mass * L_mass +          # Mass conservation
          β * L_kl +                 # KL divergence (with free bits)
          λ_sparse * L_sparse +      # Sparsity (wet fraction)
          λ_multi * L_multiscale +   # Multi-scale reconstruction
          λ_spatial * L_spatial      # Spatial structure (gradient matching)
```

### New Loss Components

**1. Spatial Gradient Matching**
```python
def spatial_gradient_loss(y_true, y_hat):
    """Preserve spatial gradients (edges, fronts)"""
    grad_true_x = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    grad_true_y = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    grad_hat_x = y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]
    grad_hat_y = y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]

    loss_x = F.l1_loss(grad_hat_x, grad_true_x)
    loss_y = F.l1_loss(grad_hat_y, grad_true_y)
    return loss_x + loss_y
```

**2. Extreme Event Focal Loss**
Instead of just tail_boost, use focal loss:
```python
def extreme_focal_loss(y_true, y_hat, p99_threshold, gamma=2.0):
    """Focus on hard-to-predict extreme events"""
    extreme_mask = (y_true >= p99_threshold).float()
    error = torch.abs(y_hat - y_true)

    # Focal weight: (1 - e^(-error))^gamma
    focal_weight = (1 - torch.exp(-error)) ** gamma

    # Extra weight for extreme events
    weights = 1.0 + extreme_mask * focal_weight * 10.0

    return (weights * error).mean()
```

### Recommended Weights
```yaml
loss:
  type: "enhanced"
  scale: 20.0
  w_min: 0.1
  w_max: 3.0
  tail_boost: 2.0
  lambda_mass: 0.005
  lambda_sparse: 0.01
  lambda_multiscale: 0.1
  lambda_spatial: 0.05
  beta_kl: 0.005  # Reduce from 0.01
  warmup_epochs: 20
  cyclical_kl: true
```

---

## Training Improvements

### 1. Better Stratified Sampling
```yaml
train:
  min_heavy_fraction: 0.4  # Increase from 0.2
  # Ensures 40% of each batch is heavy events
```

### 2. Data Augmentation
```python
class AugmentedDataset:
    def __getitem__(self, idx):
        X, Y, S = self.load_sample(idx)

        # Random horizontal flip (Florida is roughly symmetric)
        if random.random() > 0.5:
            X = torch.flip(X, [-1])
            Y = torch.flip(Y, [-1])
            S = torch.flip(S, [-1])

        # Random noise (±2% of std)
        if self.training:
            noise = torch.randn_like(Y) * 0.02
            Y = Y + noise

        return X, Y, S
```

### 3. Gradient Accumulation
For effective larger batch size:
```python
accumulation_steps = 2  # Effective batch size = 64 * 2 = 128
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. ✓ Increase d_z: 64 → 128
2. ✓ Reduce beta_kl: 0.01 → 0.005
3. ✓ Add sparsity loss
4. ✓ Increase training: 150 → 250 epochs
5. ✓ Increase min_heavy_fraction: 0.2 → 0.4

**Expected improvement**: +30-40% in diversity, +20% in spatial correlation

### Phase 2: Architectural Changes (3-5 days)
1. Add skip connections to decoder
2. Add spatial attention
3. Increase base_filters: 64 → 96
4. Add multi-scale loss

**Expected improvement**: +40-50% in spatial correlation, better extreme capture

### Phase 3: Advanced Features (5-7 days)
1. Implement cyclical KL annealing
2. Add focal loss for extremes
3. Add spatial gradient loss
4. Implement data augmentation

**Expected improvement**: +20-30% overall quality, better training stability

---

## Expected Final Results

### Sample Quality Metrics (Estimated)
| Metric | Current | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| Diversity | 0.04 | 0.15 | 0.20 | 0.25 |
| Spatial Corr | 0.24 | 0.35 | 0.60 | 0.70 |
| Mass Bias | -1.28 | -0.50 | -0.20 | -0.10 |
| Wet Fraction | 100% | 85% | 78% | 76% |
| KS Statistic | 0.056 | 0.040 | 0.025 | 0.015 |

### SRDRN Performance (Estimated)
| Metric | Original | Current Aug_B | Phase 1 | Phase 2 | Phase 3 |
|--------|----------|---------------|---------|---------|---------|
| CSI (P95) test | 0.003 | 0.089 | 0.12 | 0.15 | 0.18 |
| SEDI (P95) test | 0.047 | 0.427 | 0.50 | 0.60 | 0.68 |
| Correlation test | 0.114 | 0.250 | 0.28 | 0.30 | 0.32 |
| GPD shape | 0.120 | 0.328 | 0.20 | 0.15 | 0.13 |

---

## Validation Strategy

After each phase:
1. Run `train_cvae.py` with new config
2. Generate samples with `sample_cvae.py`
3. Evaluate with `evaluate_samples.py`
4. Mix with real data using `prepare_augmented_data.py`
5. Retrain SRDRN
6. Compare metrics with previous versions

Keep all versions to compare:
- `config_v1_baseline.yaml` (current)
- `config_v2_phase1.yaml`
- `config_v3_phase2.yaml`
- `config_v4_phase3.yaml`

---

## Risk Mitigation

### Risk 1: Increased Model Size → Overfitting
- Mitigation: Increase dropout, use weight decay, monitor train/val gap

### Risk 2: Complex Loss → Training Instability
- Mitigation: Add losses incrementally, careful weight tuning, gradient clipping

### Risk 3: Longer Training → Computational Cost
- Mitigation: Use early stopping, mixed precision, efficient data loading

---

## Next Steps

1. **Immediate**: Implement Phase 1 changes (config_v2_phase1.yaml)
2. **This week**: Run full training with Phase 1 config
3. **Next week**: Evaluate results, decide on Phase 2
4. **Following week**: Implement and test Phase 2
5. **Month 1**: Complete all phases, final comparison

Would you like me to implement any of these phases now?
