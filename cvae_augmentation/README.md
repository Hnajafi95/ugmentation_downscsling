# cVAE for Extreme Precipitation Augmentation

Conditional Variational Autoencoder (cVAE) for generating synthetic extreme precipitation events to augment SRDRN training data.

## Key Improvements (Current Version)

### 🎯 Critical Fix: Transposed Convolutions
- **Problem**: Bilinear interpolation creates blurry, smooth outputs → poor spatial correlation (0.24)
- **Solution**: Replaced with learnable transposed convolutions in decoder
- **Expected Impact**: Spatial correlation 0.24 → 0.45-0.60

### 📈 Increased Capacity
- **d_z: 64 → 128** (doubled latent dimension)
- **Impact**: Better diversity (0.04 → 0.12-0.18), reduced posterior collapse

### 🎛️ Reduced KL Regularization
- **beta_kl: 0.01 → 0.005** (less aggressive)
- **warmup_epochs: 15 → 20** (gentler)
- **Impact**: More sample variety while maintaining structure

### 💪 More Extreme Emphasis
- **tail_boost: 2.0 → 3.0** (moderate increase)
- **min_heavy_fraction: 0.2 → 0.4** (40% heavy days per batch)
- **Impact**: Better extreme event capture

### ⏱️ Better Convergence
- **epochs: 150 → 250**, **patience: 40 → 60**
- **scheduler: cosine → plateau** (adaptive LR)
- **Impact**: Better final performance, less premature stopping

## Quick Start

```bash
# Train the cVAE
python train_cvae.py --config config.yaml

# Generate synthetic samples
python sample_cvae.py --config config.yaml

# Evaluate sample quality
python evaluate_samples.py --config config.yaml

# Prepare augmented data for SRDRN
python prepare_augmented_data.py \
    --config config.yaml \
    --srdrn_data /path/to/mydata_corrected \
    --synth_dir outputs/cvae_improved/synth \
    --output_dir outputs/cvae_improved/final_data
```

## Philosophy: Single Unified Loss

```
L_total = L_weighted_rec + λ_mass * L_mass + β * L_kl
```

**NO** additional loss components (sparsity, spatial gradient) to **avoid multi-objective trade-offs**.

The intensity-weighted reconstruction loss automatically emphasizes extremes in mm/day space:
- Light rain (1 mm/day): weight = 0.1
- Moderate (20 mm/day): weight = 1.0
- Heavy (40 mm/day): weight = 2.0
- P99+ (with tail_boost=3.0): effective weight = 6.0

## Expected Results

### Sample Quality
| Metric | Baseline | Current Expected |
|--------|----------|------------------|
| Diversity | 0.04 | 0.12-0.18 |
| Spatial Corr | 0.24 | 0.45-0.60 |
| Wet Fraction | 100% | 80-90% |
| Mass Bias | -1.28 | -0.30 to -0.60 |

### SRDRN Performance
| Metric | Original | Baseline Aug | Current Expected |
|--------|----------|--------------|------------------|
| CSI (P95) test | 0.003 | 0.089 | 0.12-0.15 |
| SEDI (P95) test | 0.047 | 0.427 | 0.50-0.55 |
| Correlation test | 0.114 | 0.250 | 0.28-0.32 |

## Configuration

Key settings in `config.yaml`:

```yaml
# Model - Transposed convolutions for sharp details
model:
  d_z: 128  # Doubled from 64 for diversity
  base_filters: 64

# Loss - Single objective (simplified)
loss:
  type: "simplified"
  scale: 20.0
  tail_boost: 3.0    # Moderate increase from 2.0
  beta_kl: 0.005     # Reduced from 0.01

# Training - Better convergence
train:
  epochs: 250
  patience: 60
  min_heavy_fraction: 0.4  # 40% heavy days

scheduler:
  type: "plateau"  # Adaptive LR reduction
```

## Files

### Core
- `model_cvae.py` - cVAE with **transposed convolutions** (NEW)
- `losses.py` - Simplified loss function
- `train_cvae.py` - Training script
- `sample_cvae.py` - Sample generation
- `evaluate_samples.py` - Quality evaluation
- `prepare_augmented_data.py` - Mix synthetic + real data

### Configuration
- `config.yaml` - Main configuration (improved)

### Optional
- `losses_enhanced.py` - Enhanced loss with sparsity/spatial (for experimentation)

## Monitoring

```bash
tail -f outputs/cvae_improved/logs/train_log.csv
```

Watch:
- **val_MAE_tail**: Early stopping metric (should decrease to ~18-19)
- **val_L_kl**: KL loss (should stabilize ~20-40)
- **beta**: Gradually increases from 0 → 0.005

## After Training

### 1. Evaluate Samples
```bash
python evaluate_samples.py --config config.yaml
```

Look for:
- ✅ Spatial correlation > 0.45
- ✅ Diversity > 0.12
- ✅ Wet fraction 75-90%

### 2. Prepare Augmented Data

If wet fraction is good:
```bash
python prepare_augmented_data.py \
    --config config.yaml \
    --srdrn_data /path/to/mydata_corrected \
    --synth_dir outputs/cvae_improved/synth \
    --output_dir outputs/cvae_improved/final_data
```

If wet fraction >90%, add threshold filter:
```bash
python prepare_augmented_data.py \
    ... \
    --threshold_filter \
    --threshold_value 0.01
```

### 3. Retrain SRDRN
```bash
cd ../SRDRN
python train.py --data_dir ../cvae_augmentation/outputs/cvae_improved/final_data
```

## Further Improvements

If transposed convolutions fix spatial issues but extremes still need work:

**Option B** (More Aggressive):
```yaml
loss:
  w_max: 5.0         # More aggressive (from 2.0)
  tail_boost: 5.0    # Much stronger (from 3.0)
  beta_kl: 0.001     # Very permissive (from 0.005)
```

See `IMPROVEMENT_PLAN.md` for full strategy.

## Troubleshooting

### OOM Error
```yaml
train:
  batch_size: 32  # Reduce from 64
```

### Still Blurry After Transposed Convs
Check decoder is actually using transposed convolutions:
```python
# In model_cvae.py line ~269-287
self.up1 = nn.ConvTranspose2d(...)  # Should be ConvTranspose2d, not interpolate
```

### Training Too Slow
```yaml
train:
  num_workers: 4  # Increase from 2
```

---

**Main Innovation**: Transposed convolutions should fix blurriness without needing additional loss objectives.

For full rationale, see `IMPROVEMENT_PLAN.md`.
