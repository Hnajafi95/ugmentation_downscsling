# Conditional VAE for Precipitation Downscaling Augmentation

A Conditional Variational Autoencoder (cVAE) for generating synthetic extreme precipitation events to augment training data for the SRDRN downscaling model.

## Overview

The cVAE learns to model `p(Y_hr | X_lr, S)` where:
- **X_lr**: Low-resolution GCM inputs (6, 13, 11)
- **Y_hr**: High-resolution precipitation (1, 156, 132)
- **S**: Static maps (land/sea mask, distance to coast)

The model uses an **intensity-weighted loss** (like SRDRN) that naturally emphasizes heavy precipitation without competing loss terms.

## Quick Start (3 Steps)

### Step 1: Prepare Data

```bash
python prepare_data.py \
    --input_dir /path/to/mydata_corrected \
    --output_dir /path/to/cvae_augmentation
```

This will:
- Split data into per-day files
- Create train/val/test splits
- Copy normalization parameters
- Compute P99 in mm/day space (for intensity weighting)

### Step 2: Train

```bash
python train_cvae.py --config config.yaml
```

Key hyperparameters in `config.yaml`:
```yaml
loss:
  scale: 36.6       # Intensity weighting scale (mm/day)
  w_min: 0.1        # Min weight (light rain)
  w_max: 2.0        # Max weight (heavy rain)
  tail_boost: 1.5   # Extra boost for P99+ pixels (increase to 2.0 or 2.5 for better tail)
  beta_kl: 0.01     # KL divergence weight
  warmup_epochs: 15 # KL warm-up period
```

### Step 3: Sample

```bash
python sample_cvae.py \
    --checkpoint outputs/cvae_simplified/checkpoints/cvae_best.pt \
    --mode posterior \
    --K 2 \
    --days heavy_only
```

## Loss Function

The simplified loss works in **mm/day space** (like SRDRN):

```
L_total = L_weighted_rec + lambda_mass * L_mass + beta * L_kl
```

Where `L_weighted_rec` is an intensity-weighted MAE:
- Light rain (1 mm/day): weight = 0.1
- Moderate rain (36.6 mm/day): weight = 1.0
- Heavy rain (73 mm/day): weight = 2.0
- P99+ with tail_boost=1.5: weight = 3.0 (30x more than light rain!)

This approach:
- Automatically emphasizes heavy precipitation
- Avoids multi-objective trade-offs
- Uses the same philosophy as SRDRN's MaskedWeightedMAEPrecip

## Files

```
cvae_augmentation/
├── config.yaml          # Configuration
├── prepare_data.py      # Data preparation
├── train_cvae.py        # Training script
├── sample_cvae.py       # Sampling script
├── model_cvae.py        # cVAE architecture
├── losses.py            # Intensity-weighted loss functions
├── data_io.py           # Data loading
├── utils_metrics.py     # Evaluation metrics
└── evaluate_samples.py  # Sample evaluation
```

## Hyperparameter Tuning

**If model underpredicts extremes** (high Tail MAE):
- Increase `tail_boost` (e.g., 2.0 or 2.5)
- Check that P99 in mm/day is correctly computed (~40-70 mm/day)

**If generated samples are too smooth**:
- Decrease `beta_kl`
- Use `mode=prior` sampling
- Increase sampling `temperature`

**If training is unstable**:
- Reduce learning rate
- Increase `warmup_epochs`
- Reduce `grad_clip`

## Expected Results

With the simplified loss, you should see:
- **MAE_all**: ~0.67 (normalized log-scale)
- **MAE_tail**: ~0.50-0.55 (normalized log-scale)
- Smooth convergence without oscillation
- Training converges in ~100-135 epochs

## Requirements

```bash
pip install torch torchvision numpy scipy pyyaml tqdm
```

Requires PyTorch with CUDA support.
