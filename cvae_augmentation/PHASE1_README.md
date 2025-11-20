# Phase 1: cVAE Quick Wins Implementation

## Overview

This is the **Phase 1 implementation** of the cVAE improvement plan, focusing on quick wins that can be implemented immediately without major architectural changes.

### Key Improvements

1. **Doubled Latent Dimension**: d_z: 64 → 128 (better diversity)
2. **Reduced KL Weight**: β: 0.01 → 0.005 (less aggressive regularization)
3. **Enhanced Loss Function**: Added sparsity and spatial gradient losses
4. **Free Bits KL**: Prevents posterior collapse (0.5 nats/dim)
5. **Longer Training**: 150 → 250 epochs with patience=60
6. **More Heavy Samples**: min_heavy_fraction: 0.2 → 0.4 (40% per batch)
7. **Stronger Extreme Emphasis**: tail_boost: 2.0 → 2.5

### Expected Improvements

| Metric | Baseline | Phase 1 Target |
|--------|----------|----------------|
| Diversity | 0.04 | 0.15+ |
| Spatial Correlation | 0.24 | 0.35+ |
| Mass Bias | -1.28 | -0.50 |
| Wet Fraction | 100% | ~85% |

---

## Files Modified/Created

### New Files
1. **config_v2_phase1.yaml** - Enhanced configuration with all Phase 1 changes
2. **losses_enhanced.py** - New loss functions (sparsity, spatial gradient, free bits KL)
3. **IMPROVEMENT_PLAN.md** - Comprehensive improvement strategy document
4. **PHASE1_README.md** - This file

### Modified Files
1. **train_cvae.py** - Added support for "enhanced" loss type and new metrics logging

---

## Installation & Setup

### 1. Verify Files

Ensure you have all required files:
```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation

# Check new files exist
ls -lh config_v2_phase1.yaml
ls -lh losses_enhanced.py
ls -lh IMPROVEMENT_PLAN.md
ls -lh PHASE1_README.md

# Check modified files
ls -lh train_cvae.py
```

### 2. Test Enhanced Loss Functions

Before running full training, test the new loss functions:
```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation
python losses_enhanced.py
```

Expected output:
```
Testing enhanced losses...

1. Sparsity loss:
   Loss: 0.XXXX
   Wet fraction: 0.XXXX

2. Spatial gradient loss:
   Loss: 0.XXXX

3. KL with free bits:
   Loss: XX.XXXX
   Dims at free bits: XXX

4. Enhanced criterion:
   Total loss: XX.XXXX
   L_rec: XX.XXXX
   L_sparse: 0.XXXX
   L_spatial: 0.XXXX
   L_kl: XX.XXXX
   Wet fraction: 0.XXXX

✓ All tests passed!
```

### 3. Verify Data Paths

Update paths in `config_v2_phase1.yaml` if needed:
```yaml
data_root: "/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation"
outputs_root: "/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation/outputs/cvae_phase1"
```

---

## Running Phase 1 Training

### Option 1: Interactive Session (Testing)

For quick testing on a small subset:
```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation

# Run for just 5 epochs to verify everything works
python train_cvae.py --config config_v2_phase1.yaml
```

### Option 2: SLURM Batch Job (Full Training)

Create a batch script `train_phase1.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=cvae_phase1
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules
module load python/3.9
module load cuda/11.8

# Activate environment
source /path/to/your/venv/bin/activate

# Run training
cd /home/user/ugmentation_downscsling/cvae_augmentation
python train_cvae.py --config config_v2_phase1.yaml
```

Submit the job:
```bash
sbatch train_phase1.sh
```

Monitor progress:
```bash
# Check job status
squeue -u $USER

# Watch output
tail -f logs/phase1_*.out

# Monitor training log
tail -f outputs/cvae_phase1/logs/train_log.csv
```

---

## Monitoring Training

### 1. Training Logs

The training log is saved to:
```
outputs/cvae_phase1/logs/train_log.csv
```

Key columns to watch:
- **val_MAE_tail**: Main metric for early stopping (lower is better)
- **val_wet_fraction**: Should approach 0.75 (currently 1.0 in baseline)
- **val_L_sparse**: Sparsity loss (should decrease)
- **val_L_spatial**: Spatial gradient loss (should decrease)
- **beta**: KL weight (should gradually increase from 0 to 0.005)

### 2. Checkpoints

Checkpoints are saved to:
```
outputs/cvae_phase1/checkpoints/
├── cvae_best.pt           # Best model (lowest MAE_tail)
├── cvae_epoch_020.pt      # Periodic checkpoint every 20 epochs
├── cvae_epoch_040.pt
└── ...
```

### 3. Visualize Training Progress

```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation

# Create plots (you may need to create this script)
python plot_training.py --log outputs/cvae_phase1/logs/train_log.csv
```

Example Python script `plot_training.py`:
```python
import pandas as pd
import matplotlib.pyplot as plt
import sys

log_file = sys.argv[2]
df = pd.read_csv(log_file)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss
axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train')
axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val')
axes[0, 0].set_title('Total Loss')
axes[0, 0].legend()

# MAE_tail (early stopping metric)
axes[0, 1].plot(df['epoch'], df['val_MAE_tail'])
axes[0, 1].set_title('Val MAE Tail')

# Wet fraction
axes[0, 2].plot(df['epoch'], df['val_wet_fraction'])
axes[0, 2].axhline(y=0.75, color='r', linestyle='--', label='Target')
axes[0, 2].set_title('Val Wet Fraction')
axes[0, 2].legend()

# KL loss
axes[1, 0].plot(df['epoch'], df['train_L_kl'], label='Train')
axes[1, 0].plot(df['epoch'], df['val_L_kl'], label='Val')
axes[1, 0].set_title('KL Loss')
axes[1, 0].legend()

# Sparsity loss
axes[1, 1].plot(df['epoch'], df['val_L_sparse'])
axes[1, 1].set_title('Sparsity Loss')

# Spatial loss
axes[1, 2].plot(df['epoch'], df['val_L_spatial'])
axes[1, 2].set_title('Spatial Gradient Loss')

plt.tight_layout()
plt.savefig('outputs/cvae_phase1/training_progress.png', dpi=150)
print("Plot saved to outputs/cvae_phase1/training_progress.png")
```

---

## After Training Completes

### 1. Generate Synthetic Samples

```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation

python sample_cvae.py \
    --config config_v2_phase1.yaml \
    --checkpoint outputs/cvae_phase1/checkpoints/cvae_best.pt \
    --mode posterior \
    --K 5 \
    --days heavy_only
```

Expected output:
```
================================================================================
cVAE Sampling
================================================================================
...
Generating 5 samples for 186 days...
Generated: 930 samples
Output directory: outputs/cvae_phase1/synth
```

### 2. Evaluate Sample Quality

```bash
python evaluate_samples.py \
    --config config_v2_phase1.yaml \
    --checkpoint outputs/cvae_phase1/checkpoints/cvae_best.pt \
    --split train \
    --num_days 100 \
    --samples_per_day 5
```

Expected improvements:
```
EVALUATION SUMMARY
================================================================================

1. SPATIAL PATTERN CORRELATION
   Mean correlation: 0.35-0.40  (was 0.24)
   Status: ⚠ FAIR

2. EXTREME VALUE DISTRIBUTION
   KS statistic:     0.03-0.04  (was 0.056)
   Status: ✓ GOOD

3. MASS CONSERVATION
   Mean bias:        -0.40 to -0.60  (was -1.28)
   Status: ⚠ FAIR

4. SAMPLE DIVERSITY
   Mean diversity:   0.12-0.18  (was 0.04)
   Status: ⚠ FAIR

Quality score: 2-3/4
Status: ⚠ FAIR - Improved but still needs work
```

### 3. Visualize Samples

```bash
python visualize_samples.py \
    --config config_v2_phase1.yaml \
    --checkpoint outputs/cvae_phase1/checkpoints/cvae_best.pt \
    --num_days 20
```

Check wet fraction in statistics:
```
Metric                  Real        Synth       Diff %
-------------------------------------------------------
wet_fraction           0.758        0.85       +12%    # Should be much better!
```

### 4. Prepare Augmented Data

If satisfied with sample quality, prepare augmented dataset:

```bash
# Option A: Use raw samples (if wet_fraction ≈ 0.75-0.80)
python prepare_augmented_data.py \
    --config config_v2_phase1.yaml \
    --srdrn_data /scratch/user/u.hn319322/ondemand/Downscaling/mydata_corrected \
    --synth_dir outputs/cvae_phase1/synth \
    --output_dir outputs/cvae_phase1/final_data_augmented

# Option B: With threshold filtering (if wet_fraction still > 0.85)
python prepare_augmented_data.py \
    --config config_v2_phase1.yaml \
    --srdrn_data /scratch/user/u.hn319322/ondemand/Downscaling/mydata_corrected \
    --synth_dir outputs/cvae_phase1/synth \
    --output_dir outputs/cvae_phase1/final_data_augmented \
    --threshold_filter \
    --threshold_value 0.01
```

### 5. Retrain SRDRN

```bash
cd /home/user/ugmentation_downscsling/SRDRN

# Update config to point to new augmented data
# Then train SRDRN
python train.py --data_dir /path/to/outputs/cvae_phase1/final_data_augmented
```

### 6. Evaluate SRDRN Performance

After SRDRN training completes, run metrics:
```bash
python calculate_metrics.py \
    --model_dir /path/to/srdrn/checkpoint \
    --output metrics_phase1.xlsx
```

Compare with baseline:
- Original SRDRN
- Augmented with baseline cVAE
- Augmented with Phase 1 cVAE

Expected improvements in `metrics_phase1.xlsx`:
- CSI (P95) test: 0.089 → 0.10-0.13
- SEDI (P95) test: 0.427 → 0.45-0.52
- Correlation test: 0.250 → 0.26-0.29

---

## Troubleshooting

### Issue 1: Out of Memory (OOM)

If you get CUDA OOM errors:

1. Reduce batch size in config:
```yaml
train:
  batch_size: 32  # Reduce from 64
```

2. Reduce model capacity:
```yaml
model:
  d_z: 96  # Reduce from 128
  base_filters: 48  # Reduce from 64
```

### Issue 2: Training is Very Slow

1. Check GPU utilization:
```bash
nvidia-smi -l 1
```

2. Increase num_workers (if CPU is bottleneck):
```yaml
train:
  num_workers: 4  # Increase from 2
```

3. Enable mixed precision (should already be on):
```yaml
train:
  amp: true
```

### Issue 3: Wet Fraction Not Improving

If `val_wet_fraction` stays at ~1.0:

1. Increase sparsity weight:
```yaml
loss:
  lambda_sparse: 0.02  # Increase from 0.01
```

2. Check target wet fraction matches data:
```python
# Verify target is correct
import numpy as np
y_train = np.load('data/per_day/train/day_0000_Y.npy')
land_mask = np.load('data/static/land_sea_mask.npy')[0]
wet_fraction = (y_train[0] > 0.01).sum() / land_mask.sum()
print(f"Actual wet fraction: {wet_fraction}")
```

### Issue 4: KL Loss Explodes

If `train_L_kl` > 100:

1. Reduce learning rate:
```yaml
train:
  lr: 0.0001  # Reduce from 0.0003
```

2. Increase KL warmup:
```yaml
loss:
  warmup_epochs: 30  # Increase from 20
```

### Issue 5: Sample Diversity Still Low

If diversity < 0.10 after Phase 1:

1. Further reduce KL weight:
```yaml
loss:
  beta_kl: 0.002  # Reduce from 0.005
```

2. Increase free bits:
```yaml
loss:
  free_bits: 1.0  # Increase from 0.5
```

3. Increase latent dimension:
```yaml
model:
  d_z: 192  # Increase from 128
```

---

## Comparison with Baseline

### Training Time
- **Baseline**: ~106 epochs × 15s = 27 minutes
- **Phase 1**: ~150-250 epochs × 15s = 38-63 minutes
- **Increase**: +40-130%

### Sample Quality
| Metric | Baseline | Phase 1 Target | Improvement |
|--------|----------|----------------|-------------|
| Diversity | 0.04 | 0.15 | +275% |
| Spatial Corr | 0.24 | 0.35 | +46% |
| Wet Fraction | 100% | 85% | +15% |
| Mass Bias | -1.28 | -0.50 | +61% |

### SRDRN Performance (Estimated)
| Metric | Original | Baseline Aug | Phase 1 Aug | Improvement |
|--------|----------|--------------|-------------|-------------|
| CSI (P95) test | 0.003 | 0.089 | 0.12 | +35% |
| SEDI (P95) test | 0.047 | 0.427 | 0.50 | +17% |
| Correlation test | 0.114 | 0.250 | 0.28 | +12% |

---

## Next Steps

### If Phase 1 Results are Good (Quality Score ≥ 2/4)
1. Proceed to **Phase 2**: Architectural improvements
   - Add skip connections (U-Net decoder)
   - Add spatial attention
   - Increase model capacity

### If Phase 1 Results are Poor (Quality Score < 2/4)
1. Debug loss weights:
   - Try different lambda_sparse values [0.005, 0.01, 0.02, 0.05]
   - Try different lambda_spatial values [0.01, 0.02, 0.05]
   - Try different tail_boost values [2.0, 2.5, 3.0]

2. Experiment with latent dimension:
   - Try d_z = [96, 128, 192, 256]
   - Check diversity vs reconstruction trade-off

3. Adjust KL scheduling:
   - Try longer warmup: 30-50 epochs
   - Try lower final beta: 0.002-0.003
   - Try higher free bits: 0.8-1.2

---

## Questions?

For detailed explanation of improvements, see:
- **IMPROVEMENT_PLAN.md** - Full strategy and justification
- **losses_enhanced.py** - Implementation details with comments
- **config_v2_phase1.yaml** - All hyperparameter choices

Good luck with Phase 1! 🚀
