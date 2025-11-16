# Conditional VAE for Precipitation Downscaling Augmentation

This directory contains a complete implementation of a **Conditional Variational Autoencoder (cVAE)** for generating synthetic high-resolution precipitation data to augment training for extreme precipitation events.

## Overview

The cVAE learns to model the distribution `p(Y_hr | X_lr, S)` where:
- **X_lr**: Low-resolution multi-variable GCM inputs (C, 13, 11)
- **Y_hr**: High-resolution precipitation from PRISM (1, 156, 132)
- **S**: Static maps (land/sea mask, distance to coast, etc.)

The model is specifically designed to generate **synthetic extreme precipitation cases** to address the class imbalance problem where heavy precipitation events are rare in the training data.

## Architecture

### Components

1. **EncoderX**: Encodes low-res inputs X_lr → embedding h_X
2. **EncoderY**: Encodes high-res Y_hr + statics S → embedding h_Y
3. **LatentHead**: Maps [h_Y; h_X] → latent distribution (μ, log σ²)
4. **Decoder**: Generates Y_hr from latent z conditioned on h_X

### Key Features

- **No naive upsampling**: Conditioning via embeddings, not spatial upsampling
- **Extreme-weighted loss**: Emphasizes accurate reconstruction of heavy precipitation
- **Mass conservation**: Penalizes deviation from total precipitation over land
- **KL warm-up**: Gradual introduction of KL divergence regularization

## Directory Structure

```
cvae_augmentation/
├── data/                          # Data directory (created by prepare_data.py)
│   ├── X_lr/                      # Per-day low-res inputs
│   │   ├── day_000000.npy
│   │   └── ...
│   ├── Y_hr/                      # Per-day high-res precipitation
│   │   ├── day_000000.npy
│   │   └── ...
│   ├── statics/                   # Static maps
│   │   ├── land_sea_mask.npy
│   │   ├── dist_to_coast_km.npy
│   │   └── ...
│   └── metadata/                  # Metadata files
│       ├── split.json             # Train/val/test split
│       ├── categories.json        # Day categories (dry/moderate/heavy)
│       ├── thresholds.json        # P95, P99 thresholds
│       └── H_W.json               # High-res dimensions
├── outputs/cvae/                  # Training outputs
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # Training logs (CSV)
│   └── synth/                     # Generated synthetic samples
├── config.yaml                    # Configuration file
├── prepare_data.py                # Data preparation script
├── data_io.py                     # Data loading utilities
├── model_cvae.py                  # cVAE model architecture
├── losses.py                      # Loss functions
├── utils_metrics.py               # Evaluation metrics
├── train_cvae.py                  # Training script
├── sample_cvae.py                 # Sampling script
└── README.md                      # This file
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision numpy scipy pyyaml tqdm

# Optional (for visualization)
pip install matplotlib
```

**Note**: This implementation requires PyTorch with CUDA support for GPU training.

## Usage

### Step 1: Prepare Data

Convert your existing SRDRN preprocessed data into per-day format:

```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation

python prepare_data.py \
    --input_dir /home/user/ugmentation_downscsling/mydata_corrected \
    --output_dir /home/user/ugmentation_downscsling/cvae_augmentation
```

This will:
- Split data into per-day files (X_lr, Y_hr)
- Create train/val/test splits (80/20 split of training data)
- Generate metadata (categories, thresholds, etc.)
- Create static maps (land mask, distance to coast)

**Output**:
- `data/X_lr/`: ~14,500 files (one per day)
- `data/Y_hr/`: ~14,500 files
- `data/statics/`: Static maps
- `data/metadata/`: JSON metadata files

### Step 2: Configure Training

Edit `config.yaml` to adjust hyperparameters:

```yaml
# Key parameters to adjust
train:
  epochs: 80
  batch_size: 6      # Reduce if OOM errors occur
  lr: 0.001

loss:
  lambda_ext: 3.0    # Weight for extreme precipitation (increase to emphasize more)
  lambda_mass: 0.1   # Weight for mass conservation
  beta_kl: 0.5       # KL divergence weight

model:
  d_z: 64            # Latent dimension
  base_filters: 64   # Increase for larger model capacity
```

### Step 3: Train cVAE

Train the model:

```bash
python train_cvae.py --config config.yaml
```

**Training time**: ~2-4 hours on a single GPU (NVIDIA V100/A100) for 80 epochs with batch_size=6.

**Outputs**:
- `outputs/cvae/checkpoints/cvae_best.pt`: Best model by validation loss
- `outputs/cvae/checkpoints/cvae_epoch_*.pt`: Periodic checkpoints
- `outputs/cvae/logs/train_log.csv`: Training metrics

**Monitor training**:
```bash
# View training log
tail -f outputs/cvae/logs/train_log.csv

# Or use pandas
import pandas as pd
df = pd.read_csv('outputs/cvae/logs/train_log.csv')
df[['epoch', 'train_loss', 'val_loss', 'val_MAE_tail']].tail(20)
```

### Step 4: Generate Synthetic Samples

Generate synthetic precipitation for heavy days:

```bash
# Generate samples for all heavy days (default: K=2 samples per day)
python sample_cvae.py \
    --config config.yaml \
    --mode posterior \
    --K 2 \
    --days heavy_only
```

**Options**:

- `--mode`: Sampling mode
  - `posterior`: Sample from posterior q(z|Y,X) (more faithful to specific days)
  - `prior`: Sample from prior p(z)=N(0,I) (more diverse but less realistic)

- `--K`: Number of samples per day (default: 2)

- `--days`: Which days to generate samples for
  - `heavy_only`: Only heavy precipitation days (default)
  - `split:val`: All validation days
  - `split:train`: All training days
  - `file:path/to/list.txt`: Custom list of day IDs

- `--min_threshold`: Discard samples with total mass < threshold × real mass (default: 0.1)

**Output**:
```
outputs/cvae/synth/
├── day_000123_sample_00.Y_hr_syn.npy    # Synthetic Y_hr
├── day_000123_sample_00.X_lr_ref.npy    # Corresponding X_lr
├── day_000123_sample_01.Y_hr_syn.npy
├── day_000123_sample_01.X_lr_ref.npy
└── ...
```

### Step 5: Use Synthetic Data for SRDRN Training

The generated synthetic samples can be used to augment your SRDRN training:

**Option A: Direct integration**
Modify your SRDRN training script to load synthetic samples from `outputs/cvae/synth/` and mix them with real data (e.g., 1 synthetic sample per real heavy day per epoch).

**Option B: Create augmented dataset**
Create a script to merge synthetic and real data into a new dataset:

```python
import numpy as np
from pathlib import Path

# Load real data
X_train_real = np.load('mydata_corrected/predictors_train_mean_std_separate.npy')
Y_train_real = np.load('mydata_corrected/obs_train_mean_std_single.npy')

# Load synthetic samples
synth_dir = Path('cvae_augmentation/outputs/cvae/synth')
X_synth_list = []
Y_synth_list = []

for X_file in sorted(synth_dir.glob('*.X_lr_ref.npy')):
    Y_file = X_file.parent / X_file.name.replace('X_lr_ref', 'Y_hr_syn')

    X_synth = np.load(X_file)  # (C, 13, 11)
    Y_synth = np.load(Y_file)  # (1, 156, 132)

    X_synth_list.append(X_synth)
    Y_synth_list.append(Y_synth)

# Stack and transpose to match SRDRN format
X_synth_all = np.stack(X_synth_list).transpose(0, 2, 3, 1)  # (N, 13, 11, C)
Y_synth_all = np.stack(Y_synth_list)[:, 0]  # (N, 156, 132)

# Combine with real data
X_train_aug = np.concatenate([X_train_real, X_synth_all], axis=0)
Y_train_aug = np.concatenate([Y_train_real, Y_synth_all], axis=0)

# Save augmented dataset
np.save('mydata_corrected/predictors_train_augmented.npy', X_train_aug)
np.save('mydata_corrected/obs_train_augmented.npy', Y_train_aug)
```

## Model Details

### Loss Function

The total loss is:

```
L_total = L_rec + λ_mass × L_mass + β × L_kl
```

Where:
- **L_rec**: Extreme-weighted MAE
  ```
  L_rec = λ_base × MAE_all + λ_ext × MAE_extreme
  ```
  - `MAE_all`: Mean absolute error over all pixels
  - `MAE_extreme`: MAE over pixels where Y ≥ P95

- **L_mass**: Mass conservation loss
  ```
  L_mass = |sum_land(Y_true) - sum_land(Y_hat)|
  ```

- **L_kl**: KL divergence
  ```
  L_kl = KL(q(z|Y,X) || p(z)) = -0.5 × sum(1 + log(σ²) - μ² - σ²)
  ```

### Hyperparameter Tuning

**If model underpredicts extremes**:
- Increase `lambda_ext` (e.g., 5.0 or 10.0)
- Decrease `lambda_mass` (to allow more flexibility)
- Increase model capacity: `base_filters=128`, `d_z=128`

**If generated samples are too smooth**:
- Decrease `beta_kl` (less regularization)
- Use `mode=prior` sampling (more diverse)
- Increase sampling `temperature` > 1.0

**If training is unstable**:
- Reduce learning rate (e.g., 0.0005)
- Increase `warmup_epochs` (e.g., 20)
- Reduce `batch_size` and increase `grad_clip`

## Evaluation

### Metrics

Training logs include:
- **MAE_all**: Mean absolute error over all pixels
- **MAE_tail**: MAE for Y ≥ P95 (extreme values)
- **RMSE_all**: Root mean squared error
- **mass_bias**: Difference in total precipitation

**Good model characteristics**:
- `val_MAE_tail` should be reasonably low (~0.5-1.0 in normalized log-scale)
- `val_mass_bias` should be close to 0
- `val_L_rec` should decrease steadily

### Visual Inspection

Create a simple visualization script to inspect synthetic samples:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load a synthetic sample
Y_syn = np.load('outputs/cvae/synth/day_000123_sample_00.Y_hr_syn.npy')[0]
X_lr = np.load('outputs/cvae/synth/day_000123_sample_00.X_lr_ref.npy')

# Load corresponding real data
Y_real = np.load('data/Y_hr/day_000123.npy')[0]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(Y_real, cmap='viridis')
axes[0].set_title('Real')
axes[1].imshow(Y_syn, cmap='viridis')
axes[1].set_title('Synthetic')
plt.savefig('comparison.png')
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `batch_size` in config.yaml (try 4 or 2)
2. Reduce model capacity: `base_filters=32`, `d_x=32`, `d_y=128`
3. Disable AMP: `amp: false` (uses less memory but slower)
4. Use gradient accumulation (requires code modification)

### Poor Generation Quality

**Symptoms**: Generated samples are too smooth or unrealistic

**Diagnosis**:
1. Check `val_MAE_tail` - should be < 1.0
2. Check `beta` - may be too high, over-regularizing
3. Inspect training curves for mode collapse (loss plateaus early)

**Solutions**:
1. Reduce `beta_kl` (e.g., 0.2 or 0.3)
2. Increase `lambda_ext` to emphasize extremes
3. Train longer (100-150 epochs)
4. Use `mode=prior` sampling for more diversity

### Model Underpredicts Extremes

**Symptoms**: Synthetic samples have lower peak values than real data

**Solutions**:
1. Increase `lambda_ext` significantly (e.g., 5.0, 10.0, or higher)
2. Add a "peak preservation" term to the loss (requires code modification)
3. Use temperature > 1.0 in sampling (e.g., 1.5)
4. Ensure P95 threshold is correctly loaded from `thresholds.json`

### NaN or Inf in Training

**Symptoms**: `loss=nan` in training logs

**Solutions**:
1. Check data for NaNs: Run `prepare_data.py` again
2. Reduce learning rate (e.g., 0.0005 or 0.0001)
3. Increase gradient clipping: `grad_clip: 0.5`
4. Disable AMP: `amp: false`

## Advanced Usage

### Custom Thresholds

To use a different threshold (e.g., P99 instead of P95):

Edit `losses.py`, line ~100 in `CVAELoss.__init__`:
```python
self.p95 = p99  # Use P99 instead
```

Or load P99 in `train_cvae.py`:
```python
p95 = thresholds['P99']  # Change this line
```

### Multi-GPU Training

Wrap model in DataParallel (requires code modification in `train_cvae.py`):
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Continue Training from Checkpoint

```python
# In train_cvae.py, after creating model and optimizer
checkpoint = torch.load('outputs/cvae/checkpoints/cvae_epoch_040.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Citation

If you use this code, please cite the original SRDRN work and acknowledge the cVAE architecture.

## License

This code is provided for research purposes. Modify and extend as needed for your specific application.

## Contact

For questions or issues, please refer to the main repository or contact the maintainers.
