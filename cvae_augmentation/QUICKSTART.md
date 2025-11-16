# Quick Start Guide - cVAE Precipitation Augmentation

This guide will help you get started quickly with the cVAE system for generating synthetic extreme precipitation data.

## Prerequisites

You should have already run the SRDRN preprocessing script (`SRDRN/preprocessing.py`) which creates the directory `mydata_corrected/` with normalized numpy files.

## Installation

```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation
pip install -r requirements.txt
```

## Option 1: Run Complete Pipeline (Recommended)

The easiest way to run the entire pipeline:

```bash
./run_pipeline.sh
```

This will:
1. Prepare data (convert to per-day format)
2. Train the cVAE model (~2-4 hours on GPU)
3. Generate synthetic samples for heavy precipitation days
4. Display summary with output locations

## Option 2: Step-by-Step Execution

### Step 1: Prepare Data

```bash
python prepare_data.py \
    --input_dir /home/user/ugmentation_downscsling/mydata_corrected \
    --output_dir /home/user/ugmentation_downscsling/cvae_augmentation
```

**Expected output**:
- ~14,500 per-day files in `data/X_lr/` and `data/Y_hr/`
- Metadata files in `data/metadata/`
- Static maps in `data/statics/`

**Time**: ~5-10 minutes

### Step 2: Train cVAE

```bash
python train_cvae.py --config config.yaml
```

**Expected output**:
- Checkpoints in `outputs/cvae/checkpoints/`
- Training log in `outputs/cvae/logs/train_log.csv`

**Time**: 2-4 hours on single GPU (NVIDIA V100/A100)

**Monitor progress**:
```bash
# In another terminal
tail -f outputs/cvae/logs/train_log.csv
```

### Step 3: Generate Synthetic Samples

```bash
python sample_cvae.py \
    --config config.yaml \
    --mode posterior \
    --K 2 \
    --days heavy_only
```

**Expected output**:
- Synthetic samples in `outputs/cvae/synth/`
- Format: `day_XXXXXX_sample_YY.Y_hr_syn.npy` (synthetic precipitation)
- Format: `day_XXXXXX_sample_YY.X_lr_ref.npy` (corresponding low-res input)

**Time**: ~10-30 minutes depending on number of heavy days

## Verify Installation

Test that all modules work correctly:

```bash
# Test loss functions
python losses.py

# Test metrics
python utils_metrics.py
```

Both should print "All tests passed!" if everything is working.

## What to Expect

### Data Preparation Output

```
[1/6] Loading data...
   X_train shape: (12419, 13, 11, 7)
   Y_train shape: (12419, 156, 132)
   ...
[6/6] Summary
   Train: 9935 days
   Val:   2484 days
   Test:  1826 days
   Category distribution:
     dry                  : 8956 (61.4%)
     moderate             : 4835 (33.1%)
     heavy_coast          :  634 ( 4.3%)
     heavy_interior       :  160 ( 1.1%)
   Heavy days (for augmentation): 794 (5.4%)
```

### Training Output

```
Epoch 1/80
  Train - Loss: 2.3456, Rec: 2.1234, KL: 0.4567
  Val   - Loss: 2.1987, Rec: 1.9876, MAE: 0.8765, Tail: 1.2345
  Saved best model (val_L_rec=1.9876)

Epoch 40/80
  Train - Loss: 1.1234, Rec: 0.9876, KL: 0.2345
  Val   - Loss: 1.0567, Rec: 0.8901, MAE: 0.4321, Tail: 0.6789
  Saved best model (val_L_rec=0.8901)
```

Good indicators:
- Loss decreases steadily
- MAE_tail < 1.0 (in normalized log-scale)
- mass_bias close to 0

### Sampling Output

```
Generating 2 samples for 794 days...
Generating samples: 100%|████████████████| 794/794 [00:15<00:00, 51.23it/s]

Sampling complete!
Generated: 1588 samples
Discarded: 0 samples (below mass threshold)
Output directory: outputs/cvae/synth
```

## Next Steps

After generating synthetic samples, you can:

1. **Visualize samples** (create your own script):
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   Y_syn = np.load('outputs/cvae/synth/day_000123_sample_00.Y_hr_syn.npy')
   Y_real = np.load('data/Y_hr/day_000123.npy')

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   ax1.imshow(Y_real[0], cmap='viridis')
   ax1.set_title('Real')
   ax2.imshow(Y_syn[0], cmap='viridis')
   ax2.set_title('Synthetic')
   plt.savefig('comparison.png')
   ```

2. **Integrate with SRDRN training**:
   - Load synthetic samples from `outputs/cvae/synth/`
   - Mix with real training data (e.g., add 1 synthetic sample per real heavy day)
   - Retrain SRDRN model with augmented dataset

3. **Experiment with settings**:
   - Adjust `lambda_ext` in `config.yaml` to emphasize extremes more
   - Try different sampling modes (`prior` vs `posterior`)
   - Generate more samples per day (`--K 5`)

## Troubleshooting

### Out of Memory
**Error**: `CUDA out of memory`

**Solution**: Reduce `batch_size` in `config.yaml` (try 4 or 2)

### Poor Quality Samples
**Symptom**: Synthetic samples look too smooth

**Solution**:
- Increase `lambda_ext` in `config.yaml` (try 5.0 or 10.0)
- Reduce `beta_kl` (try 0.3)
- Train longer (100 epochs)

### No Heavy Days Found
**Error**: `Found 0 heavy precipitation days`

**Solution**: Check that `prepare_data.py` completed successfully and `data/metadata/categories.json` exists

## Getting Help

For detailed documentation, see:
- `README.md`: Full documentation
- `config.yaml`: Configuration options with comments
- Individual module files have docstrings

For common issues:
- Check the "Troubleshooting" section in `README.md`
- Verify all prerequisites are met
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## Performance Tips

**For faster training**:
- Use AMP (automatic mixed precision): `amp: true` in config
- Increase `batch_size` if you have enough GPU memory
- Use multiple workers: `num_workers: 4` in config

**For better quality**:
- Train longer (100-150 epochs)
- Increase model capacity: `base_filters: 128`, `d_z: 128`
- Emphasize extremes: `lambda_ext: 10.0`

## Citation

If you use this cVAE system in your research, please cite the original SRDRN paper and acknowledge this augmentation framework.
