# Quick Start: Phase 1 cVAE Training

## TL;DR - Start Training Now

```bash
cd /home/user/ugmentation_downscsling/cvae_augmentation

# 1. Test enhanced losses work
python losses_enhanced.py

# 2. Start training with Phase 1 config
python train_cvae.py --config config_v2_phase1.yaml
```

That's it! Training will run for up to 250 epochs with early stopping.

---

## What Changed from Baseline?

### 5 Key Improvements
1. **d_z: 64 → 128** (better diversity)
2. **β_kl: 0.01 → 0.005** (less posterior collapse)
3. **New losses**: Sparsity + Spatial gradient + Free bits KL
4. **More heavy samples**: 40% per batch (was 20%)
5. **Longer training**: 250 epochs (was 150)

### Expected Results
- **Diversity**: 0.04 → 0.15+ (4x improvement)
- **Wet fraction**: 100% → 85% (more realistic)
- **Spatial correlation**: 0.24 → 0.35+ (better patterns)
- **SRDRN CSI (P95)**: 0.089 → 0.12+ (better extreme detection)

---

## After Training

### Generate Samples
```bash
python sample_cvae.py \
    --config config_v2_phase1.yaml \
    --checkpoint outputs/cvae_phase1/checkpoints/cvae_best.pt
```

### Evaluate Quality
```bash
python evaluate_samples.py \
    --config config_v2_phase1.yaml \
    --checkpoint outputs/cvae_phase1/checkpoints/cvae_best.pt
```

### Prepare Augmented Data
```bash
# If wet_fraction looks good (75-85%)
python prepare_augmented_data.py \
    --config config_v2_phase1.yaml \
    --srdrn_data /scratch/user/u.hn319322/ondemand/Downscaling/mydata_corrected \
    --synth_dir outputs/cvae_phase1/synth \
    --output_dir outputs/cvae_phase1/final_data_augmented

# If wet_fraction still too high (>85%), add threshold filter
python prepare_augmented_data.py \
    --config config_v2_phase1.yaml \
    --srdrn_data /scratch/user/u.hn319322/ondemand/Downscaling/mydata_corrected \
    --synth_dir outputs/cvae_phase1/synth \
    --output_dir outputs/cvae_phase1/final_data_augmented \
    --threshold_filter \
    --threshold_value 0.01
```

### Retrain SRDRN
```bash
cd /home/user/ugmentation_downscsling/SRDRN
# Point to new augmented data and train
python train.py --data_dir ../cvae_augmentation/outputs/cvae_phase1/final_data_augmented
```

---

## Monitor Progress

Watch training:
```bash
tail -f outputs/cvae_phase1/logs/train_log.csv
```

Key metrics to watch:
- **val_MAE_tail**: Should decrease (early stopping metric)
- **val_wet_fraction**: Should approach 0.75
- **val_L_sparse**: Sparsity loss
- **beta**: KL weight (slowly increases from 0 to 0.005)

---

## Files You Need

✅ **config_v2_phase1.yaml** - New config
✅ **losses_enhanced.py** - New loss functions
✅ **train_cvae.py** - Updated to support "enhanced" loss

📖 **PHASE1_README.md** - Detailed guide
📖 **IMPROVEMENT_PLAN.md** - Full strategy

---

## Troubleshooting

**OOM Error?**
```yaml
# Reduce batch size in config_v2_phase1.yaml
train:
  batch_size: 32  # was 64
```

**Wet fraction not improving?**
```yaml
# Increase sparsity weight in config_v2_phase1.yaml
loss:
  lambda_sparse: 0.02  # was 0.01
```

**Training too slow?**
```yaml
# Increase workers in config_v2_phase1.yaml
train:
  num_workers: 4  # was 2
```

---

## Expected Training Time

- **~40-60 hours** on single GPU (V100/A100)
- **~150-200 epochs** before early stopping
- **~15s per epoch** with batch_size=64

Plan accordingly! 🕐

---

For detailed documentation, see **PHASE1_README.md**
