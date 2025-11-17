# Precipitation Downscaling with Data Augmentation

This repository contains a complete system for statistical downscaling of precipitation data using deep learning, with a focus on improving prediction of extreme precipitation events through data augmentation.

## Overview

The project addresses a critical challenge in climate downscaling: **extreme precipitation events are rare in training data, causing models to systematically underpredict heavy rainfall**. We solve this using a two-stage approach:

1. **SRDRN (Super-Resolution Deep Residual Network)**: A CNN-based model for downscaling low-resolution GCM data (13×11) to high-resolution precipitation fields (156×132)

2. **cVAE (Conditional Variational Autoencoder)**: Generates synthetic extreme precipitation samples to augment training data, addressing class imbalance

## Repository Structure

```
ugmentation_downscsling/
├── SRDRN/                          # Original SRDRN downscaling model
│   ├── preprocessing.py            # Data normalization and preparation
│   ├── Network.py                  # ResNet architecture
│   ├── train_original.py           # Training script
│   ├── Custom_loss_original.py     # Loss functions
│   └── org-postprocessing.py       # Convert predictions back to original scale
│
└── cvae_augmentation/              # cVAE-based data augmentation system
    ├── prepare_data.py             # Convert SRDRN data to per-day format
    ├── model_cvae.py               # cVAE architecture
    ├── data_io.py                  # Data loading utilities
    ├── losses.py                   # Extreme-weighted loss functions
    ├── utils_metrics.py            # Evaluation metrics
    ├── train_cvae.py               # Training script
    ├── sample_cvae.py              # Generate synthetic samples
    ├── config.yaml                 # Configuration
    ├── run_pipeline.sh             # Automated pipeline
    ├── requirements.txt            # Dependencies
    ├── README.md                   # Detailed documentation
    └── QUICKSTART.md               # Quick start guide
```

## Quick Start

### 1. Train SRDRN (Baseline Model)

First, train the baseline downscaling model:

```bash
cd SRDRN
python preprocessing.py          # Normalize data
python train_original.py         # Train SRDRN
python org-postprocessing.py     # Generate predictions
```

This will likely underpredict extreme precipitation events due to class imbalance.

### 2. Generate Synthetic Extreme Events (cVAE)

Use the cVAE system to generate synthetic heavy precipitation samples:

```bash
cd cvae_augmentation

# Option A: Run complete pipeline (recommended)
./run_pipeline.sh

# Option B: Step-by-step
python prepare_data.py --input_dir ../mydata_corrected
python train_cvae.py --config config.yaml      # ~2-4 hours on GPU
python sample_cvae.py --mode posterior --K 2 --days heavy_only
```

This generates synthetic samples in `outputs/cvae/synth/`.

### 3. Retrain SRDRN with Augmented Data

Integrate synthetic samples with real training data and retrain SRDRN with the augmented dataset for improved extreme precipitation prediction.

## cVAE System Features

The Conditional VAE learns to model `p(Y_hr | X_lr, S)` where:
- **X_lr**: Low-resolution GCM inputs (6 variables, 13×11 grid)
- **Y_hr**: High-resolution PRISM precipitation (156×132 grid)
- **S**: Static maps (land mask, distance to coast, elevation)

**Key innovations:**
- **Extreme-weighted loss**: Emphasizes accurate reconstruction of P95+ precipitation
- **Mass conservation**: Maintains total precipitation over land areas
- **No naive upsampling**: Conditioning via learned embeddings, not spatial upsampling
- **Posterior sampling**: Generates variations of specific heavy precipitation days
- **Quality filtering**: Discards unrealistic samples below mass threshold

**Architecture:**
```
EncoderX: (6, 13, 11) → h_X (64-dim embedding)
EncoderY: (1+S_ch, 156, 132) → h_Y (256-dim embedding)
LatentHead: [h_Y; h_X] → (μ, log σ²) for z (64-dim latent)
Decoder: [z; h_X] → Y_hr (1, 156, 132)
```

## Training Results

Based on HPC testing with ~14,500 days of data:
- **Heavy precipitation days identified**: 186 (1.3% of total)
- **Training convergence**: Epoch 4 (early stopping at epoch 19)
- **Best metrics**:
  - Validation Tail MAE: **0.32** (excellent - target was <1.0)
  - Validation L_rec: 6.22
  - Mass conservation: Well-maintained

The trained model successfully generates realistic extreme precipitation samples suitable for data augmentation.

## Expected Improvements

Using cVAE-augmented training data:
- ✅ Better representation of rare extreme precipitation events
- ✅ Improved skill for heavy rainfall prediction (P95+, P99+ events)
- ✅ Addresses class imbalance (only 1-5% of days are extreme events)
- ✅ Maintains physical consistency (mass conservation, spatial patterns)

## Requirements

**SRDRN:**
- Python 3.7+
- TensorFlow/Keras
- NumPy, SciPy

**cVAE:**
- Python 3.8+
- PyTorch ≥2.0 with CUDA support
- See `cvae_augmentation/requirements.txt`

## Documentation

- `cvae_augmentation/README.md`: Comprehensive cVAE documentation
- `cvae_augmentation/QUICKSTART.md`: Quick start guide with examples
- `cvae_augmentation/config.yaml`: Configuration options with detailed comments

## Citation

If you use this cVAE augmentation system in your research, please cite the original SRDRN work and acknowledge this augmentation framework.

## License

Research use. Modify and extend as needed for your specific application.
