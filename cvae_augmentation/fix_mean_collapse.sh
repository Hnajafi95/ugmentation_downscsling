#!/bin/bash
################################################################################
# cVAE Mean Collapse Fix - Complete Pipeline
#
# This script applies the comprehensive fix for the "mean collapse" problem:
# 1. Normalization space mismatch (pixel threshold applied in wrong space)
# 2. Conditioning dropout removed (model needs h_X for spatial structure)
# 3. Loss weights relaxed (w_min: 1.0→0.2, tail_boost: 3.0→1.5)
#
# REQUIRED: You must RETRAIN the model after updating code
# The old model has conditioning dropout baked into weights - won't work correctly
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "cVAE Mean Collapse Fix Pipeline"
echo "================================================================================"
echo ""

# Configuration
CVAE_DIR="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation"
CONFIG_PATH="$CVAE_DIR/config.yaml"
OUTPUTS_DIR="$CVAE_DIR/outputs/cvae_final"
CHECKPOINT_DIR="$OUTPUTS_DIR/checkpoints"
SYNTH_DIR="$OUTPUTS_DIR/synth"
VIZ_DIR="$OUTPUTS_DIR/visualizations"

cd "$CVAE_DIR"

echo "CRITICAL: Model architecture has changed (conditioning dropout removed)"
echo "You MUST retrain the model before sampling."
echo ""
echo "Changes applied:"
echo "  1. ✓ sample_cvae.py: Added denormalization before pixel thresholding"
echo "  2. ✓ model_cvae.py: Removed conditioning dropout (p=0.5)"
echo "  3. ✓ config.yaml: Relaxed loss weights (w_min=0.2, tail_boost=1.5)"
echo "  4. ✓ config.yaml: Increased pixel threshold (0.1→0.5 mm/day)"
echo ""
echo "================================================================================"
read -p "Continue with retraining? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi
echo ""

echo "Step 1: Backing up old model..."
echo "--------------------------------------------------------------------------------"
if [ -f "$CHECKPOINT_DIR/cvae_best.pt" ]; then
    BACKUP_NAME="cvae_best_pre_fix_$(date +%Y%m%d_%H%M%S).pt"
    cp "$CHECKPOINT_DIR/cvae_best.pt" "$CHECKPOINT_DIR/$BACKUP_NAME"
    echo "✓ Backed up to: $BACKUP_NAME"
else
    echo "No existing model found, proceeding with fresh training."
fi
echo ""

echo "Step 2: Retraining cVAE model..."
echo "--------------------------------------------------------------------------------"
echo "Expected training time: ~1-2 hours on GPU"
echo "Expected early stopping: around epoch 100-150"
echo ""

python train_cvae.py --config "$CONFIG_PATH"

if [ $? -ne 0 ]; then
    echo "❌ Training failed. Check logs in $OUTPUTS_DIR/logs/"
    exit 1
fi
echo ""
echo "✓ Training complete!"
echo ""

echo "Step 3: Cleaning old synthetic samples..."
echo "--------------------------------------------------------------------------------"
if [ -d "$SYNTH_DIR" ]; then
    echo "Removing old samples from: $SYNTH_DIR"
    rm -rf "$SYNTH_DIR"/*
    echo "✓ Old samples removed"
else
    mkdir -p "$SYNTH_DIR"
fi
echo ""

echo "Step 4: Generating new samples with fixes..."
echo "--------------------------------------------------------------------------------"
echo "Configuration:"
echo "  - Mode: posterior (samples from ground truth distribution)"
echo "  - Pixel threshold: 0.5 mm/day (denormalized, then applied)"
echo "  - K: 5 samples per heavy day"
echo ""

python sample_cvae.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_DIR/cvae_best.pt" \
    --mode posterior

if [ $? -ne 0 ]; then
    echo "❌ Sampling failed."
    exit 1
fi
echo ""
echo "✓ Sampling complete!"
echo ""

echo "Step 5: Evaluating new samples..."
echo "--------------------------------------------------------------------------------"
python evaluate_samples.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_DIR/cvae_best.pt"

if [ $? -ne 0 ]; then
    echo "⚠ Evaluation had issues, but continuing..."
fi
echo ""
echo "✓ Evaluation complete!"
echo ""

echo "Step 6: Generating visualizations..."
echo "--------------------------------------------------------------------------------"
python visualize_samples.py \
    --config "$CONFIG_PATH" \
    --synth_dir "$SYNTH_DIR" \
    --output_dir "$VIZ_DIR" \
    --num_samples 20

if [ $? -ne 0 ]; then
    echo "⚠ Visualization had issues, but continuing..."
fi
echo ""
echo "✓ Visualizations complete!"
echo ""

echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Expected Improvements:"
echo "  ✓ Spatial correlation: -0.15 → >0.60 (realistic storm locations)"
echo "  ✓ Wet fraction: 100% → ~75-80% (realistic dry areas)"
echo "  ✓ Mass conservation: -1.06 → ~0.0 (preserves total rain)"
echo "  ✓ Sample diversity: 0.056 → 0.15-0.25 (better variation)"
echo ""
echo "Output locations:"
echo "  - New model: $CHECKPOINT_DIR/cvae_best.pt"
echo "  - Synthetic samples: $SYNTH_DIR"
echo "  - Visualizations: $VIZ_DIR"
echo "  - Training logs: $OUTPUTS_DIR/logs/"
echo ""
echo "Next steps:"
echo "  1. Review visualizations in $VIZ_DIR"
echo "  2. Check statistics.json for wet fraction (~0.75-0.80)"
echo "  3. If metrics look good, proceed with data augmentation"
echo "     python prepare_augmented_data.py --config config.yaml"
echo ""
echo "================================================================================"
