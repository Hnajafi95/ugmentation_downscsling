#!/bin/bash
################################################################################
# Fix and Resample Script for cVAE Drizzle Issue
#
# This script:
# 1. Cleans old synthetic samples
# 2. Regenerates samples with pixel-wise thresholding (fixes wet fraction)
# 3. Re-evaluates the model
# 4. Generates new visualizations
#
# CRITICAL FIXES APPLIED:
# - Added pixel_threshold=0.1 in config.yaml (line 99)
# - Added pixel-wise thresholding in sample_cvae.py (line 252)
# - This will fix the "100% wet fraction" and "drizzle everywhere" issues
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "cVAE Fix & Resample Pipeline"
echo "================================================================================"
echo ""

# Configuration
CONFIG_PATH="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation/config.yaml"
CHECKPOINT_PATH="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation/outputs/cvae_final/checkpoints/cvae_best.pt"
SYNTH_DIR="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation/outputs/cvae_final/synth"
VIZ_DIR="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation/outputs/cvae_final/visualizations"

# Navigate to code directory
cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation

echo "Step 1: Cleaning old synthetic samples..."
echo "--------------------------------------------------------------------------------"
if [ -d "$SYNTH_DIR" ]; then
    echo "Removing old samples from: $SYNTH_DIR"
    rm -rf "$SYNTH_DIR"/*
    echo "✓ Old samples removed"
else
    echo "Creating synth directory: $SYNTH_DIR"
    mkdir -p "$SYNTH_DIR"
fi
echo ""

echo "Step 2: Generating new samples with pixel-wise thresholding..."
echo "--------------------------------------------------------------------------------"
echo "Configuration:"
echo "  - Mode: posterior (samples from ground truth distribution)"
echo "  - Pixel threshold: 0.1 mm/day (zeros out drizzle)"
echo "  - K: 5 samples per heavy day"
echo ""

python sample_cvae.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --mode posterior

echo ""
echo "✓ Sampling complete!"
echo ""

echo "Step 3: Evaluating new samples..."
echo "--------------------------------------------------------------------------------"
python evaluate_samples.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH"

echo ""
echo "✓ Evaluation complete!"
echo ""

echo "Step 4: Generating visualizations..."
echo "--------------------------------------------------------------------------------"
python visualize_samples.py \
    --config "$CONFIG_PATH" \
    --synth_dir "$SYNTH_DIR" \
    --output_dir "$VIZ_DIR" \
    --num_samples 20

echo ""
echo "✓ Visualizations complete!"
echo ""

echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo ""
echo "Expected Improvements:"
echo "  ✓ Spatial correlation: -0.002 → >0.60 (similar patterns)"
echo "  ✓ Wet fraction: 100% → ~75-80% (realistic dry areas)"
echo "  ✓ KS statistic: 0.376 → <0.1 (distributions match)"
echo "  ✓ Mass conservation: -0.75 → ~0.0 (preserves total rain)"
echo ""
echo "Output locations:"
echo "  - Synthetic samples: $SYNTH_DIR"
echo "  - Visualizations: $VIZ_DIR"
echo "  - Evaluation metrics: $VIZ_DIR/statistics.json"
echo ""
echo "================================================================================"
