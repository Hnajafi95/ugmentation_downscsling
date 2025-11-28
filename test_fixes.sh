#!/bin/bash
# Test script for cVAE fixes - Run after applying temperature fix
# This script will regenerate samples and evaluate them

set -e  # Exit on error

echo "========================================="
echo "Testing cVAE Fixes - Temperature=0.9"
echo "========================================="

# Configuration
CONFIG="config.yaml"
SYNTH_DIR="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation_v1/outputs/cvae_simplified/synth"
VIZ_DIR="/scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation_v1/outputs/cvae_simplified/visualizations_fixed"

# Step 1: Clean old samples
echo ""
echo "Step 1: Cleaning old samples..."
echo "----------------------------------------"
if [ -d "$SYNTH_DIR" ]; then
    echo "Removing old samples from: $SYNTH_DIR"
    rm -rf "$SYNTH_DIR"/*
    echo "✓ Cleaned"
else
    echo "Creating directory: $SYNTH_DIR"
    mkdir -p "$SYNTH_DIR"
fi

# Step 2: Regenerate samples with fixed temperature
echo ""
echo "Step 2: Generating new samples (temperature=0.9)..."
echo "----------------------------------------"
python sample_cvae.py \
    --config "$CONFIG" \
    --mode prior \
    --K 5 \
    --days "heavy_only"

echo "✓ Sample generation complete"

# Step 3: Evaluate samples
echo ""
echo "Step 3: Evaluating sample quality..."
echo "----------------------------------------"
python evaluate_samples.py \
    --config "$CONFIG" \
    --num_days 50 \
    --samples_per_day 5 \
    --output "outputs/cvae_simplified/evaluation_fixed.json"

echo "✓ Evaluation complete"

# Step 4: Visualize samples
echo ""
echo "Step 4: Creating visualizations..."
echo "----------------------------------------"
python visualize_samples.py \
    --config "$CONFIG" \
    --synth_dir "$SYNTH_DIR" \
    --output "$VIZ_DIR" \
    --num_samples 20 \
    --max_k 5

echo "✓ Visualization complete"

# Summary
echo ""
echo "========================================="
echo "Testing Complete!"
echo "========================================="
echo ""
echo "Check results:"
echo "  - Evaluation metrics: outputs/cvae_simplified/evaluation_fixed.json"
echo "  - Visualizations: $VIZ_DIR"
echo ""
echo "Expected improvements:"
echo "  - Synth mean: ~18-22 mm/day (was 4.83)"
echo "  - Synth max: ~120-160 mm/day (was 17.60)"
echo "  - Wet fraction: ~70-80% (was 97.9%)"
echo "  - Spatial correlation: >0.6 (was 0.33)"
echo ""
echo "If results still show underprediction, try temperature=1.0 or 1.1"
echo "See FIXES_SUMMARY.md for details."
