#!/bin/bash
################################################################################
# Quick fix for NaN correlation issue
#
# Problem: pixel_threshold=2.0 is too aggressive for moderate-heavy days
# Solution: Reduce to 1.0 mm/day
################################################################################

set -e

echo "================================================================================"
echo "Fixing NaN Correlation Issue"
echo "================================================================================"
echo ""
echo "Problem: pixel_threshold=2.0 mm/day removes ALL precipitation on moderate days"
echo "Solution: Reduced to 1.0 mm/day"
echo ""

cd /scratch/user/u.hn319322/ondemand/Downscaling/cVAE_augmentation

# Clean old samples
echo "Step 1: Cleaning old samples..."
rm -rf outputs/cvae_final/synth/*
echo "✓ Done"
echo ""

# Regenerate with threshold=1.0
echo "Step 2: Regenerating samples with pixel_threshold=1.0..."
python sample_cvae.py --config config.yaml --mode posterior
echo ""

# Evaluate
echo "Step 3: Evaluating..."
python evaluate_samples.py --config config.yaml
echo ""

# Visualize
echo "Step 4: Visualizing..."
python visualize_samples.py --config config.yaml --num_samples 20
echo ""

echo "================================================================================"
echo "Done!"
echo "================================================================================"
echo ""
echo "Expected results:"
echo "  - Spatial correlation: Should be > 0.50 (no more NaNs!)"
echo "  - Wet fraction: ~0.70-0.75 (slightly higher than before)"
echo "  - Statistics should all be valid numbers"
echo ""
echo "Check results:"
echo "  cat outputs/cvae_final/visualizations/statistics.json"
echo "================================================================================"
