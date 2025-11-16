#!/bin/bash
# Complete pipeline for cVAE-based precipitation augmentation

set -e  # Exit on error

echo "========================================="
echo "cVAE Precipitation Augmentation Pipeline"
echo "========================================="
echo ""

# Configuration
INPUT_DIR="/home/user/ugmentation_downscsling/mydata_corrected"
OUTPUT_DIR="/home/user/ugmentation_downscsling/cvae_augmentation"
CONFIG_FILE="${OUTPUT_DIR}/config.yaml"

# Step 1: Prepare data
echo "[Step 1/4] Preparing data..."
echo "  Converting SRDRN numpy files to per-day format..."
python "${OUTPUT_DIR}/prepare_data.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

if [ $? -eq 0 ]; then
    echo "  ✓ Data preparation complete"
else
    echo "  ✗ Data preparation failed"
    exit 1
fi
echo ""

# Step 2: Train cVAE
echo "[Step 2/4] Training cVAE model..."
echo "  This may take 2-4 hours on GPU..."
python "${OUTPUT_DIR}/train_cvae.py" --config "${CONFIG_FILE}"

if [ $? -eq 0 ]; then
    echo "  ✓ Training complete"
else
    echo "  ✗ Training failed"
    exit 1
fi
echo ""

# Step 3: Generate synthetic samples
echo "[Step 3/4] Generating synthetic samples..."
echo "  Generating K=2 samples for heavy precipitation days..."
python "${OUTPUT_DIR}/sample_cvae.py" \
    --config "${CONFIG_FILE}" \
    --mode posterior \
    --K 2 \
    --days heavy_only

if [ $? -eq 0 ]; then
    echo "  ✓ Sampling complete"
else
    echo "  ✗ Sampling failed"
    exit 1
fi
echo ""

# Step 4: Summary
echo "[Step 4/4] Summary"
echo "========================================="
echo "Pipeline complete!"
echo ""
echo "Outputs:"
echo "  - Data: ${OUTPUT_DIR}/data/"
echo "  - Model checkpoints: ${OUTPUT_DIR}/outputs/cvae/checkpoints/"
echo "  - Training log: ${OUTPUT_DIR}/outputs/cvae/logs/train_log.csv"
echo "  - Synthetic samples: ${OUTPUT_DIR}/outputs/cvae/synth/"
echo ""
echo "Next steps:"
echo "  1. Review training log: cat ${OUTPUT_DIR}/outputs/cvae/logs/train_log.csv"
echo "  2. Inspect synthetic samples in ${OUTPUT_DIR}/outputs/cvae/synth/"
echo "  3. Integrate synthetic data into SRDRN training"
echo "========================================="
