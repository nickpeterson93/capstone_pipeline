#!/bin/bash
set -e

echo "============================================================"
echo "       Micron Capstone — Wafer Outlier Prediction Pipeline"
echo "============================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt -q
echo "[OK] Dependencies installed"
echo ""

# Run pipeline steps
echo "============================================================"
echo "Running Pipeline..."
echo "============================================================"
echo ""

python src/01_build_target.py
echo ""

python src/02_build_sensor_features.py
echo ""

python src/03_build_spc_features.py
echo ""

python src/04_build_lot_features.py
echo ""

python src/05_merge_features.py
echo ""

python src/06_train_model.py
echo ""

python src/07_evaluate.py
echo ""

echo "============================================================"
echo "[OK] Pipeline complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Features:  outputs/features/"
echo "  - Model:     outputs/models/catboost_model.cbm"
echo "  - Plots:     outputs/plots/"
echo "  - Summary:   outputs/evaluation_summary.txt"
echo ""
