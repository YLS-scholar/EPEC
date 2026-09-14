#!/usr/bin/env bash
# ------------------------------------------------------------
# EPEC example runner
# This script runs the full EPEC analysis on the included
# example dataset (all_experiment_results.csv) and saves the
# results to the current directory (./output/).
# ------------------------------------------------------------

# Move to the directory where this script is located (example/)
cd "$(dirname "$0")"

echo "Installing dependencies (if needed)..."
pip install -r ../requirements.txt

echo "Running EPEC_full.py on the example dataset..."
python ../EPEC_full.py

echo "Analysis complete. Output files are saved in ./output/"
