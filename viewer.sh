#!/bin/bash
# Quick start script for pose viewer

# Activate conda environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "pose-tracker" ]; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate pose-tracker
fi

# Find latest parquet file if no argument provided
if [ $# -eq 0 ]; then
    LATEST=$(ls -t output/*.parquet 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo "No parquet files found in output/"
        echo "Usage: ./viewer.sh <path-to-parquet-file>"
        exit 1
    fi
    echo "Loading latest file: $LATEST"
    python src/viewer.py "$LATEST"
else
    python src/viewer.py "$@"
fi
