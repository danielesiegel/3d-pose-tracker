#!/bin/bash
# Quick start script for pose capture

# Activate conda environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "pose-tracker" ]; then
    echo "Activating conda environment..."
    eval "$(conda shell.bash hook)"
    conda activate pose-tracker
fi

# Run capture with provided arguments or defaults
python src/capture.py "$@"
