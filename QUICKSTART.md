# Quick Start Guide

## Installation (One-time setup)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate pose-tracker
```

## Capture Pose Data

### Method 1: Using Python directly
```bash
python src/capture.py --frames 30
```

### Method 2: Using convenience script
```bash
./capture.sh --frames 30
```

**Press 'q' to quit and save!**

## View Saved Data

### Method 1: Using Python directly
```bash
python src/viewer.py output/pose_data_TIMESTAMP.parquet
```

### Method 2: Using convenience script (auto-loads latest)
```bash
./viewer.sh
```

## Common Options

### Capture
- `--frames 60` - Capture at 60 FPS (default: 30)
- `--output data/` - Save to custom directory
- `--no-3d` - Disable 3D viewer (for display issues)

### Viewer
- `--speed 2.0` - Play at 2x speed
- `--loop` - Loop playback
- `--static` - Show trajectory instead of animation

## Examples

```bash
# High-speed capture
python src/capture.py --frames 60

# Review with 2x playback speed
python src/viewer.py output/pose_data.parquet --speed 2.0

# Static trajectory view
python src/viewer.py output/pose_data.parquet --static
```

## Troubleshooting

**3D display errors (macOS)?**
```bash
python src/capture.py --frames 30 --no-3d
```

**Webcam not found?**
- Check camera permissions in System Preferences (macOS)
- Try a different camera index in capture.py line 189

**Dependencies missing?**
```bash
conda activate pose-tracker
conda env update -f environment.yml
```

**Low FPS?**
- Reduce `--frames` value
- Ensure good lighting
- Close other apps
- Try `--no-3d` flag
