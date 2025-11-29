# 3D Pose Tracker

Real-time 3D pose tracking and visualization using MediaPipe BlazePose. Captures left arm joint positions (shoulder, elbow, wrist) with live 3D visualization and saves data to Parquet format.

## Features

- **Real-time 3D Pose Tracking**: Uses MediaPipe BlazePose for accurate 3D pose estimation
- **Dual Visualization**:
  - 2D video feed with pose overlay (OpenCV)
  - Real-time 3D spatial viewer (Open3D)
- **Optimized Performance**: Configurable frame rate, efficient processing pipeline
- **Data Persistence**: Saves joint positions to Parquet format with Snappy compression
- **Interactive Playback**: Replay and analyze captured motion with playback controls
- **Color-coded Joints**:
  - Left Shoulder: Red
  - Left Elbow: Green
  - Left Wrist: Blue

## Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Webcam** (for live capture)
- **macOS, Linux, or Windows**

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd manifold-learning
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate environment**:
   ```bash
   conda activate pose-tracker
   ```

### Environment Details

- **Python**: 3.10 (optimal compatibility)
- **Key Dependencies**:
  - MediaPipe 0.10.9 (BlazePose model)
  - OpenCV 4.8.1 (video capture)
  - Open3D 0.18.0 (3D visualization)
  - Pandas 2.0.3 + PyArrow 12.0.1 (Parquet I/O)
  - Click 8.1.7 (CLI framework)

## Usage

### Capture Mode

Start capturing pose data from your webcam:

```bash
python src/capture.py --frames 30
```

**Options**:
- `--frames`, `-f`: Frames per second to capture (default: 30)
- `--output`, `-o`: Output directory for saved data (default: `output/`)

**Controls**:
- Press `q` to quit and save data

**Example**:
```bash
# Capture at 60 FPS
python src/capture.py --frames 60

# Save to custom directory
python src/capture.py --frames 30 --output data/
```

### Viewer Mode

Replay and analyze saved pose data:

```bash
python src/viewer.py output/pose_data_20231129_153045.parquet
```

**Options**:
- `--speed`, `-s`: Playback speed multiplier (default: 1.0)
- `--loop`, `-l`: Loop playback continuously
- `--static`: Show static trajectory view

**Interactive Controls**:
- `SPACE`: Play/Pause
- `R`: Reset to beginning
- `Mouse`: Rotate view
- `ESC`: Exit

**Examples**:
```bash
# Play at 2x speed
python src/viewer.py output/pose_data.parquet --speed 2.0

# Loop continuously
python src/viewer.py output/pose_data.parquet --loop

# Static trajectory view
python src/viewer.py output/pose_data.parquet --static
```

## Project Structure

```
manifold-learning/
├── src/
│   ├── capture.py          # Main capture script
│   └── viewer.py           # Playback viewer
├── output/                 # Saved Parquet files (auto-created)
├── data/                   # Optional data directory
├── environment.yml         # Conda environment specification
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Data Format

Captured data is saved in Parquet format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | ISO 8601 timestamp |
| `frame` | int64 | Frame number |
| `shoulder_x` | float64 | Left shoulder X coordinate |
| `shoulder_y` | float64 | Left shoulder Y coordinate |
| `shoulder_z` | float64 | Left shoulder Z coordinate |
| `elbow_x` | float64 | Left elbow X coordinate |
| `elbow_y` | float64 | Left elbow Y coordinate |
| `elbow_z` | float64 | Left elbow Z coordinate |
| `wrist_x` | float64 | Left wrist X coordinate |
| `wrist_y` | float64 | Left wrist Y coordinate |
| `wrist_z` | float64 | Left wrist Z coordinate |

### Coordinate System

- MediaPipe uses a right-handed coordinate system
- Origin is at the center of the detected pose
- Units are in meters relative to hip center
- Z-axis points toward the camera

## Technical Details

### MediaPipe BlazePose

- **Model Complexity**: 2 (heavy model for best 3D accuracy)
- **Landmark Format**: 33 3D landmarks (COCO format)
- **Left Arm Indices**:
  - Shoulder: 11
  - Elbow: 13
  - Wrist: 15

### Performance Optimization

- Efficient frame processing with configurable FPS
- Non-blocking 3D visualization using threading
- Parquet with Snappy compression for fast I/O
- Minimal memory footprint

### Visualization

- **2D View**: OpenCV window (640x480)
- **3D View**: Open3D real-time renderer
  - Point size: 15.0
  - Line width: 5.0
  - Auto-rotating view support

## Troubleshooting

### Webcam Issues

If webcam doesn't open:
```bash
# Check available cameras (macOS/Linux)
ls /dev/video*

# Try different camera index in capture.py
cap = cv2.VideoCapture(1)  # Change from 0 to 1
```

### Performance Issues

If experiencing lag:
- Reduce frame rate: `--frames 15`
- Close other applications
- Ensure good lighting for better tracking

### Import Errors

If packages are missing:
```bash
conda activate pose-tracker
conda env update -f environment.yml
```

## References

### MediaPipe
- **Official Docs**: https://google.github.io/mediapipe/solutions/pose
- **BlazePose Paper**: [BlazePose: On-device Real-time Body Pose tracking](https://arxiv.org/abs/2006.10204)
- **Landmark Guide**: https://google.github.io/mediapipe/solutions/pose#pose-landmark-model-blazepose-ghum-3d

### Open3D
- **Documentation**: http://www.open3d.org/docs/
- **Visualization**: http://www.open3d.org/docs/latest/tutorial/visualization/visualization.html

### Apache Parquet
- **Format Spec**: https://parquet.apache.org/docs/
- **Python Integration**: https://arrow.apache.org/docs/python/parquet.html

## License

MIT License - feel free to use and modify for your projects.

## Contributing

Contributions welcome! Please open issues or pull requests.

## Future Enhancements

Potential additions:
- Full body tracking (all 33 landmarks)
- Multiple person support
- Export to additional formats (CSV, JSON, HDF5)
- Real-time metrics (joint angles, velocity)
- Web-based viewer using Three.js
- Video file input support
- GPU acceleration for processing

## Acknowledgments

- Google MediaPipe team for BlazePose
- Open3D contributors
- Apache Arrow/Parquet communities

---

**Disclaimer**: This project was created in part with AI-assisted code and is still a work in progress.
