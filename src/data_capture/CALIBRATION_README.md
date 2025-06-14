# ToF-RGB Camera Calibration System

This system provides robust ArUco-based calibration for mapping between RGB camera and Time-of-Flight (ToF) camera coordinates, optimized for production use.

## System Overview

The system uses a **pre-computed calibration approach** instead of real-time ArUco detection for optimal performance:

1. **One-time setup**: Run ArUco calibration to generate `camera_calibration.json`
2. **Production use**: Main application loads pre-computed calibration for fast landmark transformation
3. **Efficient processing**: RGB → MediaPipe → Transform landmarks to ToF coordinates

## Quick Start

### 1. Initial Calibration Setup

```bash
# Generate ArUco marker and create calibration
python setup_calibration.py
```

This will:
- Generate ArUco marker ID 77 (`aruco_marker_77_calibration.png`)
- Create sample calibration data (`camera_calibration.json`)
- Provide instructions for real camera setup

### 2. Run Main Application

```bash
cd src/data_capture
python calibration.py
```

Features:
- Real-time pose detection with MediaPipe
- Automatic landmark transformation to ToF coordinates
- Performance monitoring and statistics
- Press 'c' to reload calibration, 'q' to quit

## Files Structure

```
├── camera_calibration.json          # Pre-computed calibration data
├── setup_calibration.py            # One-time calibration setup
├── aruco_marker_77_calibration.png  # ArUco marker for setup
└── src/data_capture/
    ├── calibration.py              # Main application (optimized)
    ├── camera_calibration.py       # Calibration manager
    └── camera_calibration_utils.py # Utility functions
```

## Key Improvements

### ✅ Production Optimized
- **Pre-computed calibration**: No real-time ArUco detection overhead
- **Fast landmark transformation**: Direct homography application
- **Stable performance**: Consistent frame rates without calibration interruptions

### ✅ Quality Assurance
- **Calibration validation**: Automatic quality metrics
- **Valid region filtering**: Only transform landmarks in reliable regions
- **Error handling**: Graceful fallback when calibration unavailable

### ✅ User Experience
- **Clear status display**: Transformation success rates shown
- **Easy setup**: Simple calibration process with visual guides
- **Runtime control**: Reload calibration without restarting

## Calibration Quality Metrics

The system tracks and displays:
- **Reprojection error**: Pixel-level accuracy of transformation
- **Coverage percentage**: Portion of image with reliable transformation
- **Success rate**: Percentage of landmarks successfully transformed
- **Condition number**: Numerical stability of homography matrix

## Configuration

### Camera Settings
```python
# In calibration.py
WEBCAM_INDEX = 8        # RGB camera index
WEBCAM_WIDTH = 640      # RGB resolution
WEBCAM_HEIGHT = 480
CAMERA_RANGE_MM = 4000  # ToF range
```

### MediaPipe Settings
```python
MP_MODEL_COMPLEXITY = 1              # Balance speed/accuracy
MP_MIN_DETECTION_CONFIDENCE = 0.75   # Detection threshold
MP_MIN_TRACKING_CONFIDENCE = 0.75    # Tracking threshold
```

## Troubleshooting

### Calibration Issues
- Ensure ArUco marker ID 77 is clearly visible to both cameras
- Check lighting conditions for proper marker detection
- Verify camera connections and permissions

### Performance Issues
- Monitor transformation success rates in display
- Check calibration quality metrics
- Consider recalibration if success rate < 80%

### Runtime Issues
- Press 'c' to reload calibration if transformation fails
- Check `camera_calibration.json` file exists and is valid
- Verify camera indices and connections

## Development Notes

### Real Camera Integration
For production use with actual cameras:

1. **Replace test images** in `setup_calibration.py` with live camera capture
2. **Implement camera interfaces** for RGB webcam and ToF camera
3. **Validate calibration** with actual camera pair before deployment

### Calibration Timing Strategy
- **Setup phase**: One-time ArUco calibration (computationally expensive)
- **Runtime phase**: Pre-computed transformation (fast and stable)
- **Maintenance**: Periodic recalibration as needed

This approach separates the expensive calibration process from real-time operation, ensuring optimal performance in production environments.
