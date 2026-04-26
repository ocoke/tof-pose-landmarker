# ToF Pose Estimation V8

Hybrid 3D front-end and tiny depth-native pose runtime for the Arducam ToF camera on Raspberry Pi 5.

The package implements the runtime architecture discussed in the plan:

1. Capture `depth_data`, `amplitude_data`, and `confidence_data` from `ArducamDepthCamera` `0.1.24`
2. Build a point cloud and estimate a floor plane for a fixed camera
3. Remove floor/background, isolate the best human-sized foreground cluster, and stabilize a person ROI
4. Run either:
   - a no-fine-tune demo baseline using MoveNet on normalized amplitude ROI
   - a production tiny pose model on `depth + confidence + amplitude`
5. Lift 2D joints into metric 3D using the original depth map and temporal filtering

## Installation

Core package:

```bash
python3 -m pip install -e .
```

Pi runtime extras:

```bash
python3 -m pip install -e '.[runtime]'
```

Training/export extras:

```bash
python3 -m pip install -e '.[train]'
```

## Runtime assumptions

- Raspberry Pi 5
- CPU-only inference
- Fixed camera mount
- Arducam ToF camera in `4 m` range mode
- Single-person tracking
- Target runtime `10-15 FPS`

## CLI

Inspect the camera and verify the latest SDK surface:

```bash
tof-pose inspect-camera
```

Download the demo model into the repository:

```bash
tof-pose fetch-models
```

Calibrate and save the floor plane from empty-scene frames:

```bash
tof-pose calibrate-floor \
  --depth-unit mm \
  --preview \
  --output floor_plane.json \
  --frames 45
```

If the camera is mounted upside down, add:

```bash
--rotate 180
```

Run the no-fine-tune demo baseline with MoveNet:

```bash
tof-pose run-demo \
  --rotate 180 \
  --depth-unit mm \
  --preview \
  --movenet-model models/movenet_singlepose_lightning_int8.tflite \
  --floor-plane floor_plane.json
```

Run the production path with a custom tiny TFLite model:

```bash
tof-pose run-production \
  --rotate 180 \
  --depth-unit mm \
  --preview \
  --pose-model /path/to/depth_pose_int8.tflite \
  --floor-plane floor_plane.json
```

The Arducam SDK returns depth in millimeters on current Pi builds, so pass `--depth-unit mm` if calibration shows zero usable depth points. The calibration preview shows amplitude, depth, and the exact candidate mask used for floor-plane fitting. The runtime preview shows amplitude and depth side-by-side with the tracked ROI and pose overlay. Press `q` or `Esc` to close either window. This requires a local desktop session, VNC, or X11 forwarding; it will not open in a headless shell.

## Data contracts

Internal runtime frame:

```text
DepthFrame {
  depth_m, amplitude, confidence, valid_mask,
  fx, fy, cx, cy, timestamp
}
```

Tracking result:

```text
TrackedPerson {
  roi_px, cluster_id, centroid_xyz, extent_xyz, quality
}
```

Pose output:

```text
Pose2D { joints_uv[15], scores[15], roi_px }
Pose3D { joints_xyz[15], valid[15], scores[15] }
```

## Training scaffold

The runtime does not depend on TensorFlow or PyTorch. Training/export lives in `training/tiny_pose_tf.py`.

Expected training dataset format for the scaffold:

- one or more `.npz` files
- keys:
  - `images`: `(N, 128, 128, 3)` float32 tensors containing `[depth_norm, confidence_norm, amplitude_norm]`
  - `heatmaps`: `(N, H, W, 15)` float32 target heatmaps
  - `offsets`: `(N, H, W, 30)` float32 local offset targets

Example export:

```bash
python3 training/tiny_pose_tf.py export \
  --saved-model /path/to/saved_model \
  --output /path/to/depth_pose_int8.tflite \
  --representative /path/to/representative_samples.npz
```

## Notes

- The no-fine-tune demo path is intentionally separated from the production path. It is a visibility baseline, not the final architecture.
- The geometry front-end is deliberately stronger than a pure 2D image pipeline because the fixed-mount `4 m` setup benefits from floor/background geometry and point-cloud-derived ROI stabilization.
- The repository tracks model metadata and download instructions, not large third-party `.tflite` binaries directly.
