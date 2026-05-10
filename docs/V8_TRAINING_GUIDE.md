# V8 Training and Deployment Guide

End-to-end instructions for capturing data, training the V8 ToF pose model,
exporting it as INT8 TFLite, and running it on a Raspberry Pi 5. Pi-specific
debugging is in [Troubleshooting](#troubleshooting) at the end.

## Contents

1. [Overview](#overview)
2. [Hardware](#hardware)
3. [Software install](#software-install)
4. [Workflow at a glance](#workflow-at-a-glance)
5. [Step 1 - Camera calibration](#step-1---camera-calibration)
6. [Step 2 - Capture training data](#step-2---capture-training-data)
7. [Step 3 - Convert labels](#step-3---convert-labels)
8. [Step 4 - Sanity-check the dataset](#step-4---sanity-check-the-dataset)
9. [Step 5 - Train](#step-5---train)
10. [Step 6 - Export INT8 TFLite](#step-6---export-int8-tflite)
11. [Step 7 - Deploy and run on the Pi](#step-7---deploy-and-run-on-the-pi)
12. [Iterating: how to improve a trained model](#iterating-how-to-improve-a-trained-model)
13. [Troubleshooting](#troubleshooting)

---

## Overview

V8 has three lifecycle stages:

```
  CAPTURE                  TRAIN                    DEPLOY
  ----------------         ----------------         ----------------
  webcam + ToF      ->     data/  +  PyTorch   ->   tflite + tof_pose
  + MediaPipe              GPU box                  runtime on Pi 5
  ArUco-projected
  labels
```

**Capture** uses an RGB webcam to run MediaPipe and project pose landmarks
into the ToF camera's pixel frame via an ArUco-based homography. The webcam
is only a label generator. It is **not used at runtime**.

**Training** reads the captured depth/confidence frames + COCO-17 labels and
produces a tiny depth-native pose model (`HeatmapOffsetPoseEstimator`). The
model has no knowledge of RGB.

**Deploy** loads only the `.tflite` and uses the Arducam ToF camera. The
webcam can be unplugged forever after capture.

Joint scheme: 15 V8 joints (head, neck, mid_spine, R/L shoulder/elbow/hand,
R/L hip/knee/foot), derived from COCO-17 labels via the conversion in
[training/train_v8.py](../training/train_v8.py).

---

## Hardware

| Component | Required | Notes |
|---|---|---|
| Raspberry Pi 5 (4 GB or 8 GB) | yes | Runtime target |
| Arducam ToF Camera (CSI) | yes | The depth sensor |
| Active cooling on the Pi | strongly recommended | Thermal throttling during long capture wrecks frame rate |
| USB webcam (any) | for capture only | Used to generate labels via MediaPipe |
| ArUco marker 6x6 #77 with retroreflective tape on corners | for capture only | See `assets/aruco_marker_77_calibration.png` |
| Tripod or fixed mount | yes | V8 assumes a fixed camera pose. Re-calibrate if you move it |
| Training machine (GPU helpful) | yes | TensorFlow training; an x86 box with an NVIDIA GPU is far faster than the Pi |

---

## Software install

### On the Raspberry Pi 5

```bash
# system packages
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv libatlas-base-dev

# Arducam ToF SDK (follow Arducam's official guide for the exact version)
# Pin to 0.1.24 to match what the V8 camera adapter targets.

# Python venv for the V8 runtime
python3 -m venv ~/.venvs/tof-pose
source ~/.venvs/tof-pose/bin/activate

cd ~/tof-pose-landmarker
pip install -e ".[runtime]"

# Optional: capture deps (only needed if you also capture on the Pi)
pip install mediapipe==0.10.11
```

Verify the camera adapter loads:

```bash
python3 -m tof_pose.cli inspect-camera
```

### On the training machine

```bash
git clone https://github.com/ocoke/tof-pose-landmarker.git
cd tof-pose-landmarker
git checkout v8

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[train]"
```

This pulls TensorFlow 2.x. A CUDA-capable GPU helps enormously - training on
CPU works but takes ~10x longer.

---

## Workflow at a glance

```
1. Calibrate webcam <-> ToF (once per mount)         Pi
2. Capture training data                              Pi
3. Convert MediaPipe 33-pt -> COCO-17                 Pi or any
4. Sanity-check the dataset                           any
5. Train                                              training box
6. Export INT8 TFLite                                 training box
7. Copy .tflite to Pi, run inference                  Pi
```

---

## Step 1 - Camera calibration

The capture script needs a calibrated homography from webcam pixels to ToF
pixels. Without it, every pose label is off.

Stick retroreflective tape on the corners of the ArUco marker (see
`assets/aruco_marker_77_calibration.png`). The tape makes those corners
bright in the ToF amplitude image so the same marker can be detected by both
cameras.

```bash
cd ~/tof-pose-landmarker
python3 src/data_capture/webcam_tof_pose_mapper.py --webcam-id 0 --tof-range 4000
```

In the GUI:
1. Hold the ArUco marker in front of both cameras.
2. Move it to many positions across the frame (corners, edges, centre, various depths).
3. Press the on-screen "calibrate" key when prompted; the script writes
   `camera_calibration.json` at repo root.

Verify the calibration:

```bash
python3 src/data_capture/example/analyze_calibration.py
```

If the analysis shows large per-point reprojection errors, recalibrate. Bad
calibration is the #1 cause of poor label quality.

Optionally also capture a floor plane (only used by the V8 runtime, not by
training):

```bash
python3 -m tof_pose.cli calibrate-floor \
    --depth-unit mm \
    --preview \
    --output floor_plane.json \
    --frames 45
```

---

## Step 2 - Capture training data

```bash
python3 src/data_capture/webcam_tof_pose_mapper.py \
    --webcam-id 0 \
    --tof-range 4000 \
    --no-rotate
```

Controls in the GUI:
- `r` - start recording
- `s` - stop recording
- `q` - quit

While recording, each saved frame writes:

| File | Content |
|---|---|
| `data/depth/{ts}.npy` | raw depth in mm |
| `data/confidence/{ts}.npy` | ToF confidence (0-255) |
| `data/pose/{ts}.json` | MediaPipe 33 pts projected into ToF pixels |
| `data/tof/{ts}.png` | ToF visualisation (for human inspection) |
| `data/webcam/{ts}.jpg` | webcam frame |
| `data/original_pose/{ts}.json` | MediaPipe in webcam pixels |

The capture rate is gated to 1 frame/second by default (see
`webcam_tof_pose_mapper.py` line ~757). Plan for ~30-60 minutes of recording
to accumulate a few thousand frames.

### Capture protocol

Vary across these axes to avoid a model that only works in one corner of the
room:

| Axis | Suggestion |
|---|---|
| Distance | 1.0 m, 1.8 m, 2.5 m, 3.3 m |
| Orientation | front, back, left/right side, 45 deg |
| Pose | standing arms-down, T-pose, arms-up, sitting, crouching, walking |
| Occlusion | hands behind back, arm across chest |
| Clothing | 2+ outfits per subject |
| Subjects | 2-4 if possible |
| Background | clean + slightly cluttered |

Capture a **separate test session** later (different day, ideally different
subject/lighting). Without an isolated test set the PCK numbers will lie.

---

## Step 3 - Convert labels

The training script wants COCO-17 keypoints. Main captured MediaPipe 33-pt.
Run the converter:

```bash
python3 src/model_slim/convert_dataset.py
```

This reads `data/pose/*.json` and writes `data/pose_coco17/*.json`. Re-run
whenever you add new capture data.

---

## Step 4 - Sanity-check the dataset

Before training, eyeball labels and depth quality. Bad labels in -> bad model
out, no matter how good the trainer is.

### Count samples

```bash
ls data/depth | wc -l
ls data/pose_coco17 | wc -l
```

Both should match (within a couple). If `pose_coco17` is much smaller, the
converter dropped frames that didn't have valid 33-pt data - check the
converter's "skipping" messages.

### Inspect a few labels visually

There's no built-in overlay tool yet. Quick script:

```python
import json, numpy as np, cv2
from pathlib import Path

stem = Path("data/depth").glob("*.npy").__next__().stem
depth = np.load(f"data/depth/{stem}.npy")
kp = json.load(open(f"data/pose_coco17/{stem}.json"))["keypoints"]

vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
for x, y in kp:
    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
cv2.imwrite("/tmp/check.png", vis)
```

The 17 points should land on the body. If they're way off, ArUco calibration
drifted - re-do Step 1.

### Verify depth units

```bash
python3 -c "import numpy as np; \
  a = np.load(sorted(__import__('glob').glob('data/depth/*.npy'))[0]); \
  print('shape', a.shape, 'dtype', a.dtype, 'min', a.min(), 'max', a.max())"
```

You should see values in the 200-4000 range (millimetres). If they're 0.2-4
(metres), set `--depth-scale-mm 4` when running training (or edit the
`TrainConfig` default).

---

## Step 5 - Train

On the training box:

```bash
cd ~/tof-pose-landmarker
source .venv/bin/activate

# transfer the captured `data/` directory here first (scp, rsync, or share)
ls data/  # should show: confidence depth original_pose pose pose_coco17 tof webcam

python3 training/train_v8.py train \
    --data-dir data \
    --output artifacts/v8_run1 \
    --epochs 60 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --alpha 0.5
```

What you should see each epoch:

```
[epoch 7/60] val_loss=0.0234  pck@0.05=0.612  lr=1.0e-03
  -> new best PCK, saved artifacts/v8_run1/best.keras
```

- `val_loss` should fall in the first 10 epochs then plateau.
- `pck@0.05` (fraction of joints within 5 percent of the ROI diagonal) is the
  one to watch. Expect 0.5+ within a few epochs on a clean dataset.
- LR drops automatically on plateau; early stop after 10 epochs without PCK
  improvement.

Outputs in `artifacts/v8_run1/`:
- `best.keras` - best checkpoint by PCK
- `saved_model/` - TF SavedModel for export
- `history.json` - per-epoch metrics
- `representative.npz` - sample inputs for INT8 quantisation calibration

### Trainable knobs

| Flag | Default | What it does |
|---|---|---|
| `--epochs` | 60 | hard cap; early stop usually triggers first |
| `--batch-size` | 32 | drop if you run out of GPU memory |
| `--learning-rate` | 1e-3 | drop to 3e-4 if loss is unstable |
| `--alpha` | 0.5 | MobileNetV3-Small width; 0.75 is more accurate but slower |
| `--backbone-weights` | `imagenet` | set to `''` to train from scratch (worse, but no internet needed) |
| `--no-augment` | off | disable all augmentation - use only for debugging |

### Train on the Pi instead

Possible but slow. Use `--batch-size 4` and expect ~10x slower epochs than a
desktop GPU. Keep the Pi cool and connected to mains power.

---

## Step 6 - Export INT8 TFLite

```bash
python3 training/train_v8.py export \
    --saved-model artifacts/v8_run1/saved_model \
    --representative artifacts/v8_run1/representative.npz \
    --output models/depth_pose_int8.tflite
```

The exporter:
- Quantises to INT8 (`uint8` input, `float32` outputs)
- Uses the representative.npz the trainer dumped during validation, so the
  quantisation calibration sees realistic ROI tensors

Resulting `.tflite` should be 0.5-3 MB depending on alpha.

Sanity check the exported model:

```python
import tensorflow as tf
i = tf.lite.Interpreter("models/depth_pose_int8.tflite")
i.allocate_tensors()
print("inputs:", i.get_input_details())
print("outputs:", i.get_output_details())
```

Verify:
- Input shape `(1, 128, 128, 2)` - if it shows 3 channels, the trained
  config was changed and the pipeline must produce 3 channels too.
- Output 1: heatmaps `(1, 64, 64, 15)`.
- Output 2: offsets `(1, 64, 64, 30)`.

---

## Step 7 - Deploy and run on the Pi

Copy the `.tflite` to the Pi:

```bash
# on the training box
scp models/depth_pose_int8.tflite pi@<pi-ip>:~/tof-pose-landmarker/models/

# also copy floor_plane.json if you made one
scp floor_plane.json pi@<pi-ip>:~/tof-pose-landmarker/
```

On the Pi:

```bash
source ~/.venvs/tof-pose/bin/activate
cd ~/tof-pose-landmarker

python3 -m tof_pose.cli run-production \
    --rotate 180 \
    --depth-unit mm \
    --preview \
    --pose-model models/depth_pose_int8.tflite \
    --floor-plane floor_plane.json
```

The pipeline will:
1. Read the trained model and detect its input channel count.
2. Build a matching 2-channel ROI tensor using **the same fixed-scale
   normalisation** that training used (`depth_m / 4.0`, `confidence / 255`).
3. Run inference, lift 2D joints to 3D using metric depth, and overlay them
   on the preview window.

Press `q` or `Esc` to quit.

For headless operation, drop `--preview`. The CLI still runs the pipeline -
useful for piping pose output to your own application.

---

## Iterating: how to improve a trained model

Order of operations once you have a baseline:

1. **Capture more data.** This dominates everything else until you're past
   ~5000 samples. Focus on whatever poses or distances fail in the live
   preview.
2. **Curate labels.** Drop sessions where calibration drifted; eyeball a
   couple hundred frames and exclude clearly wrong ones.
3. **Add held-out test capture.** Until you have one, PCK in the trainer is
   self-reported on val frames seconds away from train frames - inflated.
4. **Try `--alpha 0.75`.** If the model has capacity issues but you still
   meet your latency budget on the Pi.
5. **Synthetic data.** Render SMPL bodies through a simulated ToF intrinsic.
   Use this only after real data plateaus.

---

## Troubleshooting

### Pi: install / environment

| Symptom | Cause | Fix |
|---|---|---|
| `import ArducamDepthCamera` fails | Arducam SDK not installed or wrong version | Reinstall the Arducam SDK pinned to 0.1.24 |
| `OSError: libGL.so.1: cannot open` | OpenCV needs system libs | `sudo apt install -y libgl1-mesa-glx libglib2.0-0` |
| `pip install mediapipe` fails on Pi 5 (arm64) | No prebuilt wheel for newer versions | Try `pip install mediapipe==0.10.11`; if still fails, use the `mediapipe-rpi` fork or capture on a different machine |
| `pip install tensorflow` fails on Pi | No official aarch64 wheels for recent TF | Don't install TF on the Pi - it's only needed for **training**. Inference uses `tflite-runtime` (installed via `.[runtime]`) |
| `Permission denied: /dev/video0` | User not in `video` group | `sudo usermod -aG video $USER` and log out/in |

### Pi: capture / runtime

| Symptom | Cause | Fix |
|---|---|---|
| `inspect-camera` shows no device | CSI cable or SDK fail | Re-seat the CSI ribbon; check `dmesg` for camera enumeration; reinstall the Arducam SDK |
| Black or all-zero depth in preview | Wrong depth unit | Add `--depth-unit mm` (default `auto` sometimes guesses wrong) |
| Frame rate drops over time | Thermal throttling | `vcgencmd measure_temp`; add an active cooler; if temp >80 C the SoC throttles |
| `--preview` window doesn't open | No DISPLAY / SSH without X | Use VNC or `ssh -X`; or drop `--preview` and run headless |
| ROI keeps snapping to background | Floor plane bad | Re-run `calibrate-floor` with more frames; ensure no person was in the empty-scene capture |
| MediaPipe inference is very slow on Pi during capture | Pi is small | Capture data on a faster machine if you have a separate ToF rig, or accept slower capture |
| Many `Permission denied` errors writing `data/` | Filesystem permissions | `chmod -R u+w data/` from inside the project directory |
| `ArduCamSDK: getDepthData returned None` repeatedly | Frame timeout too tight | Increase `--timeout-ms` (default 200); check USB/CSI cabling |

### Training (off-Pi)

| Symptom | Cause | Fix |
|---|---|---|
| `tensorflow.python.framework.errors_impl.UnknownError: OOM` | GPU OOM | Reduce `--batch-size` (try 16, then 8) |
| `ValueError: Cannot import name 'image_dataset_from_directory'` | Wrong TF version | Ensure `tensorflow>=2.14`; check `pip show tensorflow` |
| Training loss = `nan` after first batch | Bad data values | Run the "verify depth units" check; look for inf/nan in `data/depth/*.npy` |
| PCK stuck near 0 after 10 epochs | Bad labels | Run Step 4's overlay sanity check on 20 random frames |
| PCK trains well but real inference is awful | Train/inference distribution mismatch | Verify the pipeline patch landed: `grep -n input_c tof_pose/pipeline.py` should show two references in `_build_roi_tensor` |
| `model.export` is missing | Old TF | Upgrade to TF 2.13+; otherwise it falls back to `tf.saved_model.save` automatically |
| INT8 export errors with `Cannot quantize` | Hard-swish / hard-sigmoid ops | `train_v8.py` already passes `minimalistic=True` to MobileNetV3-Small to avoid this. If you changed the backbone, prefer ReLU activations |

### Inference

| Symptom | Cause | Fix |
|---|---|---|
| `InferenceError: Model expects 2 channels, got 3` | Model/pipeline channel mismatch | Pull the latest `v8` (commit 32ac2d3+); the pipeline now adapts to `pose_estimator.input_c` |
| Pose joints all clustered at one corner | Train/infer normalisation mismatch | Check `PipelineConfig.depth_scale_m == 4.0` and `confidence_scale == 255.0`; these must equal training's `depth_scale_mm`/1000 and `confidence_scale` |
| Joints jitter heavily frame-to-frame | One-Euro filter too loose | Tune `PipelineConfig.joint_filter_min_cutoff` upward (e.g., 2.0) and `beta` down (e.g., 0.01) |
| Inference latency >100 ms / frame | INT8 not applied or alpha too high | Confirm input dtype is `uint8` via the sanity check in Step 6; try `--alpha 0.35` |
| Joints stick to body even when person moves | Pose 1-Euro filter over-damped | Lower `joint_filter_min_cutoff` |

### Quick diagnostic commands

```bash
# Pi: is the ToF camera enumerated?
v4l2-ctl --list-devices

# Pi: SoC temperature
vcgencmd measure_temp

# Pi: how much memory is free?
free -h

# Any box: how many samples do we actually have?
ls data/depth/ | wc -l

# Any box: check what's in the trained tflite
python3 -c "import tensorflow as tf; i=tf.lite.Interpreter('models/depth_pose_int8.tflite'); i.allocate_tensors(); \
  print('IN', i.get_input_details()[0]['shape'], i.get_input_details()[0]['dtype']); \
  [print('OUT', d['name'], d['shape'], d['dtype']) for d in i.get_output_details()]"

# Any box: do the depth/confidence pairs have matching names?
diff <(ls data/depth | sed 's/.npy$//') <(ls data/confidence | sed 's/.npy$//')

# Pi: verify the pipeline channel-adaptation patch is in place
grep -n input_c tof_pose/pipeline.py
```

---

## Reference

- Training script: [training/train_v8.py](../training/train_v8.py)
- Runtime CLI: [tof_pose/cli.py](../tof_pose/cli.py)
- Pipeline: [tof_pose/pipeline.py](../tof_pose/pipeline.py)
- Inference adapters: [tof_pose/inference.py](../tof_pose/inference.py)
- Capture: [src/data_capture/webcam_tof_pose_mapper.py](../src/data_capture/webcam_tof_pose_mapper.py)
- Label conversion: [src/model_slim/convert_dataset.py](../src/model_slim/convert_dataset.py)
- Joint list: [tof_pose/types.py](../tof_pose/types.py) `JOINT_NAMES_15`
- Runtime README: [README_V8.md](../README_V8.md)
