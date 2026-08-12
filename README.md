# Time-of-Flight Depth Pose Estimator

The project estimates human pose from Time-of-Flight (ToF) depth at inference time, eliminating RGB domain shift. 

V9 replaces the U-Net style CNN model path with a COCO-pretrained YOLO26 Nano pose model fine-tuned on Arducam depth frames.


## Dataset

The v9 source pool contains 4,109 timestamp-matched samples with depth, confidence, and pose files. 
After validation, 3,895 samples have at least 12 finite, in-frame COCO-17 keypoints and are eligible for training or evaluation.

The dataset is splited by scenes.

No scene appears in more than one split. Validation selects checkpoints; test samples are never used for training or checkpoint selection.

### Capture and annotation

An Arducam ToF camera and a USB webcam captured paired depth and RGB frames. A calibration procedure using ArUco marker ID 77 from `DICT_6X6_250` mapped RGB coordinates into the ToF frame. Retroreflective tape on parts of the marker was intended to increase its infrared return so it could also be located in the ToF view.

MediaPipe Pose produced 33 landmarks from the RGB frame. Those landmarks were projected into the depth frame, converted to COCO-17 beginning with v4, and converted to YOLO pose labels for v9. The v9 model receives depth only. RGB is used during dataset annotation, not inference.

## Models and results

Versions v3 through v7 used custom U-Net-style heatmap regressors. 

Version 9 uses `yolo26n-pose.pt` as its pretrained starting point and fine-tunes it on the scene-split Arducam data. 

Each raw depth array is clipped to `[0, 4000]` mm, scaled to an 8-bit image, and copied into three identical channels. 
Training uses a 256-pixel input.


| Model | Evaluation protocol | Raspberry Pi 5 latency | FPS | MPJPE (px) | PCK@12 px | PCK@24 px | Detection coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.0.0 Edge, FP32 | Legacy random frame-level validation | 113.01 ms | 8.85 | 11.93 | 69.35% | 89.38% | N/A |
| 9.0.0 Macchiato, FP32 | Held-out test scenes, 793 frames | Not benchmarked | Not benchmarked | **8.60** | **77.89%** | **92.49%** | **97.35%** |

For v9, PCK is measured in the original 240x180 frame using fixed 12- and
24-pixel thresholds. Missed person detections count as PCK failures. MPJPE is
reported over visible keypoints in detected samples. 

## Usage

### Installation

Install the package and test dependencies in a training environment:

```bash
python -m pip install -e '.[test]'
```

The project pins `ultralytics==8.4.103`; use the same version to load YOLO26
checkpoints.

### Data preparation

Build manifests only:

```bash
python -m macchiato.create_manifest --config configs/macchiato_finetune.yaml
```

Build manifests plus normalized YOLO images and labels:

```bash
python -m macchiato.create_manifest \
  --config configs/macchiato_finetune.yaml \
  --prepare-yolo
```

The command refuses to replace a nonempty generated YOLO dataset. Add `--force`
only for an intentional full rebuild.

### Guarded training

Both commands below are dry runs. They validate paths, manifests, scene
isolation, dependencies, and hyperparameters without starting training or
downloading model weights:

```bash
python -m macchiato.train yolo --config configs/macchiato_finetune.yaml
python -m macchiato.train unet --config configs/macchiato_finetune.yaml
```

Training starts only when `--run` is explicitly added. The U-Net's default
`all` stage runs Stage A and then initializes Stage B from its best checkpoint.

```bash
python -m macchiato.train yolo --config configs/macchiato_finetune.yaml --run
python -m macchiato.train unet --config configs/macchiato_finetune.yaml --run
```

### Evaluation

Evaluate YOLO on the held-out test scenes:

```bash
python -m macchiato.evaluation.evaluate_generalization \
  --config configs/macchiato_finetune.yaml \
  --split test \
  --yolo-checkpoint experiments/macchiato_b/yolo/yolo26n_pose_depth/weights/best.pt
```

After retraining the canonical U-Net on the same manifests, compare both
checkpoints on the identical test split:

```bash
python -m macchiato.evaluation.evaluate_generalization \
  --config configs/macchiato_finetune.yaml \
  --split test \
  --unet-checkpoint checkpoints/macchiato_b/unet/stage_b_best.pth \
  --yolo-checkpoint experiments/macchiato_b/yolo/yolo26n_pose_depth/weights/best.pt
```

The evaluator makes three unmeasured warm-up calls per model by default, then
reports mean, p50, p95, minimum, and maximum prediction-pipeline latency plus
sequential throughput. Its timing scope includes input loading, preprocessing,
inference, and postprocessing, but not ground-truth loading or metric updates.
Change the warm-up count with `--latency-warmup-samples`.

For a YOLO-only accuracy and CPU performance run on Raspberry Pi 5:

```bash
python -m macchiato.evaluation.evaluate_generalization \
  --config configs/macchiato_finetune.yaml \
  --split test \
  --yolo-checkpoint experiments/macchiato_b/yolo/yolo26n_pose_depth/weights/best.pt \
  --device cpu \
  --latency-warmup-samples 10 \
  --output experiments/macchiato_b/evaluation_raspberry_pi_5.json
```

### Raspberry Pi live preview

Install the Arducam ToF SDK on the Raspberry Pi, connect the CSI camera, open a
graphical desktop session, and run:

```bash
macchiato-preview \
  --checkpoint experiments/macchiato_b/yolo/yolo26n_pose_depth/weights/best.pt \
  --device cpu
```

The equivalent module command is:

```bash
python -m macchiato.live_preview \
  --checkpoint experiments/macchiato_b/yolo/yolo26n_pose_depth/weights/best.pt \
  --device cpu
```

The window displays the highest-confidence COCO-17 skeleton and person box on
the normalized depth frame. It also shows Ultralytics inference latency and a
rolling camera-to-window FPS with p50/p95 pipeline latency. Press `q` or Escape
to close the stream. The depth frame is rotated 180 degrees by default to match
the capture setup; pass `--no-rotate` if the preview is upside down.

Useful Raspberry Pi controls include:

```text
--threads 4                 PyTorch CPU thread count
--confidence 0.25           person detection threshold
--keypoint-confidence 0.25  skeleton drawing threshold
--warmup-frames 3           model warm-up calls
--latency-window 120        rolling timing window
--scale 3                   nearest-neighbor display scale
--max-frames 0              zero runs until q/Escape
```
