# Time-of-Flight Pose Estimator (v7)

A Raspberry Pi 5 oriented pose landmark estimator for Arducam ToF data. The current supported model family is `v7`, a hardened version of the previous 3-channel edge model that keeps the same Pi-safe inference graph while improving scene-aware training.

## What v7 Uses

- Input resolution: `240x240`
- Input channels: `[depth, confidence, depth * confidence]`
- Backbone: lightweight depthwise-separable encoder/decoder
- Target format: COCO-17 keypoints
- Deployment target: Raspberry Pi 5

## Data Layout

Place your dataset under `data/`:

```text
data/
├── confidence/      # ToF confidence maps (.npy)
├── depth/           # ToF depth maps (.npy)
├── pose/            # Raw transformed MediaPipe JSON
├── pose_coco17/     # COCO17 labels for training
├── tof/             # Saved ToF frames (.png)
└── webcam/          # Optional webcam frames (.jpg)
```

## Convert Labels

Convert MediaPipe-33 transformed points into COCO-17 labels:

```bash
python src/model_v7/convert_dataset.py --input-dir ./data/pose --output-dir ./data/pose_coco17
```

## Build a Manifest

The v7 trainer uses a manifest with inferred `session_id` values and scene-aware splits.

```bash
python src/model_v7/manifest.py --data ./data
```

If you have explicit scene metadata, provide a JSON map of `sample_id -> scene_id`:

```bash
python src/model_v7/manifest.py --data ./data --scene-map ./data/scene_map.json
```

## Train v7

Stage A bootstraps on `train_core` and tracks `val_seen`:

```bash
python src/model_v7/train.py --data ./data --stage a
```

Stage B fine-tunes the best Stage A checkpoint for harder cross-scene generalization:

```bash
python src/model_v7/train.py --data ./data --stage b --resume ./models/v7_stage_a_best.pth
```

Key training behavior:

- uses the canonical `EdgePoseUNetV7`
- applies masked heatmap and coordinate losses
- filters weak frames (`< 12` valid keypoints) from training
- uses scene-balanced sampling
- logs `val_seen` and `val_unseen` separately

## Evaluate v7

Evaluate the PyTorch checkpoint on a manifest split:

```bash
python src/model_v7/evaluate.py --model ./models/v7_stage_a_best.pth --data ./data --split val_seen
```

Use the compatibility entrypoint if you prefer the old script path:

```bash
python src/evaluation/pytorch.py --model ./models/v7_stage_a_best.pth --data ./data --split val_unseen
```

Optional MediaPipe baseline comparison:

```bash
python src/model_v7/evaluate.py --model ./models/v7_stage_a_best.pth --data ./data --split test_unseen --with-mediapipe
```

## Visualize Predictions

```bash
python src/model_v7/visualize.py --model ./models/v7_stage_a_best.pth --data ./data --split val_seen --count 4
```

## Quantize for TFLite

```bash
python src/quantization/main.py --model ./models/v7_stage_a_best.pth --data ./data --output ./models/v7_float16.tflite
```

## Pi 5 Baseline Reference

The current no-scene-split FP32 baseline that v7 preserves as its latency target is:

| Model | Latency (ms) | FPS | Avg. Error (px) | PCK@5% | PCK@10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6.0.0 Edge (FP32) | 113.01 | 8.85 | 11.93 | 69.35 | 89.38 |

v7 keeps the same deployed graph class so the first-pass latency goal remains effectively this budget while training focuses on better unseen-scene robustness.
