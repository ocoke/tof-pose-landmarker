# Macchiato v9

Macchiato is version 9 of the Time-of-Flight pose landmarker. Experiment B
compares a COCO-pretrained YOLO26 Nano pose model against the canonical v7 U-Net
on the same complete-scene Arducam split.

## Experiment B

- YOLO input: three identical channels produced from raw depth clipped to
  4,000 mm and losslessly stored as PNG.
- U-Net input: normalized depth, log-normalized confidence, and their product.
- Labels: existing COCO-17 projected keypoints, with out-of-frame points masked.
- Split: complete timestamp scenes separated by gaps greater than 180 seconds.
- Test policy: test samples are never used for training or checkpoint selection.

Install the package and test dependencies in a training environment:

```bash
python -m pip install -e '.[test]'
```

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

The command refuses to replace an existing generated YOLO dataset. Add
`--force` only for an intentional full rebuild.

## Guarded training

Both commands below are dry runs. They validate paths, manifests, scene
isolation, and hyperparameters without importing model frameworks or downloading
weights.

```bash
python -m macchiato.train yolo --config configs/macchiato_finetune.yaml
python -m macchiato.train unet --config configs/macchiato_finetune.yaml
```

Training starts only when `--run` is explicitly added. U-Net's default `all`
stage runs Stage A and then initializes Stage B from its best checkpoint.

```bash
python -m macchiato.train yolo --config configs/macchiato_finetune.yaml --run
python -m macchiato.train unet --config configs/macchiato_finetune.yaml --run
```

After both checkpoints exist, compare them on held-out test scenes:

```bash
python -m macchiato.evaluation.evaluate_generalization \
  --config configs/macchiato_finetune.yaml \
  --unet-checkpoint checkpoints/macchiato_b/unet/stage_b_best.pth \
  --yolo-checkpoint experiments/macchiato_b/yolo/yolo26n_pose_depth/weights/best.pt
```

Generated data, experiments, and checkpoints are ignored by Git. Manifest CSVs
are tracked so the exact scene assignment remains reviewable.
