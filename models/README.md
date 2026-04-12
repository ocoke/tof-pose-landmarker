# Model Store

This repository does not commit large third-party `.tflite` binaries directly.

Use:

```bash
tof-pose fetch-models
```

That command downloads the demo baseline model into this directory using the pinned manifest in `MANIFEST.json`.

Current managed artifacts:

- `movenet_singlepose_lightning_int8.tflite`: no-fine-tune demo baseline from TensorFlow Hub

The production depth-native model is intentionally not auto-downloaded because it is project-specific and should come from your own training/export pipeline.
