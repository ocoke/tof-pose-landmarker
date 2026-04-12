from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require_tensorflow():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("TensorFlow is required for training/export. Install with `.[train]`.") from exc
    return tf


def build_tiny_pose_model(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    joints: int = 15,
    alpha: float = 0.35,
):
    tf = _require_tensorflow()
    inputs = tf.keras.Input(shape=input_shape, name="roi")
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights=None,
        input_tensor=inputs,
        minimalistic=False,
    )
    x = backbone.output
    x = tf.keras.layers.Conv2D(96, 1, activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.SeparableConv2D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)

    heatmaps = tf.keras.layers.Conv2D(joints, 1, padding="same", activation="sigmoid", name="heatmaps")(x)
    offsets = tf.keras.layers.Conv2D(joints * 2, 1, padding="same", activation=None, name="offsets")(x)
    return tf.keras.Model(inputs=inputs, outputs=[heatmaps, offsets], name="tiny_depth_pose")


def load_npz_dataset(paths: list[Path]):
    images: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    for path in paths:
        payload = np.load(path)
        images.append(payload["images"])
        heatmaps.append(payload["heatmaps"])
        offsets.append(payload["offsets"])
    return (
        np.concatenate(images, axis=0).astype(np.float32),
        np.concatenate(heatmaps, axis=0).astype(np.float32),
        np.concatenate(offsets, axis=0).astype(np.float32),
    )


def train(args: argparse.Namespace) -> int:
    tf = _require_tensorflow()
    model = build_tiny_pose_model(alpha=args.alpha)
    images, heatmaps, offsets = load_npz_dataset([Path(path) for path in args.dataset])
    output_dir = Path(args.output) if args.output else None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            "heatmaps": tf.keras.losses.BinaryCrossentropy(),
            "offsets": tf.keras.losses.Huber(),
        },
        loss_weights={"heatmaps": 1.0, "offsets": 0.25},
    )
    callbacks = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=str(output_dir / "best.keras"), save_best_only=True))
    model.fit(
        images,
        {"heatmaps": heatmaps, "offsets": offsets},
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        callbacks=callbacks,
    )
    if output_dir is not None:
        export_path = output_dir / "saved_model"
        if hasattr(model, "export"):
            model.export(str(export_path))
        else:
            tf.saved_model.save(model, str(export_path))
    return 0


def export_tflite(args: argparse.Namespace) -> int:
    tf = _require_tensorflow()
    converter = tf.lite.TFLiteConverter.from_saved_model(str(args.saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    if args.representative:
        reps = np.load(args.representative)["images"].astype(np.float32)

        def representative_dataset():
            for sample in reps[: min(len(reps), 256)]:
                yield [sample[None, ...]]

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    Path(args.output).write_bytes(tflite_model)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python training/tiny_pose_tf.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--dataset", nargs="+", required=True)
    train_parser.add_argument("--output", type=Path, default=Path("artifacts"))
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--validation-split", type=float, default=0.1)
    train_parser.add_argument("--alpha", type=float, default=0.35)
    train_parser.set_defaults(func=train)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--saved-model", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--representative", type=Path)
    export_parser.set_defaults(func=export_tflite)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
