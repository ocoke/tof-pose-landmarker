import torch
import ai_edge_torch
import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Make sure your model.py and dataset.py are accessible
try:
    from model_slim.medium_model import MediumPoseUNet
    from model_slim.dataset import SlimPoseDataset
except ModuleNotFoundError:
    print("❌ Error: Could not find model or dataset files.")
    sys.exit(1)


# --- Configuration ---
PYTORCH_MODEL_PATH = "./models/pose_unet_bce_best_model_medium.pth"
DATA_DIR = "./data"
TFLITE_QUANT_MODEL_PATH = "./models/medium_model_quant_float16.tflite"
INPUT_SHAPE = (1, 2, 240, 240)
NUM_KEYPOINTS = 17


# --- Step 1: Load Your PyTorch Model ---
print("="*50)
print("Step 1: Loading PyTorch 'Medium' Model...")
pytorch_model = MediumPoseUNet(in_ch=2, n_kpts=NUM_KEYPOINTS)
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
pytorch_model.eval()
print("✅ PyTorch model loaded successfully.")


# --- Step 2: Prepare the Representative Dataset Generator ---
print("\n" + "="*50)
print("Step 2: Defining representative dataset generator...")

def representative_dataset_gen():
    rep_dataset = SlimPoseDataset(data_dir=DATA_DIR, output_res=(240, 240), num_keypoints=NUM_KEYPOINTS, augment=False)
    num_calibration_steps = min(100, len(rep_dataset))
    print(f"Generating calibration data from {num_calibration_steps} samples...")
    for i in range(num_calibration_steps):
        input_tensor, _ = rep_dataset[i]
        yield [input_tensor.unsqueeze(0).numpy()]

print(f"✅ Representative dataset generator defined.")


# --- Step 3: Define TFLite Converter Flags and Convert ---
print("\n" + "="*50)
print(f"Step 3: Converting to Quantized FLOAT16 TFLite...")

tfl_converter_flags = {
    'optimizations': [tf.lite.Optimize.DEFAULT],

    'target_spec.supported_types': [tf.float16],  # for float16

    # for int 8
    # 'representative_dataset': representative_dataset_gen,
    # 'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
    # 'inference_input_type': tf.float32,
    # 'inference_output_type': tf.float32,
}

edge_model_quantized = ai_edge_torch.convert(
    pytorch_model,
    (torch.randn(INPUT_SHAPE),),
    _ai_edge_converter_flags=tfl_converter_flags
)
print("✅ Model successfully converted and quantized.")


# --- Step 4: Save and Verify ---
print(f"\nSaving Quantized TFLite model to '{TFLITE_QUANT_MODEL_PATH}'...")
edge_model_quantized.export(TFLITE_QUANT_MODEL_PATH)
print(f"✅ Quantized model saved.")

# Verification
original_size = os.path.getsize(PYTORCH_MODEL_PATH)
quantized_size = os.path.getsize(TFLITE_QUANT_MODEL_PATH)
print("\n--- Verification ---")
print(f"Original PyTorch model size: {original_size / (1024*1024):.2f} MB")
print(f"Quantized TFLite model size: {quantized_size / (1024*1024):.2f} MB")