# visualize_dataset.py
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from dataset import PoseDataset # Make sure your dataset.py is in the same folder

# --- Configuration ---
DATA_DIR = "./data"  # The root directory for your collected data
# It's best to use your validation set for visualization if you have one
# Otherwise, your training set is fine.
DATASET_TO_VISUALIZE = '' 
BATCH_SIZE = 4 # How many samples to visualize at once
NUM_SAMPLES_TO_SHOW = 24 # Total number of images to display

def visualize_ground_truth(loader, num_samples):
    """
    Visualizes the dataset inputs and ground truth heatmaps.
    """
    # Get a batch of data
    for i, (inputs, targets) in enumerate(loader):
        # We only need to iterate enough times to show the desired number of samples
        if i * loader.batch_size >= num_samples:
            break

        # Move tensors to CPU for numpy conversion and plotting
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()

        # Iterate through the samples in the batch
        for j in range(inputs.shape[0]):
            sample_idx = i * loader.batch_size + j
            if sample_idx >= num_samples:
                break

            depth_map = inputs[j, 0, :, :]      # First channel is depth
            confidence_map = inputs[j, 1, :, :] # Second channel is confidence
            
            # To visualize all keypoints at once, we can sum the heatmaps
            summed_heatmaps = np.sum(targets[j], axis=0)
            
            # --- Plotting ---
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(f'Sample #{sample_idx}', fontsize=16)

            # 1. Plot Padded Depth Map
            im1 = axs[0].imshow(depth_map, cmap='viridis')
            axs[0].set_title('Padded Depth Map (Input Ch 1)')
            axs[0].axis('off')
            fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

            # 2. Plot Padded Confidence Map
            im2 = axs[1].imshow(confidence_map, cmap='magma')
            axs[1].set_title('Padded Confidence Map (Input Ch 2)')
            axs[1].axis('off')
            fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

            # 3. Plot Summed Ground Truth Heatmaps
            im3 = axs[2].imshow(summed_heatmaps, cmap='hot')
            axs[2].set_title('Summed Heatmaps (Ground Truth)')
            axs[2].axis('off')
            fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

            # 4. Plot Overlay
            axs[3].imshow(depth_map, cmap='viridis')
            # Use alpha for transparency. 'hot' colormap works well for overlays.
            axs[3].imshow(summed_heatmaps, cmap='hot', alpha=0.6)
            axs[3].set_title('Overlay: Heatmaps on Depth')
            axs[3].axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    print("Initializing dataset for visualization...")
    
    # Setup the dataset and dataloader
    # Ensure you have a 'train' or 'val' folder inside your DATA_DIR
    vis_dataset_path = os.path.join(DATA_DIR, DATASET_TO_VISUALIZE)
    if not os.path.exists(vis_dataset_path):
        raise FileNotFoundError(f"Could not find the dataset directory: {vis_dataset_path}. Please check your DATA_DIR and folder structure.")

    dataset = PoseDataset(data_dir=vis_dataset_path, output_res=(240, 240))
    # We use shuffle=True to see a random variety of samples each time we run.
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Found {len(dataset)} samples. Visualizing {min(NUM_SAMPLES_TO_SHOW, len(dataset))} of them...")
    
    visualize_ground_truth(dataloader, NUM_SAMPLES_TO_SHOW)
    
    print("Visualization finished.")