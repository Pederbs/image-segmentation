#!/usr/bin/env python3
"""
Simplified Skagerakk Dataset Pipeline

This script creates the final dataset and comprehensive visualizations.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
INTERMEDIATE_DATASET = "/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p"
FINAL_DATASET = "/home/anna/msc_oppgave/image-segmentation/skagerakk_dataset"
INPUT_VARIABLES = ['thetao', 'so', 'o2']
TARGET_VARIABLE = 'catch'

def log_progress(message):
    """Log progress messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def normalize_array(arr):
    """Normalize array to 0-255 range for image creation"""
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min) * 255
    return normalized.astype(np.uint8)

def load_and_process_variable(file_path):
    """Load a PNG file and convert to normalized array"""
    try:
        img = Image.open(file_path)
        arr = np.array(img)
        
        # Handle different image modes
        if img.mode == 'RGBA':
            arr = arr[:, :, 0]
        elif img.mode == 'RGB':
            arr = np.mean(arr, axis=2)
        elif len(arr.shape) == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        
        if len(arr.shape) > 2:
            arr = arr[:, :, 0]
            
        return normalize_array(arr)
    except Exception as e:
        log_progress(f"Error loading {file_path}: {e}")
        return None

def create_rgb_image(date, base_path, variables):
    """Create RGB image from three variables for a specific date"""
    channels = []
    
    for var in variables:
        var_path = os.path.join(base_path, var, f"{date}.png")
        if os.path.exists(var_path):
            channel = load_and_process_variable(var_path)
            if channel is not None:
                channels.append(channel)
            else:
                return None
        else:
            return None
    
    if len(channels) == 3:
        # Ensure all channels have the same shape
        min_height = min(ch.shape[0] for ch in channels)
        min_width = min(ch.shape[1] for ch in channels)
        
        # Crop all channels to the same size
        cropped_channels = []
        for ch in channels:
            cropped = ch[:min_height, :min_width]
            cropped_channels.append(cropped)
        
        # Stack channels to create RGB image
        rgb_image = np.stack(cropped_channels, axis=-1)
        return rgb_image
    else:
        return None

def create_final_dataset():
    """Create the final skagerakk_dataset with RGB images and target annotations"""
    log_progress("Creating final dataset with RGB images...")
    
    # Create plots directory
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "final_dataset"), exist_ok=True)
    
    # Create output directory structure
    train_images_dir = os.path.join(FINAL_DATASET, "images_prepped_train")
    train_annotations_dir = os.path.join(FINAL_DATASET, "annotations_prepped_train")
    test_images_dir = os.path.join(FINAL_DATASET, "images_prepped_test")
    test_annotations_dir = os.path.join(FINAL_DATASET, "annotations_prepped_test")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_annotations_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_annotations_dir, exist_ok=True)
    
    # Process train set
    train_path = os.path.join(INTERMEDIATE_DATASET, "train")
    train_success = 0
    train_samples = []
    
    if os.path.exists(train_path):
        first_var_train = os.path.join(train_path, INPUT_VARIABLES[0])
        if os.path.exists(first_var_train):
            train_dates = [f.replace('.png', '') for f in os.listdir(first_var_train) if f.endswith('.png')]
            train_dates.sort()
            
            log_progress(f"Processing {len(train_dates)} training images...")
            
            for date in tqdm(train_dates, desc="Creating train RGB images"):
                rgb_image = create_rgb_image(date, train_path, INPUT_VARIABLES)
                target_path = os.path.join(train_path, TARGET_VARIABLE, f"{date}.png")
                
                if rgb_image is not None and os.path.exists(target_path):
                    # Save RGB input image
                    input_file = os.path.join(train_images_dir, f"{date}.png")
                    pil_image = Image.fromarray(rgb_image, mode='RGB')
                    pil_image.save(input_file)
                    
                    # Copy target annotation
                    annotation_file = os.path.join(train_annotations_dir, f"{date}.png")
                    target_img = Image.open(target_path)
                    target_img.save(annotation_file)
                    
                    # Store sample for visualization
                    if len(train_samples) < 3:
                        train_samples.append((date, rgb_image, np.array(target_img)))
                    
                    train_success += 1
    
    # Process test set
    test_path = os.path.join(INTERMEDIATE_DATASET, "test")
    test_success = 0
    test_samples = []
    
    if os.path.exists(test_path):
        first_var_test = os.path.join(test_path, INPUT_VARIABLES[0])
        if os.path.exists(first_var_test):
            test_dates = [f.replace('.png', '') for f in os.listdir(first_var_test) if f.endswith('.png')]
            test_dates.sort()
            
            log_progress(f"Processing {len(test_dates)} test images...")
            
            for date in tqdm(test_dates, desc="Creating test RGB images"):
                rgb_image = create_rgb_image(date, test_path, INPUT_VARIABLES)
                target_path = os.path.join(test_path, TARGET_VARIABLE, f"{date}.png")
                
                if rgb_image is not None and os.path.exists(target_path):
                    # Save RGB input image
                    input_file = os.path.join(test_images_dir, f"{date}.png")
                    pil_image = Image.fromarray(rgb_image, mode='RGB')
                    pil_image.save(input_file)
                    
                    # Copy target annotation
                    annotation_file = os.path.join(test_annotations_dir, f"{date}.png")
                    target_img = Image.open(target_path)
                    target_img.save(annotation_file)
                    
                    # Store sample for visualization
                    if len(test_samples) < 3:
                        test_samples.append((date, rgb_image, np.array(target_img)))
                    
                    test_success += 1
    
    # Create visualizations
    if train_samples:
        log_progress("Creating visualization plots...")
        
        # RGB creation visualization
        fig, axes = plt.subplots(len(train_samples), 5, figsize=(20, 4*len(train_samples)))
        if len(train_samples) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (date, rgb_img, target_img) in enumerate(train_samples):
            # RGB image
            axes[i, 0].imshow(rgb_img)
            axes[i, 0].set_title(f'RGB Combined\n{date}')
            axes[i, 0].axis('off')
            
            # Individual channels
            for j, var in enumerate(INPUT_VARIABLES):
                axes[i, j+1].imshow(rgb_img[:, :, j], cmap='viridis')
                axes[i, j+1].set_title(f'{var}\n(Channel {j})')
                axes[i, j+1].axis('off')
            
            # Target
            if len(target_img.shape) > 2:
                target_img = target_img[:, :, 0]
            axes[i, 4].imshow(target_img, cmap='viridis')
            axes[i, 4].set_title(f'Target: {TARGET_VARIABLE}')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "final_dataset", "rgb_samples.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Dataset summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # File counts
        categories = ['Train Images', 'Train Annotations', 'Test Images', 'Test Annotations']
        counts = [train_success, train_success, test_success, test_success]
        ax1.bar(categories, counts)
        ax1.set_title('Dataset File Counts')
        ax1.tick_params(axis='x', rotation=45)
        
        # Channel distribution
        if len(train_samples) > 0:
            sample_rgb = train_samples[0][1]
            for i, var in enumerate(INPUT_VARIABLES):
                ax2.hist(sample_rgb[:, :, i].flatten(), bins=50, alpha=0.7, label=var)
            ax2.set_title('RGB Channel Value Distribution (Sample)')
            ax2.set_xlabel('Pixel Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # Target distribution
        if len(train_samples) > 0:
            sample_target = train_samples[0][2]
            if len(sample_target.shape) > 2:
                sample_target = sample_target[:, :, 0]
            ax3.hist(sample_target.flatten(), bins=50, alpha=0.7, color='green')
            ax3.set_title(f'{TARGET_VARIABLE} Value Distribution (Sample)')
            ax3.set_xlabel('Pixel Value')
            ax3.set_ylabel('Frequency')
        
        # Summary text
        summary_text = f"""Dataset Summary:
• Total train samples: {train_success}
• Total test samples: {test_success}
• Train/Test ratio: {train_success/(train_success+test_success)*100:.1f}%/{test_success/(train_success+test_success)*100:.1f}%
• Input variables: {', '.join(INPUT_VARIABLES)}
• Target variable: {TARGET_VARIABLE}
• Image format: RGB PNG
• Ready for training: ✓"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Dataset Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "final_dataset", "dataset_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        log_progress("Visualization plots saved to plots/final_dataset/")
    
    log_progress(f"Dataset creation complete! Train: {train_success}, Test: {test_success}")
    return train_success, test_success

def main():
    """Main pipeline execution"""
    log_progress("=== Skagerakk Dataset Pipeline Started ===")
    
    # Create final dataset
    train_count, test_count = create_final_dataset()
    
    if train_count > 0 or test_count > 0:
        summary_text = f"""
=== PIPELINE COMPLETE ===

Dataset Creation Summary:
- Final dataset: {FINAL_DATASET}
- Train samples: {train_count}
- Test samples: {test_count}
- Total samples: {train_count + test_count}
- Input variables: {', '.join(INPUT_VARIABLES)} (combined as RGB)
- Target variable: {TARGET_VARIABLE}

Files created:
- {FINAL_DATASET}/images_prepped_train/ ({train_count} RGB images)
- {FINAL_DATASET}/annotations_prepped_train/ ({train_count} target images)
- {FINAL_DATASET}/images_prepped_test/ ({test_count} RGB images)
- {FINAL_DATASET}/annotations_prepped_test/ ({test_count} target images)

Plots saved in: plots/final_dataset/
- rgb_samples.png: RGB combination examples
- dataset_summary.png: Dataset statistics and overview

Dataset is ready for training!
"""
        
        log_progress(summary_text)
        
        # Save summary
        with open("pipeline_summary.txt", "w") as f:
            f.write(summary_text)
        
        log_progress("=== Pipeline completed successfully! ===")
        return 0
    else:
        log_progress("ERROR: No samples were created!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
