#!/usr/bin/env python3
"""
Skagerakk Dataset Pipeline Automation Script

This script automates the entire pipeline for creating the skagerakk_dataset:
1. Restructures the original dataset into train/test splits (80/20) with no date overlap
2. Combines three oceanographic variables (thetao, so, o2) into RGB images
3. Creates keras-segmentation compatible dataset structure
4. Generates comprehensive visualizations and saves all plots
5. Validates the final dataset
6. Optional: Resizes all images to specified dimensions

Usage:
    python automate_pipeline.py [--skip-restructure] [--output-dir OUTPUT_DIR] [--resize WIDTH HEIGHT]
    
Arguments:
    --skip-restructure    Skip the initial dataset restructuring step
    --output-dir         Custom output directory for plots (default: current directory)
    --resize             Resize all images to specified dimensions (e.g., --resize 256 256)

Examples:
    python automate_pipeline.py
    python automate_pipeline.py --resize 256 256
    python automate_pipeline.py --skip-restructure --output-dir results --resize 512 512
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import shutil
warnings.filterwarnings('ignore')

# Configuration
ORIGINAL_DATASET = "/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p"
INTERMEDIATE_DATASET = "/home/anna/msc_oppgave/image-segmentation/intermediate_dataset"
FINAL_DATASET = "/home/anna/msc_oppgave/image-segmentation/skagerakk_dataset"
INPUT_VARIABLES = ['thetao', 'so', 'o2']
TARGET_VARIABLE = 'catch'
EXPECTED_VARIABLES = ['catch', 'chl', 'kd', 'no3', 'nppv', 'o2', 'phyc', 'po4', 'si', 'so', 'thetao', 'uo', 'vo', 'zooc']

def create_plots_directory(output_dir):
    """Create plots directory structure"""
    plots_dir = os.path.join(output_dir, "plots")
    subdirs = ["original_data", "restructured_data", "rgb_creation", "final_dataset", "validation"]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(plots_dir, subdir), exist_ok=True)
    
    return plots_dir

def log_progress(message, log_file="pipeline_log.txt"):
    """Log progress messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    
    with open(log_file, "a") as f:
        f.write(log_message + "\n")

def analyze_original_dataset(dataset_path, plots_dir):
    """Analyze and visualize the original dataset structure"""
    log_progress("Analyzing original dataset structure...")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        log_progress(f"ERROR: Dataset path {dataset_path} does not exist!")
        return False
    
    # Analyze directory structure
    variables = []
    file_counts = {}
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item in EXPECTED_VARIABLES:
            variables.append(item)
            files = [f for f in os.listdir(item_path) if f.endswith('.png')]
            file_counts[item] = len(files)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Variables bar chart
    ax1.bar(variables, [file_counts[var] for var in variables])
    ax1.set_title('Original Dataset: Files per Variable')
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Number of Files')
    ax1.tick_params(axis='x', rotation=45)
    
    # Sample image from each variable
    sample_images = []
    for var in INPUT_VARIABLES + [TARGET_VARIABLE]:
        if var in variables:
            var_path = os.path.join(dataset_path, var)
            png_files = [f for f in os.listdir(var_path) if f.endswith('.png')]
            if png_files:  # Check if there are any PNG files
                sample_file = png_files[0]
                sample_path = os.path.join(var_path, sample_file)
                try:
                    img = Image.open(sample_path)
                    sample_images.append((var, np.array(img)))
                except Exception as e:
                    log_progress(f"Warning: Could not load sample image {sample_path}: {e}")
            else:
                log_progress(f"Warning: No PNG files found in {var_path}")
    
    # Show sample images in a grid
    if sample_images:
        ax2.set_title('Sample Images from Each Variable')
        ax2.axis('off')
        
        # Create small subplots for each variable
        for i, (var, img_array) in enumerate(sample_images):
            # Use first channel if RGBA
            if len(img_array.shape) == 3:
                img_array = img_array[:, :, 0]
            
            # Create small subplot
            rect = patches.Rectangle((i*0.2, 0.2), 0.18, 0.6, transform=ax2.transAxes, 
                                   facecolor='none', edgecolor='black')
            ax2.add_patch(rect)
            
            # Add text label
            ax2.text(i*0.2 + 0.09, 0.1, var, transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "original_data", "dataset_overview.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis results
    analysis_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "variables_found": variables,
        "file_counts": file_counts,
        "total_files": sum(file_counts.values()),
        "input_variables": INPUT_VARIABLES,
        "target_variable": TARGET_VARIABLE
    }
    
    with open(os.path.join(plots_dir, "original_data", "analysis_results.json"), "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    log_progress(f"Original dataset analysis complete. Found {len(variables)} variables with {sum(file_counts.values())} total files.")
    return True

def restructure_dataset(source_path, plots_dir):
    """Restructure dataset into train/test splits with no date overlap"""
    log_progress("Restructuring dataset into train/test splits...")
    
    # Create intermediate dataset directory (DO NOT MODIFY ORIGINAL)
    intermediate_dir = INTERMEDIATE_DATASET
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Get all unique dates from the catch variable IN THE ORIGINAL DATASET
    catch_dir = os.path.join(source_path, TARGET_VARIABLE)
    if not os.path.exists(catch_dir):
        log_progress(f"ERROR: Target variable directory {catch_dir} not found!")
        return False
    
    # Extract dates from filenames
    all_files = [f for f in os.listdir(catch_dir) if f.endswith('.png')]
    dates = []
    
    for file in all_files:
        # Extract date from filename - handle both formats:
        # Format 1: catch_2019-08-22.png (with prefix)
        # Format 2: 2019-08-22.png (without prefix)
        try:
            filename_no_ext = file.replace('.png', '')
            
            # Check if file has variable prefix (e.g., "catch_")
            if '_' in filename_no_ext:
                # Split on underscore and take the date part
                parts = filename_no_ext.split('_')
                if len(parts) >= 2:
                    date_part = '_'.join(parts[1:])  # Everything after first underscore
                    # Try to parse as YYYY-MM-DD
                    if len(date_part.split('-')) == 3:
                        dates.append(date_part)
                    else:
                        # Try to parse as YYYY_MM_DD format
                        date_components = date_part.split('_')
                        if len(date_components) >= 3:
                            date_str = f"{date_components[0]}-{date_components[1]}-{date_components[2]}"
                            dates.append(date_str)
            else:
                # No prefix, assume it's just the date
                if len(filename_no_ext.split('-')) == 3:
                    dates.append(filename_no_ext)
                    
        except Exception as e:
            log_progress(f"Warning: Could not parse date from {file}: {e}")
    
    log_progress(f"Extracted {len(dates)} dates from {len(all_files)} files")
    
    # Sort dates and split 80/20
    dates = sorted(list(set(dates)))
    
    if len(dates) == 0:
        log_progress("ERROR: No valid dates found in the dataset!")
        return False
    
    split_point = int(len(dates) * 0.8)
    train_dates = dates[:split_point]
    test_dates = dates[split_point:]
    
    log_progress(f"Found {len(dates)} unique dates. Train: {len(train_dates)}, Test: {len(test_dates)}")
    
    # Create train/test directory structure IN INTERMEDIATE DIRECTORY
    train_dir = os.path.join(intermediate_dir, "train")
    test_dir = os.path.join(intermediate_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each variable
    for var in EXPECTED_VARIABLES:
        var_source = os.path.join(source_path, var)
        if not os.path.exists(var_source):
            continue
        
        # Create variable directories in train/test
        train_var_dir = os.path.join(train_dir, var)
        test_var_dir = os.path.join(test_dir, var)
        os.makedirs(train_var_dir, exist_ok=True)
        os.makedirs(test_var_dir, exist_ok=True)
        
        # Find files - they might be in subdirectories
        var_files = []
        
        # Check if files are directly in the variable directory
        if os.path.isdir(var_source):
            direct_files = [f for f in os.listdir(var_source) if f.endswith('.png')]
            if direct_files:
                # Files are directly in the variable directory
                var_files = [(var_source, f) for f in direct_files]
            else:
                # Files might be in subdirectories
                for subdir in os.listdir(var_source):
                    subdir_path = os.path.join(var_source, subdir)
                    if os.path.isdir(subdir_path):
                        subdir_files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
                        var_files.extend([(subdir_path, f) for f in subdir_files])
        
        train_count = 0
        test_count = 0
        
        for source_dir, file in var_files:
            # Extract date from filename - handle both prefixed and clean filenames
            try:
                filename_no_ext = file.replace('.png', '')
                
                # Check if file has variable prefix (e.g., "catch_")
                if '_' in filename_no_ext and filename_no_ext.startswith(var + '_'):
                    # Split on underscore and take the date part
                    parts = filename_no_ext.split('_')
                    if len(parts) >= 2:
                        date_part = '_'.join(parts[1:])  # Everything after first underscore
                        # Try to parse as YYYY-MM-DD
                        if len(date_part.split('-')) == 3:
                            date_str = date_part
                        else:
                            # Try to parse as YYYY_MM_DD format
                            date_components = date_part.split('_')
                            if len(date_components) >= 3:
                                date_str = f"{date_components[0]}-{date_components[1]}-{date_components[2]}"
                            else:
                                continue
                else:
                    # No prefix or different format, assume it's already clean (YYYY-MM-DD)
                    if len(filename_no_ext.split('-')) == 3:
                        date_str = filename_no_ext
                    else:
                        continue
                
                # Standardize filename to just the date
                new_filename = f"{date_str}.png"
                
                source_file = os.path.join(source_dir, file)
                
                if date_str in train_dates:
                    dest_file = os.path.join(train_var_dir, new_filename)
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
                        train_count += 1
                elif date_str in test_dates:
                    dest_file = os.path.join(test_var_dir, new_filename)
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, dest_file)
                        test_count += 1
            except Exception as e:
                log_progress(f"Warning: Could not process {file}: {e}")
        
        log_progress(f"  {var}: {train_count} train, {test_count} test files")
    
    # Create visualization of the split
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Date distribution
    if len(train_dates) > 0 and len(test_dates) > 0:
        train_years = [int(d.split('-')[0]) for d in train_dates]
        test_years = [int(d.split('-')[0]) for d in test_dates]
        
        ax1.hist([train_years, test_years], bins=20, alpha=0.7, label=['Train', 'Test'])
        ax1.set_title('Date Distribution Across Train/Test Splits')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Samples')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No dates to display', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Date Distribution (No Data)')
    
    # Split summary
    if len(dates) > 0:
        split_info = {
            'Train Dates': len(train_dates),
            'Test Dates': len(test_dates),
            'Train %': len(train_dates) / len(dates) * 100,
            'Test %': len(test_dates) / len(dates) * 100
        }
        
        ax2.bar(split_info.keys(), split_info.values())
        ax2.set_title('Train/Test Split Summary')
        ax2.set_ylabel('Count / Percentage')
    else:
        ax2.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Split Summary (No Data)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "restructured_data", "train_test_split.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    log_progress("Dataset restructuring complete.")
    return True

def normalize_array(arr):
    """Normalize array to 0-255 range for image creation"""
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min) * 255
    return normalized.astype(np.uint8)

def resize_image(image, target_size):
    """Resize image to target size using PIL"""
    target_width, target_height = target_size
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            # Grayscale image
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            # RGB image
            pil_image = Image.fromarray(image.astype(np.uint8))
    else:
        pil_image = image
    
    # Resize using PIL
    resized_pil = pil_image.resize((target_width, target_height), Image.LANCZOS)
    
    # Convert back to numpy array
    resized_array = np.array(resized_pil)
    
    return resized_array

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

def create_rgb_image(date, base_path, variables, resize_dims=None):
    """Create RGB image from three variables for a specific date"""
    channels = []
    
    for var in variables:
        var_path = os.path.join(base_path, var, f"{date}.png")
        if os.path.exists(var_path):
            channel = load_and_process_variable(var_path)
            if channel is not None:
                # Resize channel if requested
                if resize_dims:
                    channel = resize_image(channel, resize_dims)
                channels.append(channel)
            else:
                log_progress(f"Failed to load {var_path}")
                return None
        else:
            log_progress(f"File not found: {var_path}")
            return None
    
    if len(channels) == 3:
        # Ensure all channels have the same shape
        if resize_dims:
            # All channels should already be the same size after resizing
            min_height, min_width = resize_dims[1], resize_dims[0]
        else:
            min_height = min(ch.shape[0] for ch in channels)
            min_width = min(ch.shape[1] for ch in channels)
        
        # Crop all channels to the same size
        cropped_channels = []
        for ch in channels:
            if resize_dims:
                # Already resized, just use as is
                cropped_channels.append(ch)
            else:
                cropped = ch[:min_height, :min_width]
                cropped_channels.append(cropped)
        
        # Stack channels to create RGB image
        rgb_image = np.stack(cropped_channels, axis=-1)
        return rgb_image
    else:
        return None

def create_final_dataset(source_path, output_path, plots_dir, resize_dims=None):
    """Create the final skagerakk_dataset with RGB images and target annotations"""
    if resize_dims:
        log_progress(f"Creating final dataset with RGB images (resized to {resize_dims[0]}x{resize_dims[1]})...")
    else:
        log_progress("Creating final dataset with RGB images...")
    
    # Create output directory structure
    train_images_dir = os.path.join(output_path, "images_prepped_train")
    train_annotations_dir = os.path.join(output_path, "annotations_prepped_train")
    test_images_dir = os.path.join(output_path, "images_prepped_test")
    test_annotations_dir = os.path.join(output_path, "annotations_prepped_test")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_annotations_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)
    os.makedirs(test_annotations_dir, exist_ok=True)
    
    # Process train set
    train_path = os.path.join(source_path, "train")
    train_success = 0
    train_samples = []
    
    if os.path.exists(train_path):
        first_var_train = os.path.join(train_path, INPUT_VARIABLES[0])
        if os.path.exists(first_var_train):
            train_dates = [f.replace('.png', '') for f in os.listdir(first_var_train) if f.endswith('.png')]
            train_dates.sort()
            
            log_progress(f"Processing {len(train_dates)} training images...")
            
            for date in tqdm(train_dates, desc="Creating train RGB images"):
                rgb_image = create_rgb_image(date, train_path, INPUT_VARIABLES, resize_dims)
                target_path = os.path.join(train_path, TARGET_VARIABLE, f"{date}.png")
                
                if rgb_image is not None and os.path.exists(target_path):
                    # Save RGB input image
                    input_file = os.path.join(train_images_dir, f"{date}.png")
                    pil_image = Image.fromarray(rgb_image, mode='RGB')
                    pil_image.save(input_file)
                    
                    # Load and potentially resize target annotation
                    target_img = Image.open(target_path)
                    if resize_dims:
                        target_img = target_img.resize(resize_dims, Image.NEAREST)  # Use nearest neighbor for annotations
                    
                    # Save target annotation
                    annotation_file = os.path.join(train_annotations_dir, f"{date}.png")
                    target_img.save(annotation_file)
                    
                    # Store sample for visualization
                    if len(train_samples) < 5:
                        train_samples.append((date, rgb_image, np.array(target_img)))
                    
                    train_success += 1
    
    # Process test set
    test_path = os.path.join(source_path, "test")
    test_success = 0
    test_samples = []
    
    if os.path.exists(test_path):
        first_var_test = os.path.join(test_path, INPUT_VARIABLES[0])
        if os.path.exists(first_var_test):
            test_dates = [f.replace('.png', '') for f in os.listdir(first_var_test) if f.endswith('.png')]
            test_dates.sort()
            
            log_progress(f"Processing {len(test_dates)} test images...")
            
            for date in tqdm(test_dates, desc="Creating test RGB images"):
                rgb_image = create_rgb_image(date, test_path, INPUT_VARIABLES, resize_dims)
                target_path = os.path.join(test_path, TARGET_VARIABLE, f"{date}.png")
                
                if rgb_image is not None and os.path.exists(target_path):
                    # Save RGB input image
                    input_file = os.path.join(test_images_dir, f"{date}.png")
                    pil_image = Image.fromarray(rgb_image, mode='RGB')
                    pil_image.save(input_file)
                    
                    # Load and potentially resize target annotation
                    target_img = Image.open(target_path)
                    if resize_dims:
                        target_img = target_img.resize(resize_dims, Image.NEAREST)  # Use nearest neighbor for annotations
                    
                    # Save target annotation
                    annotation_file = os.path.join(test_annotations_dir, f"{date}.png")
                    target_img.save(annotation_file)
                    
                    # Store sample for visualization
                    if len(test_samples) < 5:
                        test_samples.append((date, rgb_image, np.array(target_img)))
                    
                    test_success += 1
    
    # Create RGB creation visualization
    if train_samples:
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
        plt.savefig(os.path.join(plots_dir, "rgb_creation", "rgb_samples.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    log_progress(f"Dataset creation complete! Train: {train_success}, Test: {test_success}")
    return train_success, test_success

def validate_final_dataset(dataset_path, plots_dir):
    """Validate the final dataset and create comprehensive visualizations"""
    log_progress("Validating final dataset...")
    
    # Check dataset structure
    required_dirs = ["images_prepped_train", "annotations_prepped_train", 
                    "images_prepped_test", "annotations_prepped_test"]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            log_progress(f"ERROR: Required directory {dir_path} not found!")
            return False
    
    # Count files in each directory
    counts = {}
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        file_count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
        counts[dir_name] = file_count
    
    # Verify matching counts
    if counts["images_prepped_train"] != counts["annotations_prepped_train"]:
        log_progress("ERROR: Train images and annotations count mismatch!")
        return False
    
    if counts["images_prepped_test"] != counts["annotations_prepped_test"]:
        log_progress("ERROR: Test images and annotations count mismatch!")
        return False
    
    # Load sample images for analysis
    train_img_dir = os.path.join(dataset_path, "images_prepped_train")
    train_ann_dir = os.path.join(dataset_path, "annotations_prepped_train")
    
    sample_files = [f for f in os.listdir(train_img_dir) if f.endswith('.png')][:5]
    
    # Analyze samples
    rgb_stats = []
    target_stats = []
    
    for filename in sample_files:
        # Load RGB image
        rgb_path = os.path.join(train_img_dir, filename)
        rgb_img = Image.open(rgb_path)
        rgb_array = np.array(rgb_img)
        
        # Load target
        target_path = os.path.join(train_ann_dir, filename)
        target_img = Image.open(target_path)
        target_array = np.array(target_img)
        if len(target_array.shape) > 2:
            target_array = target_array[:, :, 0]
        
        rgb_stats.append({
            'filename': filename,
            'shape': rgb_array.shape,
            'channels': [float(rgb_array[:, :, i].mean()) for i in range(3)],
            'range': [float(rgb_array.min()), float(rgb_array.max())]
        })
        
        target_stats.append({
            'filename': filename,
            'shape': target_array.shape,
            'mean': float(target_array.mean()),
            'std': float(target_array.std()),
            'range': [float(target_array.min()), float(target_array.max())]
        })
    
    # Create comprehensive validation plots
    fig = plt.figure(figsize=(20, 16))
    
    # Dataset overview
    ax1 = plt.subplot(4, 4, 1)
    categories = ['Train Images', 'Train Annotations', 'Test Images', 'Test Annotations']
    values = [counts[dir_name] for dir_name in required_dirs]
    ax1.bar(categories, values)
    ax1.set_title('Dataset File Counts')
    ax1.tick_params(axis='x', rotation=45)
    
    # RGB channel distribution
    ax2 = plt.subplot(4, 4, 2)
    channel_means = [[stat['channels'][i] for stat in rgb_stats] for i in range(3)]
    ax2.boxplot(channel_means, labels=[f'{var}' for var in INPUT_VARIABLES])
    ax2.set_title('RGB Channel Value Distribution')
    ax2.set_ylabel('Mean Channel Value')
    
    # Target value distribution
    ax3 = plt.subplot(4, 4, 3)
    target_means = [stat['mean'] for stat in target_stats]
    ax3.hist(target_means, bins=10, alpha=0.7)
    ax3.set_title(f'{TARGET_VARIABLE} Value Distribution')
    ax3.set_xlabel('Mean Value')
    ax3.set_ylabel('Frequency')
    
    # Sample images grid
    if len(sample_files) >= 3:
        for i in range(3):
            filename = sample_files[i]
            
            # Load images
            rgb_path = os.path.join(train_img_dir, filename)
            rgb_img = np.array(Image.open(rgb_path))
            
            target_path = os.path.join(train_ann_dir, filename)
            target_img = np.array(Image.open(target_path))
            if len(target_img.shape) > 2:
                target_img = target_img[:, :, 0]
            
            # RGB image
            ax_rgb = plt.subplot(4, 4, 4 + i*4)
            ax_rgb.imshow(rgb_img)
            ax_rgb.set_title(f'RGB Input\n{filename}')
            ax_rgb.axis('off')
            
            # Individual channels
            for j in range(3):
                ax_ch = plt.subplot(4, 4, 5 + i*4 + j)
                ax_ch.imshow(rgb_img[:, :, j], cmap='viridis')
                ax_ch.set_title(f'{INPUT_VARIABLES[j]}')
                ax_ch.axis('off')
            
            # Target
            ax_target = plt.subplot(4, 4, 8 + i*4)
            ax_target.imshow(target_img, cmap='viridis')
            ax_target.set_title(f'{TARGET_VARIABLE}')
            ax_target.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "validation", "dataset_validation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dataset summary
    summary_text = f"""
Dataset Summary:
- Total train samples: {counts['images_prepped_train']}
- Total test samples: {counts['images_prepped_test']}
- Train/Test ratio: {counts['images_prepped_train']/(counts['images_prepped_train']+counts['images_prepped_test'])*100:.1f}%/{counts['images_prepped_test']/(counts['images_prepped_train']+counts['images_prepped_test'])*100:.1f}%
- Input variables: {', '.join(INPUT_VARIABLES)} (RGB channels)
- Target variable: {TARGET_VARIABLE}
- Image format: RGB PNG files
- Ready for training: ✓
"""
    
    ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace')
    ax1.set_title('Dataset Summary')
    ax1.axis('off')
    
    # RGB channel statistics
    if rgb_stats:
        for i, var in enumerate(INPUT_VARIABLES):
            channel_values = [stat['channels'][i] for stat in rgb_stats]
            ax2.hist(channel_values, bins=10, alpha=0.7, label=var)
        ax2.set_title('RGB Channel Value Distribution')
        ax2.set_xlabel('Mean Channel Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    
    # Target statistics
    if target_stats:
        target_values = [stat['mean'] for stat in target_stats]
        ax3.hist(target_values, bins=15, alpha=0.7, color='green')
        ax3.set_title(f'{TARGET_VARIABLE} Value Distribution')
        ax3.set_xlabel('Mean Value')
        ax3.set_ylabel('Frequency')
    
    # Image shape consistency
    shapes = [stat['shape'] for stat in rgb_stats]
    unique_shapes = list(set(shapes))
    ax4.bar(range(len(unique_shapes)), [shapes.count(shape) for shape in unique_shapes])
    ax4.set_title('Image Shape Consistency')
    ax4.set_xlabel('Shape Index')
    ax4.set_ylabel('Count')
    shape_labels = [f'{shape}' for shape in unique_shapes]
    ax4.set_xticks(range(len(unique_shapes)))
    ax4.set_xticklabels(shape_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "validation", "dataset_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save validation results
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "file_counts": counts,
        "input_variables": INPUT_VARIABLES,
        "target_variable": TARGET_VARIABLE,
        "sample_rgb_stats": rgb_stats,
        "sample_target_stats": target_stats,
        "validation_passed": True
    }
    
    with open(os.path.join(plots_dir, "validation", "validation_results.json"), "w") as f:
        json.dump(validation_results, f, indent=2)
    
    log_progress(f"Dataset validation complete! Train: {counts['images_prepped_train']}, Test: {counts['images_prepped_test']}")
    return True

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Skagerakk Dataset Pipeline Automation")
    parser.add_argument("--skip-restructure", action="store_true", 
                       help="Skip the dataset restructuring step")
    parser.add_argument("--output-dir", default=".", 
                       help="Output directory for plots and logs")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help="Resize all images to specified dimensions (e.g., --resize 256 256)")
    
    args = parser.parse_args()
    
    # Create output directory
    plots_dir = create_plots_directory(args.output_dir)
    
    # Initialize log
    log_file = os.path.join(args.output_dir, "pipeline_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_progress("=== Skagerakk Dataset Pipeline Started ===")
    log_progress(f"Output directory: {args.output_dir}")
    log_progress(f"Plots directory: {plots_dir}")
    
    # Handle resize option
    resize_dims = None
    if args.resize:
        resize_dims = tuple(args.resize)
        log_progress(f"Images will be resized to: {resize_dims[0]}x{resize_dims[1]}")
    
    # Step 1: Analyze original dataset
    if not analyze_original_dataset(ORIGINAL_DATASET, plots_dir):
        log_progress("ERROR: Failed to analyze original dataset!")
        return 1
    
    # Step 2: Restructure dataset (optional)
    if not args.skip_restructure:
        if not restructure_dataset(ORIGINAL_DATASET, plots_dir):
            log_progress("ERROR: Failed to restructure dataset!")
            return 1
    else:
        log_progress("Skipping dataset restructuring step.")
    
    # Step 3: Create final dataset
    train_count, test_count = create_final_dataset(INTERMEDIATE_DATASET, FINAL_DATASET, plots_dir, resize_dims)
    if train_count == 0 and test_count == 0:
        log_progress("ERROR: Failed to create final dataset!")
        return 1
    
    # Step 4: Validate final dataset
    if not validate_final_dataset(FINAL_DATASET, plots_dir):
        log_progress("ERROR: Dataset validation failed!")
        return 1
    
    # Step 5: Create final summary
    resize_info = f"- Image dimensions: {resize_dims[0]}x{resize_dims[1]} (resized)\n" if resize_dims else "- Image dimensions: Original size (not resized)\n"
    
    summary_text = f"""
=== PIPELINE COMPLETE ===

Dataset Creation Summary:
- Original dataset: {ORIGINAL_DATASET}
- Final dataset: {FINAL_DATASET}
- Train samples: {train_count}
- Test samples: {test_count}
- Total samples: {train_count + test_count}
- Input variables: {', '.join(INPUT_VARIABLES)} (combined as RGB)
- Target variable: {TARGET_VARIABLE}
{resize_info}
Files created:
- {FINAL_DATASET}/images_prepped_train/ ({train_count} RGB images)
- {FINAL_DATASET}/annotations_prepped_train/ ({train_count} target images)
- {FINAL_DATASET}/images_prepped_test/ ({test_count} RGB images)
- {FINAL_DATASET}/annotations_prepped_test/ ({test_count} target images)

Plots saved in: {plots_dir}
- original_data/: Original dataset analysis
- restructured_data/: Train/test split analysis
- rgb_creation/: RGB combination samples
- validation/: Final dataset validation

Next steps:
1. Review all plots in {plots_dir}
2. Train your model using the skagerakk_dataset
3. Use keras-segmentation with the prepared dataset structure

Dataset is ready for training!
"""
    
    log_progress(summary_text)
    
    # Save final summary
    with open(os.path.join(args.output_dir, "pipeline_summary.txt"), "w") as f:
        f.write(summary_text)
    
    log_progress("=== Pipeline completed successfully! ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
