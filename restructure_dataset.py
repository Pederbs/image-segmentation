#!/usr/bin/env python3
"""
Script to restructure the dataset into train/test splits with 80/20 ratio.
Ensures that all images from the same date stay in the same split.
"""

import os
import shutil
import glob
from collections import defaultdict
import random
from datetime import datetime

def get_unique_dates(base_path):
    """Extract all unique dates from the dataset"""
    dates = set()
    
    # Check all subdirectories
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            # Get all files in this subdirectory
            files = os.listdir(subdir_path)
            for file in files:
                if file.endswith('.png'):
                    # Handle two naming patterns:
                    # 1. YYYY-MM-DD.png (most directories)
                    # 2. variable_YYYY-MM-DD.png (catch directory)
                    
                    if '_' in file:
                        # Format: variable_YYYY-MM-DD.png
                        parts = file.split('_')
                        if len(parts) >= 2:
                            date_part = parts[1].replace('.png', '')
                            if len(date_part) == 10 and date_part.count('-') == 2:
                                dates.add(date_part)
                    else:
                        # Format: YYYY-MM-DD.png
                        date_part = file.replace('.png', '')
                        if len(date_part) == 10 and date_part.count('-') == 2:
                            dates.add(date_part)
    
    return sorted(list(dates))

def split_dates(dates, train_ratio=0.8):
    """Split dates into train and test sets"""
    random.shuffle(dates)
    split_point = int(len(dates) * train_ratio)
    train_dates = dates[:split_point]
    test_dates = dates[split_point:]
    return train_dates, test_dates

def create_new_structure(base_path, train_dates, test_dates):
    """Create the new train/test directory structure"""
    
    # Create main train and test directories
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all variable subdirectories
    subdirs = [d for d in os.listdir(base_path) 
              if os.path.isdir(os.path.join(base_path, d)) and d not in ['train', 'test']]
    
    # Create subdirectories in train and test
    for subdir in subdirs:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
    
    # Move files based on their dates
    train_count = 0
    test_count = 0
    
    for subdir in subdirs:
        source_dir = os.path.join(base_path, subdir)
        
        # Process all files in this subdirectory
        files = os.listdir(source_dir)
        
        for file in files:
            if file.endswith('.png'):
                # Handle two naming patterns:
                # 1. YYYY-MM-DD.png (most directories)
                # 2. variable_YYYY-MM-DD.png (catch directory)
                
                date_part = None
                if '_' in file:
                    # Format: variable_YYYY-MM-DD.png
                    parts = file.split('_')
                    if len(parts) >= 2:
                        date_part = parts[1].replace('.png', '')
                else:
                    # Format: YYYY-MM-DD.png
                    date_part = file.replace('.png', '')
                
                # Validate it's a date format and process
                if date_part and len(date_part) == 10 and date_part.count('-') == 2:
                    source_path = os.path.join(source_dir, file)
                    
                    # Normalize filename - if it has variable prefix, remove it
                    if '_' in file:
                        # For catch files: catch_YYYY-MM-DD.png -> YYYY-MM-DD.png
                        dest_filename = f"{date_part}.png"
                    else:
                        # For other files: keep original filename
                        dest_filename = file
                    
                    if date_part in train_dates:
                        dest_path = os.path.join(train_dir, subdir, dest_filename)
                        shutil.copy2(source_path, dest_path)
                        train_count += 1
                    elif date_part in test_dates:
                        dest_path = os.path.join(test_dir, subdir, dest_filename)
                        shutil.copy2(source_path, dest_path)
                        test_count += 1
    
    return train_count, test_count, subdirs

def main():
    base_path = '/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p'
    
    print("Starting dataset restructuring...")
    print(f"Base directory: {base_path}")
    
    # Get all unique dates
    dates = get_unique_dates(base_path)
    print(f"Found {len(dates)} unique dates")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Split dates into train/test
    train_dates, test_dates = split_dates(dates.copy(), train_ratio=0.8)
    
    print(f"Train dates: {len(train_dates)} ({len(train_dates)/len(dates)*100:.1f}%)")
    print(f"Test dates: {len(test_dates)} ({len(test_dates)/len(dates)*100:.1f}%)")
    
    # Create new structure and move files
    train_count, test_count, subdirs = create_new_structure(base_path, train_dates, test_dates)
    
    print(f"\nRestructuring complete!")
    print(f"Variables processed: {len(subdirs)}")
    print(f"Files copied to train: {train_count}")
    print(f"Files copied to test: {test_count}")
    print(f"Total files: {train_count + test_count}")
    
    # Verify the split
    actual_train_ratio = train_count / (train_count + test_count) if (train_count + test_count) > 0 else 0
    print(f"Actual train/test ratio: {actual_train_ratio:.1%}/{1-actual_train_ratio:.1%}")
    
    # Show some sample train and test dates
    print(f"\nSample train dates: {train_dates[:5]}")
    print(f"Sample test dates: {test_dates[:5]}")
    
    print(f"\nNew structure created:")
    print(f"- {base_path}/train/")
    print(f"- {base_path}/test/")
    print(f"Each containing subdirectories: {', '.join(subdirs)}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
