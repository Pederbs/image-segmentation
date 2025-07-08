#!/usr/bin/env python3
"""
Script to verify that train and test splits have no overlapping dates
"""

import os
from collections import defaultdict

def get_dates_from_directory(dir_path):
    """Extract all dates from files in a directory"""
    dates = set()
    
    if not os.path.exists(dir_path):
        return dates
        
    # Check all subdirectories
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        if os.path.isdir(subdir_path):
            # Get all files in this subdirectory
            files = os.listdir(subdir_path)
            for file in files:
                if file.endswith('.png'):
                    # Extract date from filename (format: YYYY-MM-DD.png)
                    date_part = file.replace('.png', '')
                    if len(date_part) == 10 and date_part.count('-') == 2:
                        dates.add(date_part)
    
    return dates

def main():
    base_path = '/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p'
    
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    
    print("Verifying train/test split...")
    
    # Get dates from train and test
    train_dates = get_dates_from_directory(train_path)
    test_dates = get_dates_from_directory(test_path)
    
    print(f"Train dates: {len(train_dates)}")
    print(f"Test dates: {len(test_dates)}")
    
    # Check for overlaps
    overlap = train_dates.intersection(test_dates)
    
    if overlap:
        print(f"❌ ERROR: Found {len(overlap)} overlapping dates!")
        print(f"Overlapping dates: {sorted(list(overlap))[:10]}...")
    else:
        print("✅ SUCCESS: No overlapping dates found!")
    
    # Show date ranges
    if train_dates:
        train_sorted = sorted(list(train_dates))
        print(f"Train date range: {train_sorted[0]} to {train_sorted[-1]}")
    
    if test_dates:
        test_sorted = sorted(list(test_dates))
        print(f"Test date range: {test_sorted[0]} to {test_sorted[-1]}")
    
    # Count files per split
    train_count = 0
    test_count = 0
    
    if os.path.exists(train_path):
        for subdir in os.listdir(train_path):
            subdir_path = os.path.join(train_path, subdir)
            if os.path.isdir(subdir_path):
                train_count += len([f for f in os.listdir(subdir_path) if f.endswith('.png')])
    
    if os.path.exists(test_path):
        for subdir in os.listdir(test_path):
            subdir_path = os.path.join(test_path, subdir)
            if os.path.isdir(subdir_path):
                test_count += len([f for f in os.listdir(subdir_path) if f.endswith('.png')])
    
    total_files = train_count + test_count
    if total_files > 0:
        train_ratio = train_count / total_files
        print(f"\nFile distribution:")
        print(f"Train: {train_count} files ({train_ratio:.1%})")
        print(f"Test: {test_count} files ({1-train_ratio:.1%})")

if __name__ == "__main__":
    main()
