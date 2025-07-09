#!/usr/bin/env python3
"""
Dataset Integrity Checker and Fixer for Image Segmentation

This script checks for size mismatches between images and their corresponding 
segmentation annotations, and provides options to fix these issues.

Usage:
    python fix_dataset.py --check-only
    python fix_dataset.py --fix
    python fix_dataset.py --fix --no-backup
"""

import os
import sys
import argparse
import shutil
from PIL import Image
from tqdm import tqdm


def check_dataset_integrity(images_dir, annotations_dir, target_size=(256, 256), verbose=True):
    """
    Check if all image-annotation pairs have matching dimensions and correct target size
    
    Args:
        images_dir (str): Directory containing input images
        annotations_dir (str): Directory containing annotation images
        target_size (tuple): Target size (width, height) for ML model
        verbose (bool): Whether to print progress information
    
    Returns:
        list: List of issues found
    """
    issues = []
    
    if not os.path.exists(images_dir):
        issues.append(f"Images directory does not exist: {images_dir}")
        return issues
    
    if not os.path.exists(annotations_dir):
        issues.append(f"Annotations directory does not exist: {annotations_dir}")
        return issues
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if verbose:
        print(f"Checking {len(image_files)} image-annotation pairs for target size {target_size}...")
    
    for img_file in tqdm(image_files, desc="Checking files", disable=not verbose):
        img_path = os.path.join(images_dir, img_file)
        
        # Handle different annotation file extensions
        ann_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        ann_path = os.path.join(annotations_dir, ann_file)
        
        # Check if annotation exists
        if not os.path.exists(ann_path):
            issues.append(f"Missing annotation for {img_file}")
            continue
            
        try:
            # Load images and check dimensions
            img = Image.open(img_path)
            ann = Image.open(ann_path)
            
            # Check if image size matches target size
            if img.size != target_size:
                issues.append(f"Image wrong size: {img_file} ({img.size}) should be {target_size}")
            
            # Check if annotation size matches target size
            if ann.size != target_size:
                issues.append(f"Annotation wrong size: {ann_file} ({ann.size}) should be {target_size}")
            
            # Check if image and annotation sizes match each other
            if img.size != ann.size:
                issues.append(f"Size mismatch: {img_file} ({img.size}) vs {ann_file} ({ann.size})")
                
        except Exception as e:
            issues.append(f"Error loading {img_file}: {str(e)}")
    
    return issues


def fix_dataset_issues(images_dir, annotations_dir, target_size=(256, 256), backup=True, verbose=True):
    """
    Fix size issues by resizing both images and annotations to target size
    
    Args:
        images_dir (str): Directory containing input images
        annotations_dir (str): Directory containing annotation images
        target_size (tuple): Target size (width, height) for ML model
        backup (bool): Whether to create backup of original files
        verbose (bool): Whether to print progress information
    
    Returns:
        int: Number of files fixed
    """
    
    if backup:
        # Create backup of both directories
        images_backup_dir = images_dir + "_backup"
        annotations_backup_dir = annotations_dir + "_backup"
        
        if not os.path.exists(images_backup_dir):
            if verbose:
                print(f"Creating images backup at {images_backup_dir}")
            shutil.copytree(images_dir, images_backup_dir)
        else:
            if verbose:
                print(f"Images backup already exists at {images_backup_dir}")
        
        if not os.path.exists(annotations_backup_dir):
            if verbose:
                print(f"Creating annotations backup at {annotations_backup_dir}")
            shutil.copytree(annotations_dir, annotations_backup_dir)
        else:
            if verbose:
                print(f"Annotations backup already exists at {annotations_backup_dir}")
    
    fixed_count = 0
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if verbose:
        print(f"Processing {len(image_files)} files to resize to {target_size}...")
    
    for img_file in tqdm(image_files, desc="Fixing files", disable=not verbose):
        img_path = os.path.join(images_dir, img_file)
        
        # Handle different annotation file extensions
        ann_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
        ann_path = os.path.join(annotations_dir, ann_file)
        
        if not os.path.exists(ann_path):
            continue
            
        try:
            # Load images
            img = Image.open(img_path)
            ann = Image.open(ann_path)
            
            needs_fixing = False
            
            # Resize image if needed
            if img.size != target_size:
                if verbose:
                    print(f"Resizing image {img_file}: {img.size} -> {target_size}")
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_resized.save(img_path)
                needs_fixing = True
            
            # Resize annotation if needed
            if ann.size != target_size:
                if verbose:
                    print(f"Resizing annotation {ann_file}: {ann.size} -> {target_size}")
                # Use NEAREST interpolation to preserve label values
                ann_resized = ann.resize(target_size, Image.NEAREST)
                ann_resized.save(ann_path)
                needs_fixing = True
            
            if needs_fixing:
                fixed_count += 1
                
        except Exception as e:
            if verbose:
                print(f"Error fixing {img_file}: {str(e)}")
    
    if verbose:
        print(f"Fixed {fixed_count} files")
    
    return fixed_count


def main():
    parser = argparse.ArgumentParser(
        description="Check and fix dataset integrity for image segmentation"
    )
    parser.add_argument(
        "--train-images", 
        default="skagerakk_dataset/images_prepped_train/",
        help="Path to training images directory"
    )
    parser.add_argument(
        "--train-annotations", 
        default="skagerakk_dataset/annotations_prepped_train/",
        help="Path to training annotations directory"
    )
    parser.add_argument(
        "--test-images", 
        default="skagerakk_dataset/images_prepped_test/",
        help="Path to test images directory"
    )
    parser.add_argument(
        "--test-annotations", 
        default="skagerakk_dataset/annotations_prepped_test/",
        help="Path to test annotations directory"
    )
    parser.add_argument(
        "--target-size",
        default="256,256",
        help="Target size for images and annotations (width,height)"
    )
    parser.add_argument(
        "--check-only", 
        action="store_true",
        help="Only check for issues, don't fix them"
    )
    parser.add_argument(
        "--fix", 
        action="store_true",
        help="Fix the issues found"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true",
        help="Don't create backup when fixing (use with caution)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # If neither check-only nor fix is specified, default to check-only
    if not args.check_only and not args.fix:
        args.check_only = True
    
    verbose = not args.quiet
    
    # Check training dataset
    if verbose:
        print("=" * 60)
        print("CHECKING TRAINING DATASET")
        print("=" * 60)
    
    train_issues = check_dataset_integrity(
        args.train_images, 
        args.train_annotations, 
        target_size=target_size,
        verbose=verbose
    )
    
    if train_issues:
        if verbose:
            print(f"\nFound {len(train_issues)} issues in training dataset:")
            for issue in train_issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(train_issues) > 10:
                print(f"  ... and {len(train_issues) - 10} more issues")
    else:
        if verbose:
            print("✓ Training dataset integrity check passed!")
    
    # Check test dataset
    if verbose:
        print("\n" + "=" * 60)
        print("CHECKING TEST DATASET")
        print("=" * 60)
    
    test_issues = check_dataset_integrity(
        args.test_images, 
        args.test_annotations, 
        target_size=target_size,
        verbose=verbose
    )
    
    if test_issues:
        if verbose:
            print(f"\nFound {len(test_issues)} issues in test dataset:")
            for issue in test_issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(test_issues) > 10:
                print(f"  ... and {len(test_issues) - 10} more issues")
    else:
        if verbose:
            print("✓ Test dataset integrity check passed!")
    
    # Fix issues if requested
    if args.fix:
        if verbose:
            print("\n" + "=" * 60)
            print("FIXING DATASET ISSUES")
            print("=" * 60)
        
        total_fixed = 0
        
        if train_issues:
            if verbose:
                print("Fixing training dataset issues...")
            fixed_train = fix_dataset_issues(
                args.train_images,
                args.train_annotations,
                target_size=target_size,
                backup=not args.no_backup,
                verbose=verbose
            )
            total_fixed += fixed_train
            
            # Re-check after fixing
            if verbose:
                print("\nRe-checking training dataset after fixes...")
            train_issues_after = check_dataset_integrity(
                args.train_images,
                args.train_annotations,
                target_size=target_size,
                verbose=False
            )
            
            if not train_issues_after:
                if verbose:
                    print("✓ Training dataset issues resolved!")
            else:
                if verbose:
                    print(f"⚠ {len(train_issues_after)} issues remain in training dataset")
        
        if test_issues:
            if verbose:
                print("\nFixing test dataset issues...")
            fixed_test = fix_dataset_issues(
                args.test_images,
                args.test_annotations,
                target_size=target_size,
                backup=not args.no_backup,
                verbose=verbose
            )
            total_fixed += fixed_test
            
            # Re-check after fixing
            if verbose:
                print("\nRe-checking test dataset after fixes...")
            test_issues_after = check_dataset_integrity(
                args.test_images,
                args.test_annotations,
                target_size=target_size,
                verbose=False
            )
            
            if not test_issues_after:
                if verbose:
                    print("✓ Test dataset issues resolved!")
            else:
                if verbose:
                    print(f"⚠ {len(test_issues_after)} issues remain in test dataset")
        
        if verbose:
            print(f"\nTotal files fixed: {total_fixed}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Training dataset issues: {len(train_issues)}")
        print(f"Test dataset issues: {len(test_issues)}")
        
        if args.fix:
            print(f"Total files fixed: {total_fixed}")
            if not args.no_backup:
                print("Backups created for safety")
        elif train_issues or test_issues:
            print("\nTo fix these issues, run:")
            print("python fix_dataset.py --fix")
    
    # Exit with appropriate code
    if train_issues or test_issues:
        if not args.fix:
            sys.exit(1)  # Issues found but not fixed
        else:
            # Check if issues were actually resolved
            final_train_issues = check_dataset_integrity(
                args.train_images, args.train_annotations, target_size=target_size, verbose=False
            )
            final_test_issues = check_dataset_integrity(
                args.test_images, args.test_annotations, target_size=target_size, verbose=False
            )
            if final_train_issues or final_test_issues:
                sys.exit(1)  # Issues remain after attempted fix
    
    sys.exit(0)  # All good


if __name__ == "__main__":
    main()
