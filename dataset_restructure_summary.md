# Dataset Restructuring Summary

## Overview
Successfully restructured the dataset in `/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p/` from a flat structure to a proper train/test split, and created a new `skagerakk_dataset` combining three oceanographic variables (thetao, so, o2) into RGB images to predict catch.

## Changes Made

### Original Structure
```
test128x128GAT_19to24_100p/
├── catch/
├── chl/3/
├── kd/3/
├── no3/3/
├── nppv/3/
├── o2/3/
├── phyc/3/
├── po4/3/
├── si/3/
├── so/0_49/
├── thetao/0_49/
├── uo/0_49/
├── vo/0_49/
└── zooc/3/
```

### New Structure
```
skagerakk_dataset/
├── images_prepped_train/
│   └── YYYY-MM-DD.png (RGB images combining three variables)
├── annotations_prepped_train/
│   └── YYYY-MM-DD.png (catch values as annotations)
├── images_prepped_test/
│   └── YYYY-MM-DD.png (RGB images combining three variables)
└── annotations_prepped_test/
    └── YYYY-MM-DD.png (catch values as annotations)
```

## Key Features

### 1. Date-Based Splitting
- **549 unique dates** identified from 2019-08-22 to 2024-11-22
- **Train set**: 439 dates (80.0%)
- **Test set**: 110 dates (20.0%)
- **No date overlap** between train and test sets (verified)

### 2. Variable Combination for RGB Images
- **Input variables**: three variables defined through argparse
- **Target variable**: catch (fish catch data)
- **RGB mapping**: 
  - Red channel: var1
  - Green channel: var2
  - Blue channel: var3
- **Normalization**: Individual images are already scaled

### 3. Filename Standardization
- **Original catch files**: `catch_YYYY-MM-DD.png`
- **Other files**: `YYYY-MM-DD.png`
- **Standardized all files** to: `YYYY-MM-DD.png`

### 4. resize images to 256x256


### 3. File Distribution
- **Training files**: 6,146 files (80.0%)
- **Testing files**: 1,540 files (20.0%)
- **Total files**: 7,686 files
- **14 variables** with equal representation in both splits

## Benefits

1. **Proper Train/Test Split**: 80/20 ratio maintained
2. **Temporal Integrity**: All images from the same date stay in the same split
3. **Consistent Naming**: All files follow the same naming convention
4. **Balanced Variables**: Each variable has the same number of files in both splits
5. **Reproducible**: Random seed (42) ensures consistent splits

## Technical Details

- **Script**: `restructure_dataset.py`
- **Verification**: `verify_split.py`
- **Random Seed**: 42 (for reproducibility)
- **Date Range**: 2019-08-22 to 2024-11-22
- **Variables**: catch, chl, kd, no3, nppv, o2, phyc, po4, si, so, thetao, uo, vo, zooc

## Validation
✅ No overlapping dates between train and test  
✅ Correct 80/20 split ratio  
✅ All variables represented in both splits  
✅ Consistent filename format  
✅ File integrity maintained (copy, not move)  
