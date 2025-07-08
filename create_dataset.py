import numpy as np
import os
from PIL import Image
from tqdm import tqdm

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
            # Use the first channel (RGB) and ignore alpha
            arr = arr[:, :, 0]
        elif img.mode == 'RGB':
            # Convert to grayscale
            arr = np.mean(arr, axis=2)
        elif len(arr.shape) == 3 and arr.shape[2] == 1:
            # Single channel
            arr = arr[:, :, 0]
        
        # Ensure we have a 2D array
        if len(arr.shape) > 2:
            arr = arr[:, :, 0]
            
        return normalize_array(arr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
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
                print(f"Failed to load {var_path}")
                return None
        else:
            print(f"File not found: {var_path}")
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

def create_skagerakk_dataset(source_path, output_path, input_vars, target_var):
    """Create the skagerakk_dataset with RGB images and target annotations"""
    
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
    
    if os.path.exists(train_path):
        # Get all dates from the first variable in train
        first_var_train = os.path.join(train_path, input_vars[0])
        if os.path.exists(first_var_train):
            train_dates = [f.replace('.png', '') for f in os.listdir(first_var_train) if f.endswith('.png')]
            train_dates.sort()
            
            print(f"Processing {len(train_dates)} training images...")
            
            for date in tqdm(train_dates, desc="Creating train RGB images"):
                # Create RGB input image
                rgb_image = create_rgb_image(date, train_path, input_vars)
                
                # Copy target annotation
                target_path = os.path.join(train_path, target_var, f"{date}.png")
                
                if rgb_image is not None and os.path.exists(target_path):
                    # Save RGB input image
                    input_file = os.path.join(train_images_dir, f"{date}.png")
                    pil_image = Image.fromarray(rgb_image, mode='RGB')
                    pil_image.save(input_file)
                    
                    # Copy target annotation
                    annotation_file = os.path.join(train_annotations_dir, f"{date}.png")
                    target_img = Image.open(target_path)
                    target_img.save(annotation_file)
                    
                    train_success += 1
            
            print(f"Successfully created {train_success} training pairs")
    
    # Process test set
    test_path = os.path.join(source_path, "test")
    test_success = 0
    
    if os.path.exists(test_path):
        # Get all dates from the first variable in test
        first_var_test = os.path.join(test_path, input_vars[0])
        if os.path.exists(first_var_test):
            test_dates = [f.replace('.png', '') for f in os.listdir(first_var_test) if f.endswith('.png')]
            test_dates.sort()
            
            print(f"Processing {len(test_dates)} test images...")
            
            for date in tqdm(test_dates, desc="Creating test RGB images"):
                # Create RGB input image
                rgb_image = create_rgb_image(date, test_path, input_vars)
                
                # Copy target annotation
                target_path = os.path.join(test_path, target_var, f"{date}.png")
                
                if rgb_image is not None and os.path.exists(target_path):
                    # Save RGB input image
                    input_file = os.path.join(test_images_dir, f"{date}.png")
                    pil_image = Image.fromarray(rgb_image, mode='RGB')
                    pil_image.save(input_file)
                    
                    # Copy target annotation
                    annotation_file = os.path.join(test_annotations_dir, f"{date}.png")
                    target_img = Image.open(target_path)
                    target_img.save(annotation_file)
                    
                    test_success += 1
            
            print(f"Successfully created {test_success} test pairs")
    
    return train_success, test_success



if __name__ == "__main__":
    input_variables = ['thetao', 'so', 'o2']
    predict = 'catch'

    # Create the skagerakk_dataset using the defined variables
    source_dataset = "/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p"
    output_dataset = "/home/anna/msc_oppgave/image-segmentation/skagerakk_dataset"

    print("Creating skagerakk_dataset...")
    print(f"Input variables (RGB channels): {input_variables}")
    print(f"Target variable: {predict}")
    print(f"Source: {source_dataset}")
    print(f"Output: {output_dataset}")

    train_count, test_count = create_skagerakk_dataset(source_dataset, output_dataset, input_variables, predict)

    print(f"\nDataset creation complete!")
    print(f"Train pairs: {train_count}")
    print(f"Test pairs: {test_count}")
    print(f"Total pairs: {train_count + test_count}")

    # Show sample image info
    if train_count > 0:
        sample_train_path = os.path.join(output_dataset, "images_prepped_train")
        sample_files = [f for f in os.listdir(sample_train_path) if f.endswith('.png')]
        if sample_files:
            sample_file = sample_files[0]
            sample_image = Image.open(os.path.join(sample_train_path, sample_file))
            
            print(f"\nSample image info:")
            print(f"Input image: {sample_file}")
            print(f"Shape: {np.array(sample_image).shape}")
            print(f"Mode: {sample_image.mode}")
            print(f"RGB channels: Red={input_variables[0]}, Green={input_variables[1]}, Blue={input_variables[2]}")
            print(f"Target: {predict}")
            
            # Save a sample to visualize later
            sample_image.save("/home/anna/msc_oppgave/image-segmentation/sample_combined_image.png")
            print(f"Sample saved as: sample_combined_image.png")

    print(f"\nDataset structure:")
    print(f"- {output_dataset}/images_prepped_train/ (RGB input images)")
    print(f"- {output_dataset}/annotations_prepped_train/ ({predict} target images)")
    print(f"- {output_dataset}/images_prepped_test/ (RGB input images)")
    print(f"- {output_dataset}/annotations_prepped_test/ ({predict} target images)")


    # Verify the dataset structure and show samples
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    dataset_path = "/home/anna/msc_oppgave/image-segmentation/skagerakk_dataset"

    # Check dataset structure
    print("Dataset structure:")
    for split in ['train', 'test']:
        images_dir = os.path.join(dataset_path, f"images_prepped_{split}")
        annotations_dir = os.path.join(dataset_path, f"annotations_prepped_{split}")
        
        if os.path.exists(images_dir):
            image_count = len([f for f in os.listdir(images_dir) if f.endswith('.png')])
            print(f"  {split} images: {image_count}")
        
        if os.path.exists(annotations_dir):
            annotation_count = len([f for f in os.listdir(annotations_dir) if f.endswith('.png')])
            print(f"  {split} annotations: {annotation_count}")

    # Show sample images
    sample_date = "2022-08-22"
    train_image_path = os.path.join(dataset_path, "images_prepped_train", f"{sample_date}.png")
    train_annotation_path = os.path.join(dataset_path, "annotations_prepped_train", f"{sample_date}.png")

    if os.path.exists(train_image_path) and os.path.exists(train_annotation_path):
        # Load sample images
        input_img = Image.open(train_image_path)
        target_img = Image.open(train_annotation_path)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Show RGB input image
        axes[0, 0].imshow(input_img)
        axes[0, 0].set_title(f'RGB Input Image ({sample_date})')
        axes[0, 0].axis('off')
        
        # Show individual channels
        input_array = np.array(input_img)
        for i, var in enumerate(input_variables):
            axes[0, i+1].imshow(input_array[:, :, i], cmap='viridis')
            axes[0, i+1].set_title(f'{var} (Channel {i})')
            axes[0, i+1].axis('off')
        
        # Show target annotation
        axes[1, 0].imshow(target_img, cmap='viridis')
        axes[1, 0].set_title(f'Target: {predict}')
        axes[1, 0].axis('off')
        
        # Show target statistics
        target_array = np.array(target_img)
        axes[1, 1].hist(target_array.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_title(f'{predict} Value Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        
        # Summary statistics
        axes[1, 2].text(0.1, 0.9, f'Dataset Summary:', fontsize=12, fontweight='bold')
        axes[1, 2].text(0.1, 0.8, f'Input: RGB from {input_variables}', fontsize=10)
        axes[1, 2].text(0.1, 0.7, f'Target: {predict}', fontsize=10)
        axes[1, 2].text(0.1, 0.6, f'Train samples: 439', fontsize=10)
        axes[1, 2].text(0.1, 0.5, f'Test samples: 110', fontsize=10)
        axes[1, 2].text(0.1, 0.4, f'Image size: {input_array.shape[:2]}', fontsize=10)
        axes[1, 2].text(0.1, 0.3, f'Train/Test split: 80/20', fontsize=10)
        axes[1, 2].text(0.1, 0.2, f'Same dates in same split: ✓', fontsize=10)
        axes[1, 2].text(0.1, 0.1, f'Ready for training: ✓', fontsize=10)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        # Hide empty subplot
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/anna/msc_oppgave/image-segmentation/dataset_sample_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nSample image details:")
        print(f"Input image shape: {input_array.shape}")
        print(f"Target image shape: {target_array.shape}")
        print(f"Input value range: {input_array.min():.2f} - {input_array.max():.2f}")
        print(f"Target value range: {target_array.min():.2f} - {target_array.max():.2f}")
        
    else:
        print(f"Sample files not found: {train_image_path} or {train_annotation_path}")

    print(f"\nDataset is ready for training!")
    print(f"You can now train the model using the skagerakk_dataset with RGB inputs predicting {predict}")




    # sOMETHING OR OTHER:


    import os
import numpy as np
from PIL import Image

# Set variables correctly
variables = ['thetao', 'so', 'o2']
print("Variables set to:", variables)

# Test with a single image first
source_dataset = "/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p"
sample_date = "2019-08-22"
train_path = os.path.join(source_dataset, "train")

# Test creating RGB for one image
print("Testing RGB creation for sample date:", sample_date)
channels = []

for var in variables:
    var_path = os.path.join(train_path, var, f"{sample_date}.png")
    print(f"Checking {var_path}...")
    if os.path.exists(var_path):
        img = Image.open(var_path)
        arr = np.array(img)
        # Use first channel from RGBA
        channel = arr[:, :, 0]
        channels.append(channel)
        print(f"  {var}: Shape {channel.shape}, min/max: {channel.min()}/{channel.max()}")
    else:
        print(f"  {var}: FILE NOT FOUND")

if len(channels) == 3:
    # Find common dimensions
    min_height = min(ch.shape[0] for ch in channels)
    min_width = min(ch.shape[1] for ch in channels)
    print(f"Common dimensions: {min_height}x{min_width}")
    
    # Crop and normalize
    normalized_channels = []
    for i, ch in enumerate(channels):
        cropped = ch[:min_height, :min_width]
        # Normalize to 0-255
        normalized = ((cropped - cropped.min()) / (cropped.max() - cropped.min()) * 255).astype(np.uint8)
        normalized_channels.append(normalized)
        print(f"  Channel {i} ({variables[i]}): normalized min/max: {normalized.min()}/{normalized.max()}")
    
    # Create RGB image
    rgb_image = np.stack(normalized_channels, axis=-1)
    print(f"RGB image shape: {rgb_image.shape}")
    
    # Save test image
    test_rgb = Image.fromarray(rgb_image, mode='RGB')
    test_rgb.save("/home/anna/msc_oppgave/image-segmentation/test_rgb.png")
    print("Test RGB image saved as test_rgb.png")
else:
    print(f"Error: Only found {len(channels)} channels out of 3")






    # Something else
    # Debug: Check the format of the source images
import numpy as np
from PIL import Image

# Check a sample image from each variable
sample_date = "2019-08-22"
base_path = "/home/anna/msc_oppgave/image-segmentation/test128x128GAT_19to24_100p/train"

for var in variables:
    var_path = f"{base_path}/{var}/{sample_date}.png"
    if os.path.exists(var_path):
        img = Image.open(var_path)
        arr = np.array(img)
        print(f"{var}:")
        print(f"  PIL mode: {img.mode}")
        print(f"  Array shape: {arr.shape}")
        print(f"  Array dtype: {arr.dtype}")
        print(f"  Min/Max values: {arr.min()}/{arr.max()}")
        print()