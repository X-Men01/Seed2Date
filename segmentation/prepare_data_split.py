import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_directory_structure(base_dir):
    """Create the necessary directory structure for the project."""
    # Create main directories for splits with images and labels subdirectories
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            split_subdir = os.path.join(base_dir, split, subdir)
            os.makedirs(split_subdir, exist_ok=True)
    
    return base_dir

def split_and_organize_data(synthetic_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split data and organize into train/val/test directories, keeping images and labels paired.
    Move files instead of copying to save space.
    """
    # Validate split ratio
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"
    
    # Get paths
    images_dir = os.path.join(synthetic_dir, "images")
    labels_dir = os.path.join(synthetic_dir, "labels")
    
    # Verify both directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError("Images or labels directory not found")
    
    # Get all image files (assuming common image extensions)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images")
    
    # Split the image files
    train_files, test_val_files = train_test_split(
        image_files, 
        train_size=split_ratio[0], 
        random_state=42
    )
    
    # Further split test_val into val and test
    val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    val_files, test_files = train_test_split(
        test_val_files, 
        train_size=val_ratio, 
        random_state=42
    )
    
    # Map files to their respective splits
    split_files = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    # Move files to appropriate directories
    for split, files in split_files.items():
        print(f"Processing {split} split...")
        for img_file in tqdm(files, desc=f"{split} set"):
            # Get base filename without extension to find corresponding label
            file_base = os.path.splitext(img_file)[0]
            
            # Handle image file
            img_src_path = os.path.join(images_dir, img_file)
            img_dst_path = os.path.join(output_dir, split, "images", img_file)
            shutil.move(img_src_path, img_dst_path)
            
            # Look for corresponding label file (try common label extensions)
            for ext in ['.txt', '.xml', '.json']:
                label_file = file_base + ext
                label_src_path = os.path.join(labels_dir, label_file)
                if os.path.exists(label_src_path):
                    label_dst_path = os.path.join(output_dir, split, "labels", label_file)
                    shutil.move(label_src_path, label_dst_path)
                    break
    
    # Print summary
    print("\nDataset split summary:")
    for split in ["train", "val", "test"]:
        num_images = len(os.listdir(os.path.join(output_dir, split, "images")))
        num_labels = len(os.listdir(os.path.join(output_dir, split, "labels")))
        print(f"  {split}: {num_images} images, {num_labels} labels")
    
    return output_dir

def main():
    # Base directory
    base_dir = "/Users/ahmedalkhulayfi/Downloads/synthetic_split"
    synthetic_dir ="/Users/ahmedalkhulayfi/Desktop/Seed2Date/segmentation/synthetic"
    
    # Create directory structure
    output_dir = create_directory_structure(base_dir)
    
    # Split and organize data (moving files)
    dataset_path = split_and_organize_data(synthetic_dir, output_dir, split_ratio=(0.8, 0.1, 0.1))
    
    print("\nData organization completed successfully!")
   
if __name__ == "__main__":
    main()