import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_directory_structure(base_dir):
    """Create the necessary directory structure for the project."""
    # Create main directories for splits
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    
    return base_dir

def split_and_organize_data(raw_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split data and organize into the correct structure for YOLO classification.
    Move files instead of copying to save space.
    """
    # Validate split ratio
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    # Create class directories in each split
    for split in ["train", "val", "test"]:
        for class_name in class_dirs:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Process each class
    for class_name in class_dirs:
        print(f"Processing {class_name} images...")
        
        # Get all images in this class
        class_dir = os.path.join(raw_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split the data
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
            for img_file in tqdm(files, desc=f"{split} set"):
                # Source and destination paths
                src_path = os.path.join(class_dir, img_file)
                dst_path = os.path.join(output_dir, split, class_name, img_file)
                
                # Move the file instead of copying
                shutil.move(src_path, dst_path)
    
    # Print summary
    print("\nDataset split summary:")
    for split in ["train", "val", "test"]:
        total_images = 0
        for class_name in class_dirs:
            class_path = os.path.join(output_dir, split, class_name)
            num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_images += num_images
            print(f"  {split}/{class_name}: {num_images} images")
        print(f"\n  Total {split}: {total_images} images")
    
    # Check if raw directories are now empty
    empty_dirs = []
    for class_name in class_dirs:
        class_dir = os.path.join(raw_dir, class_name)
        remaining_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not remaining_files:
            empty_dirs.append(class_name)
    
    if empty_dirs:
        print(f"\nThe following class directories are now empty: {empty_dirs}")
    
    return output_dir

def main():
    # Base directories
    base_dir = "/Users/ahmedalkhulayfi/Downloads/Date_Seeds"
    raw_dir = os.path.join(base_dir, "raw")
    
    # Create directory structure
    output_dir = create_directory_structure(base_dir)
    
    # Split and organize data (moving files)
    dataset_path = split_and_organize_data(raw_dir, output_dir, split_ratio=(0.8, 0.1, 0.1))
    
    print("\nData organization completed successfully!")
   
if __name__ == "__main__":
    main() 