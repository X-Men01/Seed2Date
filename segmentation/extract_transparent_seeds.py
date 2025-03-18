import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_seed_masks_from_yolo(base_dir, output_dir, visualize_samples=5):
    """
    Extract seed masks from images using existing YOLO segmentation labels.
    
    Args:
        base_dir: Base directory containing images and labels folders
        output_dir: Directory to save extracted seed masks
        visualize_samples: Number of random samples to visualize per category (0 to skip)
    """
    # Categories to process
    categories = ['Khalas', 'Mabroom', 'Safawi', 'Sukkari']
    
    # Track statistics
    stats = {cat: {'processed': 0, 'extracted': 0, 'failed': 0} for cat in categories}
    
    # Process each category
    for category in categories:
        print(f"\nProcessing {category} seeds...")
        
        # Setup paths
        image_dir = os.path.join(base_dir, 'images', category)
        label_dir = os.path.join(base_dir, 'labels', category)
    
        # Create output directory
        category_output_dir = os.path.join(output_dir, category)
        os.makedirs(category_output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            continue
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Extracting {category} seeds"):
            stats[category]['processed'] += 1
            
            # Get base filename and corresponding label path
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            
            # Skip if no label file exists
            if not os.path.exists(label_path):
                print(f"No label found for {img_path}")
                stats[category]['failed'] += 1
                continue
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load image: {img_path}")
                stats[category]['failed'] += 1
                continue
                
            h, w = image.shape[:2]
            
            # Create empty mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Read YOLO segmentation and draw on mask
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7:  # Need at least class_id + 3 points (x,y)
                            continue
                            
                        class_id = int(parts[0])
                        points = parts[1:]
                        
                        # Convert to numpy array of x,y points
                        points_array = np.array([float(p) for p in points]).reshape(-1, 2)
                        
                        # Denormalize coordinates (YOLO format is normalized)
                        points_array[:, 0] *= w  # x coordinates
                        points_array[:, 1] *= h  # y coordinates
                        
                        # Convert to integer points
                        points_array = points_array.astype(np.int32)
                        
                        # Draw filled polygon on mask
                        cv2.fillPoly(mask, [points_array], 255)
            except Exception as e:
                print(f"Error processing label {label_path}: {e}")
                stats[category]['failed'] += 1
                continue
            
            # If mask is empty, skip this image
            if np.max(mask) == 0:
                print(f"Empty mask for {img_path}")
                stats[category]['failed'] += 1
                continue
                
            # Create RGBA output with transparent background
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Copy RGB values from original image where mask is active
            for c in range(3):
                rgba[:, :, c] = np.where(mask > 0, image[:, :, c], 0)
                
            # Set alpha channel
            rgba[:, :, 3] = mask
            
            # Save the extracted seed
            output_path = os.path.join(category_output_dir, f"{base_name}.png")
            cv2.imwrite(output_path, rgba)
            
            stats[category]['extracted'] += 1
        
        # Print category statistics
        print(f"Category: {category}")
        print(f"  Processed: {stats[category]['processed']} images")
        print(f"  Extracted: {stats[category]['extracted']} seeds")
        print(f"  Failed: {stats[category]['failed']} images")
    
    # Visualize random samples if requested
    if visualize_samples > 0:
        visualize_extracted_seeds(output_dir, categories, samples_per_category=visualize_samples)
    
    return stats

def visualize_extracted_seeds(output_dir, categories, samples_per_category=5):
    """
    Visualize extracted seeds on different backgrounds to check transparency.
    
    Args:
        output_dir: Directory containing extracted seeds
        categories: List of categories to visualize
        samples_per_category: Number of random samples to show per category
    """
    # Create figure
    total_samples = len(categories) * samples_per_category
    fig, axes = plt.subplots(total_samples, 3, figsize=(15, 5 * total_samples))
    
    # Background colors for testing transparency
    bg_colors = [(255, 255, 255), (0, 0, 0), (0, 120, 200)]
    
    sample_idx = 0
    
    # For each category
    for category in categories:
        # Get all extracted seeds
        seed_dir = os.path.join(output_dir, category)
        if not os.path.exists(seed_dir):
            continue
            
        seed_files = glob.glob(os.path.join(seed_dir, "*.png"))
        if not seed_files:
            continue
            
        # Select random samples
        import random
        samples = random.sample(seed_files, min(samples_per_category, len(seed_files)))
        
        # Display each sample
        for seed_path in samples:
            # Load seed with alpha channel
            seed = cv2.imread(seed_path, cv2.IMREAD_UNCHANGED)
            seed = cv2.cvtColor(seed, cv2.COLOR_BGRA2RGBA)  # Convert BGR to RGB
            
            # Display on different backgrounds
            for bg_idx, bg_color in enumerate(bg_colors):
                # Create background
                bg = np.ones((seed.shape[0], seed.shape[1], 3), dtype=np.uint8)
                bg[:, :] = bg_color
                
                # Alpha blend
                alpha = seed[:, :, 3:4] / 255.0
                rgb = seed[:, :, :3]
                result = rgb * alpha + bg * (1 - alpha)
                
                # Display
                axes[sample_idx, bg_idx].imshow(result.astype(np.uint8))
                axes[sample_idx, bg_idx].set_title(f"{category} on {bg_color}")
                axes[sample_idx, bg_idx].axis('off')
            
            sample_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "extracted_seeds_samples.png"))
    plt.show()

def main():
    # Base directory containing your dataset
    base_dir = "Date_Seeds-segment-test"
    
    # Output directory for extracted seeds
    output_dir = "Date_Seeds_Synthetic/extracted_seeds"
    
    # Extract seeds
    stats = extract_seed_masks_from_yolo(base_dir, output_dir, visualize_samples=0)
    
    # Print overall statistics
    total_processed = sum(cat['processed'] for cat in stats.values())
    total_extracted = sum(cat['extracted'] for cat in stats.values())
    total_failed = sum(cat['failed'] for cat in stats.values())
    
    print("\nExtraction Complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Total seeds extracted: {total_extracted}")
    print(f"Total failures: {total_failed}")
    print(f"Extracted seeds saved to: {output_dir}")

if __name__ == "__main__":
    main()