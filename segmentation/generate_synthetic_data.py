import os
import cv2
import numpy as np
import random
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

def load_seeds(extracted_seeds_dir):
    """Load seed metadata without loading images into memory"""
    seeds = {}
    categories = ['Khalas', 'Mabroom', 'Safawi', 'Sukkari']
    category_ids = {cat: idx for idx, cat in enumerate(categories)}
    
    for category in categories:
        category_dir = os.path.join(extracted_seeds_dir, 'images', category)
        labels_dir = os.path.join(extracted_seeds_dir, 'labels', category)
        
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory not found: {category_dir}")
            continue
            
        seeds[category] = {
            'id': category_ids[category],
            'images': []
        }
        
        seed_files = glob.glob(os.path.join(category_dir, "*.png"))
        print(f"Found {len(seed_files)} {category} seed images")
        
        for seed_file in seed_files:
            # Don't load the image yet, just store the path
            base_name = os.path.splitext(os.path.basename(seed_file))[0]
            label_file = os.path.join(labels_dir, f"{base_name}.txt")
            
            # Check if label file exists
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label_content = f.read().strip()
                    
                seeds[category]['images'].append({
                    'path': seed_file,
                    'label': label_content
                })
            else:
                print(f"Warning: No label file found for {seed_file}")
    
    return seeds

def load_backgrounds(backgrounds_dir):
    """Load background image paths only (not the actual images)"""
    backgrounds = []
    
    for filename in os.listdir(backgrounds_dir):
        filepath = os.path.join(backgrounds_dir, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                backgrounds.append(filepath)
    
    print(f"Found {len(backgrounds)} background images")
    return backgrounds

def place_seed_with_labels(background_img, seed_img, seed_label, class_id):
    """Place a seed on a background with transformed YOLO segmentation labels"""
    # Parse YOLO segmentation format labels
    polygons = []
    
    for line in seed_label.strip().split('\n'):
        if not line:
            continue
        parts = line.split()
        # We'll use the new class_id instead of the original
        points = []
        for i in range(1, len(parts), 2):
            if i+1 < len(parts):
                x, y = float(parts[i]), float(parts[i+1])
                # Convert from normalized to absolute coordinates
                x_px = x * seed_img.shape[1]
                y_px = y * seed_img.shape[0]
                points.append((x_px, y_px))
        
        if len(points) >= 3:  # Need at least a triangle
            polygon = Polygon(points, label=class_id)
            polygons.append(polygon)
    
    if not polygons:
        print(f"Warning: No valid polygons found in the label")
        return background_img, []
    
    polygons_on_image = PolygonsOnImage(polygons, shape=seed_img.shape)
    
    # Define placement parameters
    bg_h, bg_w = background_img.shape[:2]
    seed_h, seed_w = seed_img.shape[:2]
    
    # Random scale factor (0.30 to 0.70)
    scale = random.uniform(0.30, 0.70)
    
    # Ensure the seed fits within the background
    if seed_h * scale >= bg_h or seed_w * scale >= bg_w:
        scale = min(bg_h / seed_h, bg_w / seed_w) * 0.8
    
    # Random rotation
    angle = random.uniform(0, 360)
    
    # Create augmenter for seed
    seq = iaa.Sequential([
        iaa.Resize(scale),
        iaa.Rotate(angle),
    ])
    
    # Apply to seed and polygons
    seed_aug, polygons_aug = seq(image=seed_img, polygons=polygons_on_image)
    
    # Calculate placement position
    seed_h_new, seed_w_new = seed_aug.shape[:2]
    x_pos = random.randint(0, max(1, bg_w - seed_w_new))
    y_pos = random.randint(0, max(1, bg_h - seed_h_new))
    
    # Composite images
    result = background_img.copy()
    
    # Extract alpha channel
    if seed_aug.shape[2] == 4:  # Has alpha channel
        alpha = seed_aug[:, :, 3:4] / 255.0
        
        # Define ROI
        roi = result[y_pos:y_pos+seed_h_new, x_pos:x_pos+seed_w_new].copy()
        
        # Blend
        for c in range(3):
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                roi[:, :, c] = roi[:, :, c] * (1 - alpha[:, :, 0]) + seed_aug[:, :, c] * alpha[:, :, 0]
        
        # Place back
        result[y_pos:y_pos+seed_h_new, x_pos:x_pos+seed_w_new] = roi
    
    # Transform polygons to final YOLO format
    transformed_polygons = []
    for polygon in polygons_aug.polygons:
        # Get polygon points
        points = polygon.exterior
        
        # Shift polygon by placement position
        shifted_points = [(p[0] + x_pos, p[1] + y_pos) for p in points]
        
        # Normalize coordinates to 0-1 range for YOLO format
        normalized_points = [(x/bg_w, y/bg_h) for x, y in shifted_points]
        
        # Skip invalid polygons (outside image or too small)
        if len(normalized_points) >= 3:
            transformed_polygons.append({
                'class_id': class_id,
                'points': normalized_points
            })
    
    return result, transformed_polygons

def generate_synthetic_dataset(
    extracted_seeds_dir,
    backgrounds_dir,
    output_dir,
    num_images=1000,
    seeds_per_image=(1, 3),
    visualize_samples=5,
    batch_size=50  # Process in smaller batches
):
    """Generate a synthetic dataset of seeds on backgrounds"""
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load metadata (not actual images)
    backgrounds = load_backgrounds(backgrounds_dir)
    if not backgrounds:
        print("Error: No background images found!")
        return 0
    
    seeds = load_seeds(extracted_seeds_dir)
    categories = list(seeds.keys())
    if not categories:
        print("Error: No seed images found!")
        return 0
    
    # Process in batches
    image_count = 0
    total_batches = (num_images + batch_size - 1) // batch_size
    
    for batch in range(total_batches):
        print(f"Processing batch {batch+1}/{total_batches}")
        batch_start = batch * batch_size
        batch_end = min(num_images, batch_start + batch_size)
        batch_size_actual = batch_end - batch_start
        
        if batch_size_actual <= 0:
            break
            
        # Determine backgrounds for this batch
        random.shuffle(backgrounds)
        batch_backgrounds = backgrounds[:batch_size_actual]
        
        # Process backgrounds for this batch
        for i, bg_path in enumerate(tqdm(batch_backgrounds, desc=f"Batch {batch+1}")):
            # Load background on demand
            bg_image = cv2.imread(bg_path)
            if bg_image is None:
                print(f"Warning: Could not load background image: {bg_path}")
                continue
            
            # Number of seeds to place
            num_seeds = random.randint(seeds_per_image[0], seeds_per_image[1])
            
            # Copy background for this image
            synthetic_image = bg_image.copy()
            
            # Create label file
            image_idx = batch_start + i
            label_path = os.path.join(labels_dir, f"synthetic_{image_idx:06d}.txt")
            
            with open(label_path, 'w') as label_file:
                # Place seeds
                for _ in range(num_seeds):
                    # Select random seed category and image
                    category = random.choice(categories)
                    if not seeds[category]['images']:
                        continue
                        
                    seed_data = random.choice(seeds[category]['images'])
                    # Load seed image on demand
                    seed_image = cv2.imread(seed_data['path'], cv2.IMREAD_UNCHANGED)
                    
                    if seed_image is None or seed_image.shape[2] != 4:
                        continue
                        
                    seed_label = seed_data['label']
                    class_id = seeds[category]['id']
                    
                    # Place seed on background with transformed labels
                    synthetic_image, yolo_contours = place_seed_with_labels(
                        synthetic_image, seed_image, seed_label, class_id
                    )
                    
                    # Free memory
                    del seed_image
                    
                    # Write YOLO segmentation format labels
                    for contour in yolo_contours:
                        points_flat = []
                        for x, y in contour['points']:
                            points_flat.extend([str(round(x, 6)), str(round(y, 6))])
                        
                        label_file.write(f"{contour['class_id']} {' '.join(points_flat)}\n")
            
            # Save synthetic image
            image_path = os.path.join(images_dir, f"synthetic_{image_idx:06d}.jpg")
            cv2.imwrite(image_path, synthetic_image)
            
            # Only save the first few images for visualization (if needed)
            if visualize_samples > 0 and image_idx < visualize_samples:
                vis_dir = os.path.join(output_dir, 'visualization')
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"sample_{image_idx}.jpg")
                cv2.imwrite(vis_path, synthetic_image)
            
            # Free memory
            del synthetic_image
            del bg_image
            
            image_count += 1
        
        # Force garbage collection between batches
        import gc
        gc.collect()
    
    print(f"Generated {image_count} synthetic images")
    
    # Visualize at the end (optional)
    if visualize_samples > 0:
        try:
            vis_dir = os.path.join(output_dir, 'visualization')
            if os.path.exists(vis_dir):
                sample_images = []
                for i in range(min(visualize_samples, image_count)):
                    sample_path = os.path.join(vis_dir, f"sample_{i}.jpg")
                    if os.path.exists(sample_path):
                        img = cv2.imread(sample_path)
                        if img is not None:
                            sample_images.append(img)
                
                if sample_images:
                    plt.figure(figsize=(15, 3 * len(sample_images)))
                    for i, img in enumerate(sample_images):
                        plt.subplot(len(sample_images), 1, i+1)
                        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        plt.title(f"Synthetic Image {i}")
                        plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "synthetic_samples.png"))
                    plt.close()  # Close the figure to free memory
        except Exception as e:
            print(f"Visualization error: {e}")
    
    return image_count

def main():
    # Configure paths
    extracted_seeds_dir = "/Users/ahmedalkhulayfi/Desktop/Seed2Date/segmentation/extracted_seeds_transparent_background"
    backgrounds_dir = "/Users/ahmedalkhulayfi/Desktop/Seed2Date/segmentation/backgrounds"
    output_dir = "synthetic"
    
    # Generate synthetic dataset
    generate_synthetic_dataset(
        extracted_seeds_dir=extracted_seeds_dir,
        backgrounds_dir=backgrounds_dir,
        output_dir=output_dir,
        num_images=2000,  # Target number of synthetic images
        seeds_per_image=(1, 3),  # Random number of seeds per image
        visualize_samples=5,  # Show sample results
        batch_size=50  # Process 50 images at a time
    )

if __name__ == "__main__":
    main() 