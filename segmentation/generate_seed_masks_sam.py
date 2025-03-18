import torch
from ultralytics import SAM
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from skimage import measure
import matplotlib.pyplot as plt


def segment_seed_by_size(img_path, output_path=None, visualization_path=None, class_id=0):
    """Segment a date seed by taking the smallest mask from SAM's output"""
    # Load the SAM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SAM('sam_b.pt' )
    
    # Run segmentation with "everything" mode (no prompts)
    results = model(img_path, verbose=False,device="mps")
    
    # Get all masks
    masks = results[0].masks.data.cpu().numpy()
    
    if len(masks) == 0:
        print(f"No masks found for {img_path}")
        return None
    
    # Calculate area for each mask
    mask_areas = [(i, np.sum(mask > 0.5)) for i, mask in enumerate(masks)]
    
  
    max_area = 17.e6 # Size of the image in pixels (5184×3456 pixels)
    min_area = 90e3 # Minimum area of a seed in pixels (which differs between varieties)
                    # This is based on the morphometric analysis of the dataset and experimentation
                    # which can be found in the README.md file
                    # Khalas=50e3, Mabroom=90e3, Safawi=111e3, Sukkari=90e3
                 
    
    # Filter out masks that are too small or too large
    mask_areas = [ma for ma in mask_areas if max_area > ma[1] >= min_area]
    
    if len(mask_areas) == 0:
        print(f"No masks found for {img_path}")
        return None
    
    # Sort by area (smallest first)
    mask_areas.sort(key=lambda x: x[1])
    
    # Get the smallest mask (the seed)
    seed_idx = mask_areas[0][0]
    seed_mask = masks[seed_idx]
    
    # Load original image for visualization
    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    # Save segmentation mask in YOLO format if requested
    if output_path:
        contours = measure.find_contours(seed_mask, 0.5)
        with open(output_path, 'w') as f:
            for contour in contours:
                # Skip very small contours
                if len(contour) < 5:
                    continue
                    
                # Normalize coordinates
                contour_normalized = np.copy(contour)
                contour_normalized[:, 0] /= h  # y coordinates
                contour_normalized[:, 1] /= w  # x coordinates
                
                # Format for YOLO
                segment_str = []
                for point in contour_normalized:
                    segment_str.extend([str(point[1]), str(point[0])])  # x,y format
                    
                f.write(f"{class_id} " + " ".join(segment_str) + "\n")
    
    # Save visualization if requested
    if visualization_path:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Create colored mask overlay
        color_mask = np.zeros((h, w, 4), dtype=np.uint8)
        color_mask[seed_mask > 0.5] = [255, 0, 0, 128]  # Semi-transparent red
        
        plt.imshow(color_mask)
        
        # Add contour
        plt.contour(seed_mask, colors=['yellow'], levels=[0.5], linewidths=2)
        
        plt.title(f"Seed Area: {mask_areas[0][1]} pixels")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(visualization_path)
        plt.close()
    
    return results

def process_folder(input_dir, output_dir, vis_dir=None, class_id=0):
    """Process all images in a folder"""
    os.makedirs(output_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        out_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.txt')
        
        vis_path = None
        if vis_dir:
            vis_path = os.path.join(vis_dir, os.path.splitext(img_file)[0] + '.png')
        
        segment_seed_by_size(img_path, out_path, vis_path, class_id)

# Example usage
# img_path = "/Users/ahmedalkhulayfi/Downloads/Date_Seeds/test/Khalas/IMG_3490.JPG"
# output_txt = "seed_segmentation.txt"
# vis_path = "seed_visualization.png"

# segment_seed_by_size(img_path, output_txt, vis_path)

# To process entire folder
process_folder(
    input_dir="/Date_Seeds/train/Safawi",
    output_dir="/Date_Seeds/train/Safawi_segmentation",
    vis_dir=None,
    class_id=3  # Set your desired class ID
)

# Class ID 0 for Khalas
# Class ID 1 for Mabroom
# Class ID 2 for Safawi
# Class ID 3 for Sukkari