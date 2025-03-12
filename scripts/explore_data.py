import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import pandas as pd
import seaborn as sns

def explore_dataset(data_dir):
    """Explore the date seed dataset."""
    # Get class names
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(classes)} classes: {classes}")
    
    # Count images per class
    class_counts = {}
    img_paths = {}
    img_sizes = []
    aspect_ratios = []
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[cls] = len(imgs)
        img_paths[cls] = [os.path.join(cls_dir, img) for img in imgs]
        
        # Sample a few images to analyze sizes and aspect ratios
        for img_path in random.sample(img_paths[cls], min(10, len(img_paths[cls]))):
            with Image.open(img_path) as img:
                width, height = img.size
                img_sizes.append((width, height))
                aspect_ratios.append(width / height)
    
    print("\nImage count per class:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count} images")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Number of Images per Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/class_distribution.png')
    plt.close()
    
    # Visualize sample images
    visualize_samples(img_paths, classes)
    
    # Analyze image sizes
    analyze_image_properties(img_sizes, aspect_ratios)
    
    return img_paths, classes

def visualize_samples(img_paths, classes, samples_per_class=3):
    """Visualize sample images from each class."""
    plt.figure(figsize=(15, 4 * len(classes)))
    
    for i, cls in enumerate(classes):
        for j in range(samples_per_class):
            if len(img_paths[cls]) > j:
                img_path = random.choice(img_paths[cls])
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(len(classes), samples_per_class, i * samples_per_class + j + 1)
                plt.imshow(img)
                plt.title(f'{cls} - Sample {j+1}')
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/sample_images.png')
    plt.close()

def analyze_image_properties(img_sizes, aspect_ratios):
    """Analyze image sizes and aspect ratios."""
    widths, heights = zip(*img_sizes)
    
    # Image size statistics
    print("\nImage size statistics:")
    print(f"Average dimensions: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
    print(f"Min dimensions: {min(widths)} x {min(heights)}")
    print(f"Max dimensions: {max(widths)} x {max(heights)}")
    
    # Aspect ratio statistics
    print("\nAspect ratio statistics:")
    print(f"Average aspect ratio: {np.mean(aspect_ratios):.3f}")
    print(f"Min aspect ratio: {min(aspect_ratios):.3f}")
    print(f"Max aspect ratio: {max(aspect_ratios):.3f}")
    
    # Plot size distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Image Dimensions')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    
    plt.subplot(1, 2, 2)
    plt.hist(aspect_ratios, bins=20, alpha=0.7)
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('results/image_properties.png')
    plt.close()

def analyze_color_distribution(img_paths, classes, samples_per_class=10):
    """Analyze color distribution in the images."""
    color_means = {cls: [] for cls in classes}
    
    for cls in classes:
        for img_path in random.sample(img_paths[cls], min(samples_per_class, len(img_paths[cls]))):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mean_color = np.mean(img, axis=(0, 1))
            color_means[cls].append(mean_color)
    
    # Plot color distribution
    plt.figure(figsize=(15, 10))
    
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(3, 1, i+1)
        data = []
        labels = []
        
        for cls in classes:
            channel_values = [color[i] for color in color_means[cls]]
            data.extend(channel_values)
            labels.extend([cls] * len(channel_values))
        
        df = pd.DataFrame({'Class': labels, f'{channel} Channel': data})
        sns.boxplot(x='Class', y=f'{channel} Channel', data=df)
        plt.title(f'{channel} Channel Distribution by Class')
    
    plt.tight_layout()
    plt.savefig('results/color_distribution.png')
    plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Path to your dataset
    data_dir = "/Users/ahmedalkhulayfi/Downloads/Date_Seeds"  # Adjust the path as needed
    
    # Explore dataset
    img_paths, classes = explore_dataset(data_dir)
    
    # Analyze color distribution
    analyze_color_distribution(img_paths, classes)
    
    print("\nData exploration completed. Check the results directory for visualizations.") 