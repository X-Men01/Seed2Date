def analyze_entire_dataset(dataset_path, output_dir='results/morphometrics'):
    """
    Analyze all seeds in the dataset and collect comprehensive measurements
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Directory to save results
    """
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get classes
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Prepare dataframe for results
    all_results = []
    
    # Process each class
    for class_name in classes:
        print(f"\nProcessing all seeds in class: {class_name}")
        class_path = os.path.join(dataset_path, class_name)
        
        # Get all images
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        
        # Process each image
        for img_name in tqdm(images):
            img_path = os.path.join(class_path, img_name)
            
            try:
                # Load and process image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Calculate seed dimensions
                nonzero = cv2.findNonZero(edges)
                if nonzero is not None:
                    x, y, w, h = cv2.boundingRect(nonzero)
                    aspect_ratio = w/h
                    
                    # Store results
                    all_results.append({
                        'class': class_name,
                        'image': img_name,
                        'width': w,
                        'height': h,
                        'aspect_ratio': aspect_ratio,
                        'avg_r': np.mean(img_rgb[:,:,0]),
                        'avg_g': np.mean(img_rgb[:,:,1]),
                        'avg_b': np.mean(img_rgb[:,:,2]),
                    })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Convert to dataframe
    df = pd.DataFrame(all_results)
    
    # Save raw data
    df.to_csv(os.path.join(output_dir, 'seed_measurements.csv'), index=False)
    
    # Statistical analysis
    stats = df.groupby('class').agg({
        'aspect_ratio': ['mean', 'median', 'std', 'min', 'max'],
        'width': ['mean', 'std'],
        'height': ['mean', 'std'],
        'avg_r': 'mean',
        'avg_g': 'mean',
        'avg_b': 'mean'
    })
    
    # Save statistics
    stats.to_csv(os.path.join(output_dir, 'seed_stats.csv'))
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    df.boxplot(column='aspect_ratio', by='class')
    plt.title('Aspect Ratio Distribution by Date Seed Variety')
    plt.suptitle('')  # Remove pandas default suptitle
    plt.ylabel('Aspect Ratio (width/height)')
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_boxplot.png'))
    
    # Create histogram for each class
    plt.figure(figsize=(12, 8))
    for i, cls in enumerate(classes, 1):
        plt.subplot(2, 2, i)
        df[df['class'] == cls]['aspect_ratio'].hist(bins=20)
        plt.title(f'{cls} Aspect Ratio Distribution')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aspect_ratio_histograms.png'))
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return df, stats