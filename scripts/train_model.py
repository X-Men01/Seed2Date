from ultralytics import YOLO
import torch

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
        device = '0,1'
    else:
        print("CUDA is not available. Training will use CPU.")
        device = 'cpu'
    
    # Load a pretrained model
    pretrained_model = 'yolo11n-cls.pt'
    print(f"Loading pretrained model: {pretrained_model}")
    model = YOLO(pretrained_model)
    
    # Train the model with optimized augmentation parameters
    print("Starting training with seed-optimized augmentation parameters")
    model.train(
        # Basic training parameters
        data='/Users/ahmedalkhulayfi/Downloads/Date_Seeds',
        epochs=600,
        imgsz=640,
        batch=64,
        device=device,
        workers=8,
        cache=True,
        patience=20,
        verbose=True,
        lr0=1e-3,
        lrf=1e-4,
        optimizer='AdamW',
        augment=True,
        dropout=0.2,
        
        hsv_h=0.015,          # Minimal hue variation
        hsv_s=0.4,            # Reduced saturation variation
        hsv_v=0.4,            # Moderate brightness variation
        degrees=15.0,         # Moderate rotation
        translate=0.1,        # Small translation
        scale=0.1,            # Reduced scale variation
        shear=0.0,            # No shear
        perspective=0.0,      # No perspective transform
        flipud=0.2,           # Some vertical flips
        fliplr=0.5,           # Horizontal flips
        mosaic=0.0,           # Disable mosaic
        mixup=0.0,            # No mixup
        copy_paste=0.0,       # No copy-paste
        auto_augment=None,  # Disable auto augmentation
        erasing=0.0,          # No random erasing
        crop_fraction=1.0,    # Full crop
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    model.val(data='/Users/ahmedalkhulayfi/Downloads/Date_Seeds/test')
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()