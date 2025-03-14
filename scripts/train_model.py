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
        data='/home/cs2r/Desktop/Seed2Date-main/Date_Seeds',
        epochs=600,
        imgsz=640,
        batch=64,
        device=device,
        workers=8,
        cache=True,
        patience=30,
        verbose=True,
        lr0=0.01,
        lrf=0.01,
        optimizer='auto',
        augment=True,
        dropout=0.2,
        
        hsv_h=0.015,          
        hsv_s=0.6,            
        hsv_v=0.4,           
        degrees=15.0,        
        translate=0.15,        
        scale=0.25,           
        shear=0.0,           
        perspective=0.0,     
        flipud=0.2,           
        fliplr=0.5,          
        mosaic=0.5,          
        mixup=0.0,            
        copy_paste=0.0,       
        auto_augment="randaugment",  
        erasing=0.15,          
        crop_fraction=1.0,    
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    model.val(data='/Users/ahmedalkhulayfi/Downloads/Date_Seeds/test')
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()