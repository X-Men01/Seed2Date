from ultralytics import YOLO
import torch

# Configuration - Edit these values directly
CONFIG = {
    'data': '/Users/ahmedalkhulayfi/Downloads/Date_Seeds',                  # Path to dataset
    'epochs': 600,                   # Number of epochs
    'batch': 64,                     # Batch size
    'imgsz': 640,                    # Image size
    'device': '0,1',                 # CUDA devices (use both GPUs)
    'workers': 8,                    # Number of worker threads
    "cache": True,
    'pretrained': 'yolo11n-cls.pt',  # Pretrained model
    'patience': 20,                  # Early stopping patience
    'lr0': 1e-3,                     # Initial learning rate
    'lrf': 1e-4,                     # Final learning rate
    'optimizer': 'AdamW',             # Optimizer
    'augment': True,                 # Use built-in augmentations
    'dropout': 0.2,                  # Dropout rate for regularization
}

def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
    else:
        print("CUDA is not available. Training will use CPU.")
        CONFIG['device'] = 'cpu'
    
    # Load a pretrained model
    print(f"Loading pretrained model: {CONFIG['pretrained']}")
    model = YOLO(CONFIG['pretrained'])
    
    # Train the model
    print(f"Starting training on {CONFIG['data']} for {CONFIG['epochs']} epochs")
    model.train(
        data=CONFIG['data'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        device=CONFIG['device'],
        workers=CONFIG['workers'],
        name=CONFIG['name'],
        patience=CONFIG['patience'],
        verbose=True,
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        optimizer=CONFIG['optimizer'],
        augment=CONFIG['augment'],
        dropout=CONFIG['dropout'],
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    model.val(data=f"{CONFIG['data']}/test")
    
    print("\nTraining and evaluation completed!")
    print(f"Results saved to: runs/classify/{CONFIG['name']}")

if __name__ == "__main__":
    main() 