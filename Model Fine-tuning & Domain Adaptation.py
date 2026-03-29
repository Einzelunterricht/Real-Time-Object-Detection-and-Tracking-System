import torch
from ultralytics import YOLO
from pathlib import Path


def execute_training_pipeline():
    """
    Launches the training process with domain-specific hyperparameters.
    Ensures hue stability and geometric consistency for UI elements.
    """

    # Check for CUDA acceleration (NVIDIA GPU required for TensorRT targets)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Hardware Acceleration Detected: {device_name}")
    else:
        print("❌ Critical: GPU acceleration not found. Training on CPU is not recommended.")
        return

    # Project Root Resolution (Mapping to D:\04 Project\Object-Tracking-System)
    BASE_DIR = Path(__file__).parent.resolve()

    # Initialize SOTA Detection Backbone
    #
    model = YOLO(str(BASE_DIR / 'yolo26n.pt'))

    # Start Fine-tuning Process
    model.train(
        # --- Core Configurations ---
        data=str(BASE_DIR / 'hero.yaml'),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        optimizer='SGD',
        patience=20,  # Early stopping to prevent overfitting
        lr0=0.0001,  # Conservative learning rate to preserve pre-trained features

        # --- Domain-Specific Augmentation Strategy ---
        # Crucial for maintaining color-based feature extraction (Health Bars)
        augment=True,
        hsv_h=0.0,  # Hue Constancy: Prevents color shifting of UI elements
        hsv_s=0.3,  # Saturation Jitter: Simulates lighting/transparency variations
        hsv_v=0.3,  # Value/Brightness Jitter

        # --- Geometric Constraints ---
        degrees=0.0,  # UI elements are orientation-fixed; rotation is noise
        flipud=0.0,  # Vertical flip is invalid for HP bar logic
        fliplr=0.5,  # Horizontal flip: Simulates targets moving in opposite directions

        # --- Contextual Complexity ---
        mosaic=1.0  # Enhances detection robustness in cluttered environments (Teamfights)
        #
    )


if __name__ == '__main__':
    # Execute protected training loop
    execute_training_pipeline()