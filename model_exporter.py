from ultralytics import YOLO

# Load the trained model weights from the specified directory
new_pt_path = r'D:\04 Project\Object-Tracking-System\weights\best.pt'
model = YOLO(new_pt_path)

# Export the model to TensorRT engine format
# half=True enables FP16 precision, which is well supported on RTX 3060 Ti
model.export(format='engine', device=0, half=True, imgsz=640)