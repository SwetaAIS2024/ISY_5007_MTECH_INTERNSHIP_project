from ultralytics import YOLO

# Path to the trained YOLOv8 model
trained_model_path = "/home/rubesh/Desktop/sweta/Mtech_internship/Main_prj/Model_Conversion_Optimization/yolov8n-seg.pt"  # Path to the best model after training

# Load the trained YOLOv8 model
model = YOLO(trained_model_path)

# Export the model to ONNX format
model.export(format="onnx", dynamic=True)  # Export to ONNX format with dynamic input shapes

print("Model exported to ONNX format successfully!")