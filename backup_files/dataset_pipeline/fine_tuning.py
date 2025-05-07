from ultralytics import YOLO

# Path to the dataset configuration file
finetuning_dataset = "/home/rubesh/Desktop/sweta/Mtech_internship/backup_repo/dataset_pipeline/fine_tuning_dataset/dataset.yaml"

# Load a YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Load the YOLOv8n segmentation model

# Train the model
model.train(
    data=finetuning_dataset,  # Path to the dataset.yaml file
    epochs=100,                # Number of training epochs
    imgsz=640,                # Image size (YOLO will resize all images to 640x640)
    batch=16,                 # Batch size
    name="yolov8n-seg-fine-tuned",  # Name of the training run
    project="runs/train"       # Directory to save training results
)

# Export the trained model to ONNX format
trained_model_path = "runs/train/yolov8n-seg-fine-tuned/weights/best.pt"  # Path to the best model after training
# model = YOLO(trained_model_path)  # Load the trained model
# model.export(format="onnx", dynamic=True)  # Export to ONNX format with dynamic input shapes

# print("Model exported to ONNX format successfully!")

# Run inference on the test folder
test_images_path = "/home/rubesh/Desktop/sweta/Mtech_internship/backup_repo/dataset_pipeline/all_images_LTA"  # Path to the test images folder
results = model.predict(
    source=test_images_path,  # Path to the test images folder
    save=True,                # Save the predictions
    save_txt=True,            # Save predictions in YOLO format
    imgsz=640,
    project="runs/test_unknown_predict/yolov8n-seg-fine-tuned/" # Image size for inference
)

print("Inference completed. Results saved in the 'runs/test_unknown_predict' directory.")