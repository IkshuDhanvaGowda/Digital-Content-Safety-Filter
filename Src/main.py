from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Train the model on the master_yolo_data dataset for 50 epochs
results = model.train(data="config.yaml", epochs=50, imgsz=640)