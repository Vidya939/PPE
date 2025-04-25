from ultralytics import YOLO

# Load the best-performing model
model = YOLO("runs/detect/train/weights/best.pt")

# OR load the last model (e.g., to resume training)
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)
