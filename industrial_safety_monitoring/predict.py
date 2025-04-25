from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.predict(source="data/main.mp4", save=True, conf=0.25)
