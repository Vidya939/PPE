from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/last.pt")

# Run inference on an image/frame
results = model.predict(source="output/frames/frame_00100.jpg", conf=0.25)

# Optional: show results
results[0].show()
