import cv2
import csv
from datetime import timedelta
from ultralytics import YOLO

def box_center(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# Load YOLO models
person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("runs/detect/train/weights/best.pt")

# Class indices in best.pt
CUSTOM_CLASSES = {
    'helmet': 0,
    'jacket': 1
}

# Video
video_path = "data/main.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

# Logging
csv_file = open("ppe_violations.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame Number", "Timestamp", "Violation Type"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 6 == 0:  # Sample at ~5 FPS
        timestamp = str(timedelta(seconds=int(frame_count / fps)))

        # Detect persons
        person_results = person_model.predict(frame, conf=0.5, verbose=False)[0]
        person_boxes = [
            box.xyxy[0].tolist()
            for box in person_results.boxes
            if int(box.cls[0]) == 0
        ]

        # Detect PPE
        ppe_results = ppe_model.predict(frame, conf=0.5, verbose=False)[0]
        helmet_boxes = []
        jacket_boxes = []

        for box in ppe_results.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            if cls == CUSTOM_CLASSES['helmet']:
                helmet_boxes.append(coords)
            elif cls == CUSTOM_CLASSES['jacket']:
                jacket_boxes.append(coords)

        # Check each person for helmet and jacket
        for person_box in person_boxes:
            has_helmet = any(point_in_box(box_center(h), person_box) for h in helmet_boxes)
            has_jacket = any(point_in_box(box_center(j), person_box) for j in jacket_boxes)

            if not has_helmet:
                csv_writer.writerow([frame_count, timestamp, "No Helmet"])
            if not has_jacket:
                csv_writer.writerow([frame_count, timestamp, "No Jacket"])

    frame_count += 1

cap.release()
csv_file.close()
print("✅ Helmet/Jacket violations logged to 'ppe_violations.csv'")
