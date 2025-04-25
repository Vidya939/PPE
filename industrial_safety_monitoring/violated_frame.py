import cv2
import os
import csv
from ultralytics import YOLO

# Paths
video_path = "data/main.mp4"
csv_path = "ppe_violations.csv"
output_dir = "reviewed_violations"

# Load model
model = YOLO("runs/detect/train/weights/best.pt")  # Custom model

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Step 1: Read violation frame numbers from CSV
violated_frames = set()
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        violated_frames.add(int(row['Frame Number']))

print(f"🔍 Found {len(violated_frames)} violated frames.")

# Step 2: Load video
cap = cv2.VideoCapture(video_path)

# Step 3: Process each violated frame
for frame_num in sorted(violated_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"⚠️ Could not read frame {frame_num}")
        continue

    # Run detection
    results = model.predict(frame, conf=0.5, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if label in ['helmet', 'jacket'] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Save reviewed frame
    output_path = os.path.join(output_dir, f"frame_{frame_num}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"✅ Saved: {output_path}")

cap.release()
print("🎉 Auto-review completed! Check the 'reviewed_violations/' folder.")
