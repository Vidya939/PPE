import cv2
from ultralytics import YOLO
import numpy as np

# --- Load YOLO Models ---
person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("runs/detect/train/weights/best.pt")

# PPE Class Names (match your trained model)
ppe_classes = {
    0: 'helmet',
    1: 'no_helmet',
    2: 'no_jacket',
    3: 'jacket'
}

# Colors for visual labels
class_colors = {
    'person': (255, 255, 0),
    'helmet': (255, 0, 0),
    'no_helmet': (0, 0, 255),
    'jacket': (0, 255, 0),
    'no_jacket': (0, 165, 255)
}

# Define HSV color ranges (adjust red-like hues as orange)
COLOR_RANGES = {
    'orange': [(5, 100, 100), (25, 255, 255)],  # Adjusted for red and orange-like hues
    'yellow': [(26, 100, 100), (34, 255, 255)],
    'green': [(35, 100, 100), (85, 255, 255)],
    'blue': [(90, 50, 50), (130, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'black': [(0, 0, 0), (180, 255, 50)]
}

def detect_color(image):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
        if ratio > 0.3:  # More than 30% of the region is this color
            return color
    return "unknown"

cap = cv2.VideoCapture("data/sample1 (8).mov")

cv2.namedWindow("Live PPE Color Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Live PPE Color Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Counts
    color_counts = {"helmet": {}, "jacket": {}, "no_helmet": 0, "no_jacket": 0}

    # --- Person Detection ---
    person_results = person_model.predict(source=frame, conf=0.7, stream=True)
    for result in person_results:
        for box in result.boxes:
            if int(box.cls) == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors['person'], 2)
                cv2.putText(frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors['person'], 2)

    # --- PPE Detection ---
    ppe_results = ppe_model.predict(source=frame, conf=0.7, stream=True)
    for result in ppe_results:
        for box in result.boxes:
            cls_id = int(box.cls)
            label_name = ppe_classes.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            color = detect_color(cropped) if label_name in ["helmet", "jacket"] else None

            # Count by color
            if label_name == "helmet":
                color_counts["helmet"][color] = color_counts["helmet"].get(color, 0) + 1
            elif label_name == "jacket":
                color_counts["jacket"][color] = color_counts["jacket"].get(color, 0) + 1
            elif label_name == "no_helmet":
                color_counts["no_helmet"] += 1
            elif label_name == "no_jacket":
                color_counts["no_jacket"] += 1

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), class_colors[label_name], 2)
            label = f"{label_name}"
            if color:
                label += f" ({color})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_colors[label_name], 2)

    # --- Show Color Counts on Screen ---
    y_offset = 30
    for label in ["helmet", "jacket"]:
        for color, count in color_counts[label].items():
            cv2.putText(frame, f"{color.capitalize()} {label}s: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
    cv2.putText(frame, f"No Helmet: {color_counts['no_helmet']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_offset += 25
    cv2.putText(frame, f"No Jacket: {color_counts['no_jacket']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Display
    cv2.imshow("Live PPE Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
