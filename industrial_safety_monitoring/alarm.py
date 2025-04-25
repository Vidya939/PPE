import cv2
from ultralytics import YOLO
import numpy as np
import csv
from datetime import datetime
import threading
import os
import pygame
import time

# Initialize pygame for audio playback
pygame.mixer.init()

# Load models
person_model = YOLO("yolov8n.pt")
ppe_model = YOLO("runs/detect/train/weights/best.pt")

# Class names
ppe_classes = {
    0: 'helmet',
    1: 'no_helmet',
    2: 'no_jacket',
    3: 'jacket'
}

# Color detection ranges
COLOR_RANGES = {
    'orange': [(0, 100, 100), (25, 255, 255)],
    'yellow': [(26, 100, 100), (34, 255, 255)],
    'green': [(35, 100, 100), (85, 255, 255)],
    'blue': [(90, 50, 50), (130, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'black': [(0, 0, 0), (180, 255, 50)]
}

# CSV logging setup
csv_filename = "violations_log.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Violation Type"])

# Global flag to control alarm
violation_active = False

# Alarm thread function
def alarm_loop(sound_file):
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play(-1)  # Loop forever
    while violation_active:
        time.sleep(0.1)
    pygame.mixer.music.stop()

# Detect dominant color
def detect_color(image):
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
        if ratio > 0.3:
            return color
    return "unknown"

# Video source
cap = cv2.VideoCapture("data/sample1.mp4")
cv2.namedWindow("Live PPE Color Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Live PPE Color Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    color_counts = {"helmet": {}, "jacket": {}, "no_helmet": 0, "no_jacket": 0}
    helmet_violation = False
    jacket_violation = False

    # Person detection
    person_results = person_model.predict(source=frame, conf=0.7, stream=True)
    for result in person_results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255, 255, 0), 2)

    # PPE detection
    ppe_results = ppe_model.predict(source=frame, conf=0.7, stream=True)
    for result in ppe_results:
        for box in result.boxes:
            cls_id = int(box.cls)
            label_name = ppe_classes.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = frame[y1:y2, x1:x2]
            color = detect_color(cropped) if label_name in ["helmet", "jacket"] else None

            if label_name == "helmet":
                color_counts["helmet"][color] = color_counts["helmet"].get(color, 0) + 1
            elif label_name == "jacket":
                color_counts["jacket"][color] = color_counts["jacket"].get(color, 0) + 1
            elif label_name == "no_helmet":
                color_counts["no_helmet"] += 1
                helmet_violation = True
            elif label_name == "no_jacket":
                color_counts["no_jacket"] += 1
                jacket_violation = True

            box_color = (0, 255, 0) if "helmet" in label_name or "jacket" in label_name else (0, 0, 255)
            label = f"{label_name}"
            if color:
                label += f" ({color})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 2)

    # Show color counts
    y_offset = 30
    for label in ["helmet", "jacket"]:
        for color, count in color_counts[label].items():
            cv2.putText(frame, f"{color.capitalize()} {label}s: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            y_offset += 35
    cv2.putText(frame, f"No Helmet: {color_counts['no_helmet']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    y_offset += 35
    cv2.putText(frame, f"No Jacket: {color_counts['no_jacket']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)

    # Violation message
    alert_msg = ""
    if helmet_violation and jacket_violation:
        alert_msg = "No Helmet & Jacket Detected!"
    elif helmet_violation:
        alert_msg = "No Helmet Detected!"
    elif jacket_violation:
        alert_msg = "No Jacket Detected!"

    # Show alert on screen with bold letters
    if alert_msg:
        text_size = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 4.5, 8)[0]  # Increased thickness for bold
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        cv2.putText(frame, alert_msg, (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.5, (0, 0, 255), 8)  # Increased thickness        # Log to CSV

        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_msg])

    # Alarm control
    if (helmet_violation or jacket_violation) and not violation_active:
        violation_active = True
        threading.Thread(target=alarm_loop, args=("violation_alarm.wav",), daemon=True).start()
    if not (helmet_violation or jacket_violation) and violation_active:
        violation_active = False

    # Display frame
    screen_res = (1920, 1080)
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    resized_frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("Live PPE Color Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
