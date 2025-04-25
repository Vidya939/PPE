import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # Adjust path if needed

# Open the webcam
cap = cv2.VideoCapture("runs/detect/predict/main.avi")

# Make window full screen
cv2.namedWindow("YOLOv8 Live Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 Live Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Object counters
object_counts = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, stream=True)

    object_counts.clear()  # Reset counters for each frame

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Count occurrences of each class
            if label not in object_counts:
                object_counts[label] = 1
            else:
                object_counts[label] += 1

            # Add index to label (e.g., Person 1, Person 2)
            label_with_id = f"{label} {object_counts[label]} ({conf:.2f})"

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_with_id, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display frame full-screen
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
