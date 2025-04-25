import cv2
import os

# --- Input and output paths ---
video_path = 'data/main.mp4'
output_folder = 'output/frames'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# --- Open the video file ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 5)  # Number of frames to skip to get 5 fps

count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f'frame_{saved_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    count += 1

cap.release()
print(f"Extracted {saved_count} frames at 5 FPS into '{output_folder}'")
