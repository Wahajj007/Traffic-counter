import cv2
from detection import detect_vehicles

# Grab any video you already uploaded — check your uploads/ folder for the exact filename
video_path = "uploads/a1b6851d-0d79-4c9f-aaaa-14621c35c5c1_traffic.mp4"

cap = cv2.VideoCapture(video_path)
print("Opened successfully:", cap.isOpened())
ret, frame = cap.read()  # read just the first frame
cap.release()

if not ret:
    print("Couldn't read the video — check the path")
else:
    boxes, confidences, class_names = detect_vehicles(frame)
    print(f"Found {len(boxes)} vehicles")
    for box, conf, name in zip(boxes, confidences, class_names):
        print(f"  {name} — confidence {conf:.2f} — box {box}")