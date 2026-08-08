import cv2
from detection import detect_vehicles
from counting import LineCounter

video_path = "uploads/traffic_fixed.mp4"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

counter = LineCounter(frame_width=width, frame_height=height, line_y_fraction=0.5)

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, confidences, class_names = detect_vehicles(frame)
    counter.update(boxes, confidences, class_names)

    frame_num += 1

cap.release()

print(f"Processed {frame_num} frames")
print(f"Total vehicles counted: {len(counter.crossings)}")
for track_id, vehicle_class, direction in counter.crossings:
    print(f"  Vehicle #{track_id} ({vehicle_class}) — {direction}")