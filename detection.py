from ultralytics import YOLO

# COCO class IDs for vehicles (the pretrained model already knows these)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

model = YOLO("yolo11n.pt")  # "n" = nano, the smallest/fastest version

def detect_vehicles(frame):
    results = model.predict(
        frame,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.35,
        verbose=False,
    )[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = [VEHICLE_CLASSES[c] for c in class_ids]

    return boxes, confidences, class_names
