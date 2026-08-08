import numpy as np
import supervision as sv

class LineCounter:
    def __init__(self, frame_width, frame_height, line_y_fraction=0.5):
        y = int(frame_height * line_y_fraction)
        self.line_zone = sv.LineZone(
            start=sv.Point(0, y),
            end=sv.Point(frame_width, y),
        )
        self.tracker = sv.ByteTrack()
        self.counted_ids = set()
        self.crossings = []

    def update(self, boxes, confidences, class_names):
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=self._class_names_to_ids(class_names),
        )

        tracked = self.tracker.update_with_detections(detections)
        crossed_in, crossed_out = self.line_zone.trigger(tracked)

        for i, track_id in enumerate(tracked.tracker_id):
            track_id = int(track_id)
            if track_id in self.counted_ids:
                continue

            if crossed_in[i]:
                self.crossings.append((track_id, class_names[i] if i < len(class_names) else "vehicle", "in"))
                self.counted_ids.add(track_id)
            elif crossed_out[i]:
                self.crossings.append((track_id, class_names[i] if i < len(class_names) else "vehicle", "out"))
                self.counted_ids.add(track_id)

    def _class_names_to_ids(self, class_names):
        unique = {name: idx for idx, name in enumerate(set(class_names))}
        return np.array([unique[name] for name in class_names])