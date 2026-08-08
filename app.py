import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import cv2
from detection import detect_vehicles
from counting import LineCounter
from fastapi.responses import FileResponse


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_PATH = "traffic.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS count_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            vehicle_class TEXT NOT NULL,
            direction TEXT NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos (id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def process_video(video_id: str, video_path: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE videos SET status = ? WHERE id = ?", ("processing", video_id))
    conn.commit()

    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        counter = LineCounter(frame_width=width, frame_height=height, line_y_fraction=0.5)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            boxes, confidences, class_names = detect_vehicles(frame)
            counter.update(boxes, confidences, class_names)

        cap.release()

        for track_id, vehicle_class, direction in counter.crossings:
            conn.execute(
                "INSERT INTO count_events (video_id, track_id, vehicle_class, direction) VALUES (?, ?, ?, ?)",
                (video_id, track_id, vehicle_class, direction)
            )

        conn.execute("UPDATE videos SET status = ? WHERE id = ?", ("completed", video_id))
        conn.commit()

    except Exception as e:
        conn.execute("UPDATE videos SET status = ? WHERE id = ?", ("failed", video_id))
        conn.commit()
        print(f"Processing failed for {video_id}: {e}")

    finally:
        conn.close()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")
@app.post("/upload")
def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO videos (id, filename, storage_path, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (video_id, file.filename, save_path, "queued", datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    background_tasks.add_task(process_video, video_id, save_path)

    return {"id": video_id, "filename": file.filename, "status": "queued"}

@app.get("/videos")
def list_videos():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.get("/videos/{video_id}/summary")
def get_summary(video_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT vehicle_class, direction, COUNT(*) as count FROM count_events WHERE video_id = ? GROUP BY vehicle_class, direction",
        (video_id,)
    ).fetchall()
    conn.close()

    total = sum(row["count"] for row in rows)
    by_class = {}
    for row in rows:
        by_class[row["vehicle_class"]] = by_class.get(row["vehicle_class"], 0) + row["count"]

    return {
        "video_id": video_id,
        "total_count": total,
        "by_class": by_class,
        "breakdown": [dict(row) for row in rows]
    }