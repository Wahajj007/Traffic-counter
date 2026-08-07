import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File

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
    conn.commit()
    conn.close()

init_db()

@app.get("/")
def read_root():
    return {"message": "Traffic counter backend is alive"}

@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
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

    return {"id": video_id, "filename": file.filename, "status": "queued"}

@app.get("/videos")
def list_videos():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]