import cv2
import time
import threading
import numpy as np
from flask import Flask, Response

app = Flask(__name__)
latest_frame = None
lock = threading.Lock()

def update_frame(frame):
    global latest_frame
    with lock:
        latest_frame = frame.copy()

def gen():
    global latest_frame
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        # 强制 BGR -> JPEG
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )
        time.sleep(0.03)  # ~30 FPS

@app.route("/video")
def video():
    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def start_server():
    t = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=5000, threaded=True, debug=False),
        daemon=True
    )
    t.start()
