import cv2
import time
import threading
from flask import Flask, Response

app = Flask(__name__)
frame = None

def gen():
    global frame
    while True:
        if frame is None:
            time.sleep(0.01)
            continue

        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg.tobytes() +
            b"\r\n"
        )
        time.sleep(0.03)

@app.route("/video")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def run_server():
    app.run("127.0.0.1", 5000, threaded=True, debug=False)

if __name__ == "__main__":
    # 启动 server
    threading.Thread(target=run_server, daemon=True).start()

    cap = cv2.VideoCapture("cam5.mp4")
    assert cap.isOpened(), "video open failed"

    while True:
        ret, f = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = f
        time.sleep(0.1)  # 10 fps