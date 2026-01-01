# tool/web_view.py
import threading
import time
import cv2
import logging
from flask import Flask, Response, jsonify, request

# 禁用 Flask 和 Werkzeug 的日志刷屏
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

_lock = threading.Lock()
_latest_frame = None
_latest_raw_frame = None  # 用于裁剪的原始帧
_latest_metrics = {}
_paused = False
_latest_tracks_box = {}  # tid -> {x1, y1, x2, y2}

def update(frame, raw_frame=None):
    global _latest_frame, _latest_raw_frame
    with _lock:
        _latest_frame = frame
        if raw_frame is not None:
            _latest_raw_frame = raw_frame
        else:
            _latest_raw_frame = frame  # 如果没有提供raw_frame，使用frame

def update_metrics(d: dict):
    global _latest_metrics, _latest_tracks_box
    with _lock:
        _latest_metrics = dict(d)
        # 更新 tracks_box 缓存
        tracks_box = d.get("tracks_box", [])
        _latest_tracks_box = {tb["tid"]: tb for tb in tracks_box}

def get_paused() -> bool:
    with _lock:
        return bool(_paused)

def set_paused(v: bool) -> bool:
    global _paused
    with _lock:
        _paused = bool(v)
        return _paused

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/metrics", methods=["GET"])
def metrics():
    with _lock:
        return jsonify(_latest_metrics)

@app.route("/control/pause", methods=["POST", "GET", "OPTIONS"])
def control_pause():
    if request.method == "OPTIONS":
        return ("", 204)

    if request.method == "POST" and request.is_json:
        paused = bool(request.json.get("paused", False))
        return jsonify({"paused": set_paused(paused)})

    # GET ?paused=1/0 or toggle
    if "paused" in request.args:
        p = request.args.get("paused", "0")
        return jsonify({"paused": set_paused(p in ["1", "true", "True"])})

    return jsonify({"paused": set_paused(not get_paused())})

def _gen():
    while True:
        with _lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            time.sleep(0.01)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/crop/<int:tid>")
def crop_track(tid):
    """返回指定 track_id 的裁剪图像"""
    with _lock:
        # 使用 raw_frame 进行裁剪（没有侧边栏等叠加）
        frame = None if _latest_raw_frame is None else _latest_raw_frame.copy()
        box = _latest_tracks_box.get(tid)

    if frame is None or box is None:
        # 返回一个空白占位图
        placeholder = cv2.imencode(".jpg",
            cv2.putText(
                (80 * 2 + 60) * [[0, 0, 0]],  # 会报错，改用正确方式
                "No Track", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1
            )
        )
        # 简单返回 404
        return ("Not Found", 404)

    h, w = frame.shape[:2]
    x1 = max(0, int(box["x1"]))
    y1 = max(0, int(box["y1"]))
    x2 = min(w, int(box["x2"]))
    y2 = min(h, int(box["y2"]))

    if x2 <= x1 or y2 <= y1:
        return ("Invalid bbox", 404)

    crop_img = frame[y1:y2, x1:x2]

    # 调整大小到固定高度，保持比例
    target_h = 150
    ch, cw = crop_img.shape[:2]
    if ch > 0:
        scale = target_h / ch
        crop_img = cv2.resize(crop_img, (int(cw * scale), target_h))

    ok, jpg = cv2.imencode(".jpg", crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return ("Encode failed", 500)

    return Response(jpg.tobytes(), mimetype="image/jpeg")

def start_server(host="127.0.0.1", port=5000):
    def _run():
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(0.2)
