import cv2
import time

class VideoSource:
    def __init__(self, uri: str, name: str, prefer_w=None, prefer_h=None):
        self.uri = uri
        self.name = name
        self.prefer_w = prefer_w
        self.prefer_h = prefer_h
        self.cap = None
        self.last_ok = False
        self._open()

    def _open(self):
        if self.uri.isdigit():
            self.cap = cv2.VideoCapture(int(self.uri), cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1:
            self.fps = 10.0  # fallback，EPFL 就是 10 fps

        self.frame_interval_s = 1.0 / self.fps
        self.last_read_time = time.time()

        self.last_ok = self.cap.isOpened()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return None, int(time.time() * 1000), False

        # ✅ 等到该显示下一帧的时间
        now = time.time()
        wait = self.frame_interval_s - (now - self.last_read_time)
        if wait > 0:
            time.sleep(wait)

        ok, frame = self.cap.read()
        self.last_read_time = time.time()

        ts = int(self.last_read_time * 1000)
        self.last_ok = ok
        return frame, ts, ok

    def reconnect(self, sleep_s=1.0):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        time.sleep(sleep_s)
        self._open()

    def release(self):
        if self.cap is not None:
            self.cap.release()

