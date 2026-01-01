import numpy as np

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(1e-6, (ax2-ax1) * (ay2-ay1))
    area_b = max(1e-6, (bx2-bx1) * (by2-by1))
    return inter / (area_a + area_b - inter + 1e-6)

class Track:
    def __init__(self, tid, xyxy, ts_ms):
        self.id = tid
        self.xyxy = np.array(xyxy, dtype=np.float32)
        self.hits = 1
        self.age = 0
        self.last_ts = ts_ms

        self.traj = []  # list[(ts_ms, cx, cy)]
        self._push_traj(ts_ms)

        self.prev_xyxy = self.xyxy.copy()
        self.prev_speed = 0.0
        self.prev_aspect = self._aspect()

        self.loiter_start_ms = None

    def _aspect(self):
        w = max(1.0, float(self.xyxy[2]-self.xyxy[0]))
        h = max(1.0, float(self.xyxy[3]-self.xyxy[1]))
        return h / w

    def _center(self):
        cx = float((self.xyxy[0] + self.xyxy[2]) / 2)
        cy = float((self.xyxy[1] + self.xyxy[3]) / 2)
        return cx, cy

    def _push_traj(self, ts_ms, maxlen=30):
        cx, cy = self._center()
        self.traj.append((ts_ms, cx, cy))
        if len(self.traj) > maxlen:
            self.traj = self.traj[-maxlen:]

    def update(self, xyxy, ts_ms):
        self.age = 0
        self.hits += 1

        self.prev_xyxy = self.xyxy.copy()
        self.xyxy = np.array(xyxy, dtype=np.float32)

        # speed estimate (px/s)
        dt = max(1e-3, (ts_ms - self.last_ts) / 1000.0)
        cx0, cy0 = (float((self.prev_xyxy[0] + self.prev_xyxy[2]) / 2),
                    float((self.prev_xyxy[1] + self.prev_xyxy[3]) / 2))
        cx1, cy1 = self._center()
        dist = ((cx1-cx0)**2 + (cy1-cy0)**2) ** 0.5
        self.prev_speed = dist / dt

        self.prev_aspect = self._aspect()
        self.last_ts = ts_ms
        self._push_traj(ts_ms)

    def mark_missed(self):
        self.age += 1

class IOUTracker:
    def __init__(self, iou_match=0.3, max_age=20, min_hits=3):
        self.iou_match = iou_match
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self._next_id = 1

    def update(self, dets, ts_ms):
        det_boxes = [d["xyxy"] for d in dets]

        # match tracks to dets (greedy by IOU)
        unmatched_dets = set(range(len(det_boxes)))
        used_tracks = set()

        matches = []
        # compute all IOUs
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di in unmatched_dets:
                pairs.append((iou(t.xyxy, det_boxes[di]), ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for iou_val, ti, di in pairs:
            if iou_val < self.iou_match:
                break
            if ti in used_tracks or di not in unmatched_dets:
                continue
            used_tracks.add(ti)
            unmatched_dets.remove(di)
            matches.append((ti, di))

        # update matched
        for ti, di in matches:
            self.tracks[ti].update(det_boxes[di], ts_ms)

        # mark missed
        for ti, t in enumerate(self.tracks):
            if ti not in used_tracks:
                t.mark_missed()

        # create new for unmatched dets
        for di in list(unmatched_dets):
            t = Track(self._next_id, det_boxes[di], ts_ms)
            self._next_id += 1
            self.tracks.append(t)

        # prune dead
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # output active tracks
        out = []
        for t in self.tracks:
            out.append({
                "id": t.id,
                "xyxy": t.xyxy.tolist(),
                "hits": t.hits,
                "age": t.age,
                "traj": t.traj,
                "speed": t.prev_speed,
                "aspect": t._aspect(),
            })
        return out
