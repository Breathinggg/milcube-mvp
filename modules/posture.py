import time

class PostureEngine:
    """
    无骨架姿态：用 bbox 形状 + 速度做 SIT/STAND/FALL?
    带 debounce 和 FALL hold。
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_update_ms = 0

        # track_id -> state machine data
        self.state = {}  # id: {label, cand, cand_count, fall_until_ms}
        self.last_labels = {}  # 缓存上次的 labels，避免闪烁

    def update(self, tracks, frame_shape, now_ms):
        cfg = self.cfg
        H, W = frame_shape[:2]

        # 限频更新（省算力+更稳）
        if now_ms - self.last_update_ms < int(1000.0 / max(0.1, cfg.posture_update_hz)):
            return self.last_labels, []  # 返回缓存的 labels，避免闪烁
        self.last_update_ms = now_ms

        labels = {}
        events = []

        alive_ids = set()

        for t in tracks:
            tid = t["id"]
            alive_ids.add(tid)

            x1, y1, x2, y2 = t["xyxy"]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            h_ratio = bh / max(1.0, H)
            aspect = bh / bw
            speed = float(t.get("speed", 0.0))

            st = self.state.get(tid)
            if st is None:
                st = {"label": "UNK", "cand": "UNK", "cand_count": 0, "fall_until_ms": 0}
                self.state[tid] = st

            # FALL hold：在 hold 时间内强制 FALL?
            if now_ms < st["fall_until_ms"]:
                labels[tid] = "FALL?"
                continue

            cand = self._classify(cfg, h_ratio, aspect, speed)

            # 如果是 FALL? 立即触发，并 hold
            if cand == "FALL?":
                st["label"] = "FALL?"
                st["cand"] = "FALL?"
                st["cand_count"] = cfg.posture_debounce
                st["fall_until_ms"] = now_ms + int(cfg.fall_hold_seconds * 1000)

                events.append({
                    "type": "FALL_SUSPECT",
                    "track_id": tid,
                    "ts_ms": now_ms,
                    "detail": {"h_ratio": h_ratio, "aspect": aspect, "speed": speed}
                })
                labels[tid] = "FALL?"
                continue

            # debounce：连续 cfg.posture_debounce 次才切换
            if cand == st["cand"]:
                st["cand_count"] += 1
            else:
                st["cand"] = cand
                st["cand_count"] = 1

            if st["cand_count"] >= cfg.posture_debounce:
                st["label"] = cand

            labels[tid] = st["label"]

        # 清理消失的 track 状态
        dead = [tid for tid in self.state.keys() if tid not in alive_ids]
        for tid in dead:
            self.state.pop(tid, None)
            self.last_labels.pop(tid, None)

        self.last_labels = labels  # 缓存本次结果
        return labels, events

    def _classify(self, cfg, h_ratio, aspect, speed):
        # FALL?（更强信号，优先级最高）
        if aspect <= cfg.fall_aspect_max and speed >= cfg.fall_speed_min:
            return "FALL?"

        # STAND
        if h_ratio >= cfg.stand_h_ratio and aspect >= cfg.stand_aspect_min:
            return "STAND"

        # SIT
        if (cfg.sit_h_ratio_min <= h_ratio <= cfg.sit_h_ratio_max and
            cfg.sit_aspect_min <= aspect <= cfg.sit_aspect_max and
            speed <= cfg.sit_speed_max):
            return "SIT"

        return "UNK"
