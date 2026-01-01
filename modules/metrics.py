import numpy as np
import time

def roi_grid_index(cx, cy, W, H, gw, gh):
    x = min(gw-1, max(0, int(cx / max(1, W) * gw)))
    y = min(gh-1, max(0, int(cy / max(1, H) * gh)))
    return x, y

class MetricsEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_anomaly_ms = 0

        # 用于估计“主方向”（全局平均速度向量，简化版）
        self.flow_vec = np.array([0.0, 0.0], dtype=np.float32)
        self.flow_decay = 0.9

        # track_id -> last_roi, roi_enter_ts
        self.track_roi_state = {}

    def update(self, tracks, frame_shape, ts_capture_ms, ts_now_ms):
        H, W = frame_shape[:2]
        cfg = self.cfg

        # 计数：hits>=min_hits 且 age==0/小于阈值 的算“在场”
        active = []
        for t in tracks:
            if t["hits"] >= cfg.track_min_hits and t["age"] <= 2:
                active.append(t)

        count = len(active)

        # 密度网格
        grid = np.zeros((cfg.grid_h, cfg.grid_w), dtype=np.int32)
        for t in active:
            x1,y1,x2,y2 = t["xyxy"]
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            gx, gy = roi_grid_index(cx, cy, W, H, cfg.grid_w, cfg.grid_h)
            grid[gy, gx] += 1

        # 更新主方向（用轨迹末两点速度向量粗估）
        for t in active:
            traj = t["traj"]
            if len(traj) >= 2:
                (t0,x0,y0), (t1,x1,y1) = traj[-2], traj[-1]
                dt = max(1e-3, (t1 - t0)/1000.0)
                v = np.array([(x1-x0)/dt, (y1-y0)/dt], dtype=np.float32)
                self.flow_vec = self.flow_decay*self.flow_vec + (1-self.flow_decay)*v

        events = []
        # 低频异常
        if ts_now_ms - self.last_anomaly_ms >= int(1000.0 / max(0.1, cfg.anomaly_hz)):
            self.last_anomaly_ms = ts_now_ms
            events.extend(self._detect_anomaly(active, W, H, ts_now_ms))

        metrics = {
            "ts_ms": ts_now_ms,
            "count": count,
            "flow_vx": float(self.flow_vec[0]),
            "flow_vy": float(self.flow_vec[1]),
            "grid_sum": int(grid.sum()),
        }
        return metrics, grid, events

    def _detect_anomaly(self, active_tracks, W, H, ts_ms):
        cfg = self.cfg
        events = []

        # 简化：以网格作为ROI
        for t in active_tracks:
            x1,y1,x2,y2 = t["xyxy"]
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            gx, gy = roi_grid_index(cx, cy, W, H, cfg.grid_w, cfg.grid_h)
            roi = (gx, gy)

            st = self.track_roi_state.get(t["id"])
            if st is None or st["roi"] != roi:
                self.track_roi_state[t["id"]] = {"roi": roi, "enter_ms": ts_ms}
                st = self.track_roi_state[t["id"]]

            # 滞留
            if ts_ms - st["enter_ms"] >= int(cfg.loiter_seconds * 1000):
                events.append({
                    "type": "LOITER",
                    "track_id": t["id"],
                    "roi": roi,
                    "ts_ms": ts_ms,
                    "detail": {"seconds": (ts_ms - st["enter_ms"]) / 1000.0}
                })
                # 触发后重置，避免刷屏
                self.track_roi_state[t["id"]]["enter_ms"] = ts_ms

            # 逆向（与主方向夹角大 + 有一定速度）
            v = np.array([0.0, 0.0], dtype=np.float32)
            traj = t["traj"]
            if len(traj) >= 2:
                (t0,x0,y0), (t1,x1,y1) = traj[-2], traj[-1]
                dt = max(1e-3, (t1-t0)/1000.0)
                v = np.array([(x1-x0)/dt, (y1-y0)/dt], dtype=np.float32)
            speed = float(np.linalg.norm(v))
            flow = np.array([cfg.fall_speed_spike, cfg.fall_speed_spike], dtype=np.float32)  # 防止flow=0导致除0
            # 用全局flow_vec方向（若很小就跳过）
            # 这里不引入更复杂判断，保持MVP
            # 跌倒粗判：用 bbox 纵横比变化 + speed 突变
            # t["aspect"] 是当前高宽比
            # 这里我们用：aspect < 某阈值 且 speed 较大
            if t["aspect"] < 0.9 and t["speed"] > cfg.fall_speed_spike:
                events.append({
                    "type": "FALL_SUSPECT",
                    "track_id": t["id"],
                    "roi": roi,
                    "ts_ms": ts_ms,
                    "detail": {"aspect": t["aspect"], "speed": t["speed"]}
                })

        return events
