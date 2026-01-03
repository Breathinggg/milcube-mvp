class PostureEngine:
    """
    状态机姿态检测：WALK / STAND
    基于速度的帧计数，避免闪烁
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = {}
        self.last_labels = {}

        # 状态机参数
        self.speed_threshold = 3.0    # 速度阈值（像素/秒）
        self.walk_enter_count = 2     # 连续2次检测到移动就切换到WALK
        self.stand_enter_count = 3    # 连续3次检测到静止就切换到STAND

    def update(self, tracks, frame_shape, now_ms):
        labels = {}
        events = []
        current_ids = set()

        for t in tracks:
            tid = t["id"]
            current_ids.add(tid)
            speed = float(t.get("speed", 0.0))
            age = t.get("age", 0)

            # 丢失的 track 保持上一次的标签
            if age > 0:
                labels[tid] = self.state.get(tid, {}).get("label", "STAND")
                continue

            st = self.state.get(tid)
            if st is None:
                st = {
                    "label": "STAND",
                    "moving_count": 0,   # 连续移动次数
                    "stopped_count": 0,  # 连续静止次数
                }
                self.state[tid] = st

            # 只有当 speed > 0 时才更新状态（跳过无效帧）
            if speed == 0:
                labels[tid] = st["label"]
                continue

            is_moving = speed > self.speed_threshold
            current_label = st["label"]

            if is_moving:
                st["moving_count"] += 1
                st["stopped_count"] = 0

                # STAND -> WALK
                if current_label == "STAND" and st["moving_count"] >= self.walk_enter_count:
                    st["label"] = "WALK"
            else:
                st["stopped_count"] += 1
                st["moving_count"] = 0

                # WALK -> STAND
                if current_label == "WALK" and st["stopped_count"] >= self.stand_enter_count:
                    st["label"] = "STAND"

            labels[tid] = st["label"]

        # 清理消失的 track
        dead = [tid for tid in self.state.keys() if tid not in current_ids]
        for tid in dead:
            del self.state[tid]

        self.last_labels = labels
        return labels, events

    def get_fall_tracks(self):
        return {}
