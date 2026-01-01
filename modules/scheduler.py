import time

class TokenBucketScheduler:
    """
    强制“Jetson式”推理预算：单模型串行、总推理Hz固定，多路时间片分摊。
    每路一个 token bucket：按该路分配的Hz积累token，推理一次消耗1 token。
    """
    def __init__(self, total_hz: float, stream_weights: dict):
        self.total_hz = float(total_hz)
        s = sum(stream_weights.values())
        self.stream_hz = {k: (total_hz * v / s) for k, v in stream_weights.items()}

        self.tokens = {k: 0.0 for k in stream_weights}
        self.last_ms = int(time.time() * 1000)

        # 单模型串行：用一个全局“忙锁”
        self.busy_until_ms = 0

    def tick(self, now_ms: int):
        dt = max(0, now_ms - self.last_ms) / 1000.0
        self.last_ms = now_ms
        for k, hz in self.stream_hz.items():
            self.tokens[k] = min(2.0, self.tokens[k] + hz * dt)  # token上限，防止爆发

    def can_run(self, stream_id: str, now_ms: int) -> bool:
        if now_ms < self.busy_until_ms:
            return False
        return self.tokens.get(stream_id, 0.0) >= 1.0

    def consume(self, stream_id: str, now_ms: int, min_infer_ms: int = 0):
        # 消耗token并可选“人为限速”让 infer 看起来像 Jetson
        self.tokens[stream_id] = max(0.0, self.tokens.get(stream_id, 0.0) - 1.0)
        self.busy_until_ms = now_ms + int(min_infer_ms)
