import csv
import json
import os
import time

class MVPLogger:
    def __init__(self, out_dir="runs"):
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(out_dir, ts)
        os.makedirs(self.dir, exist_ok=True)

        self.metrics_path = os.path.join(self.dir, "metrics.csv")
        self.events_path = os.path.join(self.dir, "events.jsonl")

        self._csv_file = open(self.metrics_path, "w", newline="", encoding="utf-8")
        self._writer = None

        self._events = open(self.events_path, "w", encoding="utf-8")

        self._last_metrics_sec = -1

    def log_metrics(self, d: dict):
        # 每秒写一行，避免太大
        sec = int(d.get("ts_ms", 0) / 1000)
        if sec == self._last_metrics_sec:
            return
        self._last_metrics_sec = sec

        if self._writer is None:
            fields = list(d.keys())
            self._writer = csv.DictWriter(self._csv_file, fieldnames=fields)
            self._writer.writeheader()
        self._writer.writerow(d)
        self._csv_file.flush()

    def log_events(self, events: list):
        for e in events:
            self._events.write(json.dumps(e, ensure_ascii=False) + "\n")
        if events:
            self._events.flush()

    def close(self):
        try:
            self._csv_file.close()
            self._events.close()
        except Exception:
            pass
