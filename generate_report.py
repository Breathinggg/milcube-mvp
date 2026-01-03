"""
MilCube MVP - Performance Report Generator
生成可读性高的性能统计报告
"""
import os
import csv
import json
from datetime import datetime
from pathlib import Path

def find_latest_run(runs_dir="runs"):
    """找到最新的运行日志目录"""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None

    all_runs = []
    for cam_dir in runs_path.iterdir():
        if cam_dir.is_dir():
            for run_dir in cam_dir.iterdir():
                if run_dir.is_dir():
                    all_runs.append(run_dir)

    if not all_runs:
        return None

    # 按名称排序（时间戳格式）
    all_runs.sort(key=lambda x: x.name, reverse=True)
    return all_runs

def parse_metrics_csv(csv_path):
    """解析 metrics.csv 文件"""
    metrics = []
    if not csv_path.exists():
        return metrics

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append(row)
    return metrics

def parse_events_jsonl(jsonl_path):
    """解析 events.jsonl 文件"""
    events = []
    if not jsonl_path.exists():
        return events

    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except:
                    pass
    return events

def calculate_stats(metrics):
    """计算性能统计数据"""
    if not metrics:
        return None

    stats = {
        "total_frames": len(metrics),
        "duration_seconds": 0,
        "avg_fps": 0,
        "avg_infer_ms": 0,
        "avg_latency_ms": 0,
        "avg_person_count": 0,
        "max_person_count": 0,
        "min_person_count": 999,
        "total_inferences": 0,
    }

    fps_list = []
    infer_list = []
    latency_list = []
    count_list = []

    for m in metrics:
        try:
            fps = float(m.get("fps", 0))
            if fps > 0:
                fps_list.append(fps)

            infer_ms = float(m.get("infer_ms_wide", -1))
            if infer_ms > 0:
                infer_list.append(infer_ms)
                stats["total_inferences"] += 1

            latency = float(m.get("latency_ms", 0))
            if latency > 0:
                latency_list.append(latency)

            count = int(float(m.get("raw_count", 0)))
            count_list.append(count)
            stats["max_person_count"] = max(stats["max_person_count"], count)
            stats["min_person_count"] = min(stats["min_person_count"], count)
        except:
            pass

    if fps_list:
        stats["avg_fps"] = sum(fps_list) / len(fps_list)
        stats["duration_seconds"] = stats["total_frames"] / stats["avg_fps"]

    if infer_list:
        stats["avg_infer_ms"] = sum(infer_list) / len(infer_list)

    if latency_list:
        stats["avg_latency_ms"] = sum(latency_list) / len(latency_list)

    if count_list:
        stats["avg_person_count"] = sum(count_list) / len(count_list)

    if stats["min_person_count"] == 999:
        stats["min_person_count"] = 0

    return stats

def count_events_by_type(events):
    """统计事件类型"""
    event_counts = {}
    for e in events:
        etype = e.get("type", "UNKNOWN")
        event_counts[etype] = event_counts.get(etype, 0) + 1
    return event_counts

def generate_report(runs_dir="runs", output_file="performance_report.txt"):
    """生成性能报告"""

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("         MilCube MVP - Performance Report")
    report_lines.append("         Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report_lines.append("=" * 70)
    report_lines.append("")

    runs = find_latest_run(runs_dir)

    if not runs:
        report_lines.append("No run data found in 'runs/' directory.")
        report_lines.append("Please run the system first to generate logs.")
    else:
        # 按摄像头分组
        cam_runs = {}
        for run in runs:
            cam_name = run.parent.name
            if cam_name not in cam_runs:
                cam_runs[cam_name] = []
            cam_runs[cam_name].append(run)

        total_events = 0
        total_frames = 0
        all_event_counts = {}

        for cam_name, cam_run_list in sorted(cam_runs.items()):
            # 取最新的运行
            latest_run = cam_run_list[0]

            report_lines.append("-" * 70)
            report_lines.append(f"  Camera: {cam_name}")
            report_lines.append(f"  Run: {latest_run.name}")
            report_lines.append("-" * 70)

            # 解析数据
            metrics_path = latest_run / "metrics.csv"
            events_path = latest_run / "events.jsonl"

            metrics = parse_metrics_csv(metrics_path)
            events = parse_events_jsonl(events_path)

            stats = calculate_stats(metrics)
            event_counts = count_events_by_type(events)

            if stats:
                report_lines.append("")
                report_lines.append("  [Performance Metrics]")
                report_lines.append(f"    Total Frames Processed:  {stats['total_frames']}")
                report_lines.append(f"    Duration:                {stats['duration_seconds']:.1f} seconds")
                report_lines.append(f"    Average FPS:             {stats['avg_fps']:.2f}")
                report_lines.append(f"    Average Inference Time:  {stats['avg_infer_ms']:.2f} ms")
                report_lines.append(f"    Average Latency:         {stats['avg_latency_ms']:.2f} ms")
                report_lines.append(f"    Total Inferences:        {stats['total_inferences']}")
                report_lines.append("")
                report_lines.append("  [Detection Statistics]")
                report_lines.append(f"    Average Person Count:    {stats['avg_person_count']:.2f}")
                report_lines.append(f"    Max Person Count:        {stats['max_person_count']}")
                report_lines.append(f"    Min Person Count:        {stats['min_person_count']}")

                total_frames += stats['total_frames']

            if event_counts:
                report_lines.append("")
                report_lines.append("  [Events Detected]")
                for etype, count in sorted(event_counts.items()):
                    report_lines.append(f"    {etype}: {count}")
                    all_event_counts[etype] = all_event_counts.get(etype, 0) + count
                    total_events += count
            else:
                report_lines.append("")
                report_lines.append("  [Events Detected]")
                report_lines.append("    No events detected")

            report_lines.append("")

        # 汇总
        report_lines.append("=" * 70)
        report_lines.append("  SUMMARY")
        report_lines.append("=" * 70)
        report_lines.append(f"  Total Cameras:       {len(cam_runs)}")
        report_lines.append(f"  Total Frames:        {total_frames}")
        report_lines.append(f"  Total Events:        {total_events}")

        if all_event_counts:
            report_lines.append("")
            report_lines.append("  Event Breakdown:")
            for etype, count in sorted(all_event_counts.items()):
                report_lines.append(f"    - {etype}: {count}")

        report_lines.append("")
        report_lines.append("=" * 70)

    # 写入文件
    report_content = "\n".join(report_lines)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

    # 同时打印到控制台
    print(report_content)
    print(f"\nReport saved to: {output_file}")

    return report_content

if __name__ == "__main__":
    generate_report()
