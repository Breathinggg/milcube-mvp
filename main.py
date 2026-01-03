import argparse
import os
import time
import cv2
import numpy as np
from tool.web_view import start_server as mjpeg_start, update as update_frame, update_metrics, get_paused

from modules.posture import PostureEngine
from modules.viz import draw_kpi_cards_on_panel
from config import JetsonLikeConfig
from modules.video_source import VideoSource
from modules.scheduler import TokenBucketScheduler
from modules.yolo_onnx import YOLOv8ONNX
from modules.tracker_iou import IOUTracker
from modules.metrics import MetricsEngine
from modules.viz import draw_grid_overlay, draw_tracks, crop_center
from modules.logger import MVPLogger

# Re-ID feature extractor
try:
    from tool.reid_extractor import ReIDExtractor
    REID_AVAILABLE = True
except ImportError:
    REID_AVAILABLE = False
    ReIDExtractor = None
# 你原来从 modules.viz import side_by_side，但你本地又定义了同名函数
# 为避免混乱，这里直接用本文件内的 side_by_side（最小改动且可控）

latest_web_frame = None


def _resize_to_h(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(w * (target_h / float(h)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def side_by_side(a, b, gap=10):
    h = min(a.shape[0], b.shape[0])
    a2 = _resize_to_h(a, h)
    b2 = _resize_to_h(b, h)
    gap_img = np.zeros((h, gap, 3), dtype=np.uint8)
    return np.hstack([a2, gap_img, b2])


def now_ms():
    return int(time.time() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", default=[],
                    help="repeatable. if empty, use webcam 0")
    ap.add_argument("--model", default="models/yolov8n.onnx")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no_dual", action="store_true")
    ap.add_argument("--min_infer_ms", type=int, default=0)

    # ✅ 新增：双进程必须的参数
    ap.add_argument("--port", type=int, default=5000, help="MJPEG server port")
    ap.add_argument("--name", type=str, default="cam", help="run name (cam5/cam6)")
    ap.add_argument("--runs_dir", type=str, default="runs", help="base dir for logs")
    ap.add_argument("--headless", action="store_true", help="no GUI window")
    ap.add_argument("--reid_model", type=str, default="models/osnet_x0_25.onnx",
                    help="Path to Re-ID ONNX model (OSNet)")
    ap.add_argument("--no_reid", action="store_true", help="Disable Re-ID, use color histogram")

    args = ap.parse_args()

    # 1) cfg 必须先创建
    cfg = JetsonLikeConfig()
    if args.no_dual:
        cfg.enable_dual_fov = False

    # 2) src 默认
    if len(args.src) == 0:
        args.src = ["0"]

    # 3) Web server（端口可配，避免两个进程冲突）
    mjpeg_start(host="127.0.0.1", port=args.port)
    print(f"OPEN: http://127.0.0.1:{args.port}/video")


    # 4) Sources
    sources = []
    for i, s in enumerate(args.src):
        # ✅ 用 args.name 来区分 cam5/cam6（你每个进程通常只传一个 --src）
        name = args.name if len(args.src) == 1 else f"{args.name}_cam{i}"
        prefer = (640, 480) if s.isdigit() else (None, None)
        sources.append(VideoSource(s, name=name, prefer_w=prefer[0], prefer_h=prefer[1]))

    # 5) Scheduler
    stream_weights = {}
    for src in sources:
        stream_weights[f"{src.name}:wide"] = 1.0
    if cfg.enable_dual_fov:
        stream_weights[f"{sources[0].name}:narrow"] = cfg.narrow_infer_share / max(1e-6, (1.0 - cfg.narrow_infer_share))

    sched = TokenBucketScheduler(cfg.infer_hz_total, stream_weights)

    # 6) Model providers
    providers = ["CPUExecutionProvider"]
    if args.gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    yolo = YOLOv8ONNX(args.model, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres_nms, providers=providers)

    # 6.5) Re-ID model (required for cross-camera matching)
    reid_extractor = None
    use_reid = False
    if args.no_reid:
        print("[ReID] Disabled by --no_reid flag, using color histogram")
    elif not REID_AVAILABLE:
        print("[ReID] ERROR: ReIDExtractor module not available!")
        print("[ReID] Falling back to color histogram")
    else:
        try:
            # Will auto-download if model doesn't exist
            reid_extractor = ReIDExtractor(args.reid_model, providers=providers)
            use_reid = True
            print(f"[ReID] Loaded: {args.reid_model}")
        except Exception as e:
            print(f"[ReID] ERROR: {e}")
            print("[ReID] Falling back to color histogram")

    # 7) Tracker & metrics
    trackers = {f"{src.name}:wide": IOUTracker(cfg.track_iou_match, cfg.track_max_age, cfg.track_min_hits) for src in sources}
    metrics_eng = {f"{src.name}:wide": MetricsEngine(cfg) for src in sources}

    # ✅ 日志目录区分（runs/cam5, runs/cam6）
    run_out_dir = os.path.join(args.runs_dir, args.name)
    logger = MVPLogger(out_dir=run_out_dir)

    posture = PostureEngine(cfg)

    # 8) State
    last_dets = {}
    last_infer_ms = {}
    last_frame = {}
    last_infer_t = {}

    drops = {src.name: 0 for src in sources}
    reconnects = {src.name: 0 for src in sources}

    grid_accum = None
    last_grid_t_ms = None

    # Re-ID feature cache: tid -> (feature, last_update_ms)
    reid_cache = {}
    REID_CACHE_TTL_MS = 3000  # 特征缓存 3 秒
    REID_UPDATE_INTERVAL_MS = 1000  # 每 1 秒更新一次特征

    # --- Jetson-like per-stream caps ---
    last_wide_run_ms = 0
    last_narrow_run_ms = 0

    # --- narrow trigger state ---
    narrow_trigger_until_ms = 0
    prev_raw_count = None

    # Window name must be unique per process
    win = f"MilCube-MVP ({args.name})"
    headless = args.headless

    try:
        if not headless:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while True:
            last_render = None  # 放在 while 外更好：用于暂停时继续输出
            last_web_push_ms = 0
            web_fps = 8.0  # 固定先用 8fps，够演示又省 CPU
            t_ms = now_ms()
            sched.tick(t_ms)
            if get_paused():
                if last_render is not None:
                    tnowp = now_ms()
                    if (tnowp - last_web_push_ms) >= int(1000.0 / max(1.0, web_fps)):
                        update_frame(last_render)
                        last_web_push_ms = tnowp
                    if not headless:
                        cv2.imshow(win, last_render)
                if not headless:
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27 or key == ord("q"):
                        break
                else:
                    time.sleep(0.03)
                continue

            # ---------------------------
            # 1) capture latest frames
            # ---------------------------
            for src in sources:
                frame, ts_cap, ok = src.read()
                if not ok or frame is None:
                    drops[src.name] += 1
                    reconnects[src.name] += 1
                    sleep_s = min(
                        cfg.reconnect_max_sleep,
                        cfg.reconnect_base_sleep * (1 + reconnects[src.name] * 0.1),
                    )
                    src.reconnect(sleep_s=sleep_s)
                    continue

                reconnects[src.name] = 0

                wide_id = f"{src.name}:wide"
                last_frame[wide_id] = (frame, ts_cap)

                # narrow always from wide ROI crop (single-node, no extra input)
                if cfg.enable_dual_fov and src == sources[0]:
                    narrow_id = f"{src.name}:narrow"
                    crop, roi = crop_center(frame, cfg.narrow_rel_w, cfg.narrow_rel_h)
                    last_frame[narrow_id] = (crop, ts_cap)
                    last_frame[f"{src.name}:narrow_roi"] = roi

            # ---------------------------
            # 2) Decide whether to run inference (serial, batch=1)
            # ---------------------------
            if f"{sources[0].name}:wide" not in last_frame:
                continue

            wide_id = f"{sources[0].name}:wide"
            narrow_id = f"{sources[0].name}:narrow"

            canvas = last_frame[wide_id][0].copy()
            ts_cap = last_frame[wide_id][1]
            tnow = now_ms()

            dets_w = last_dets.get(wide_id, [])
            tracks = trackers[wide_id].update(dets_w, ts_cap)

            labels, posture_events = posture.update(tracks, canvas.shape, tnow)
            posture_main = "UNK"
            if tracks:
                best = max(tracks, key=lambda tr: tr.get("hits", 0))
                posture_main = labels.get(best["id"], "UNK") if isinstance(labels, dict) else "UNK"

            m, grid, events = metrics_eng[wide_id].update(tracks, canvas.shape, ts_cap, tnow)
            events = events + posture_events
            raw_count = int(m.get("count", 0))

            # ----- narrow trigger logic -----
            trigger = False
            if cfg.narrow_trigger_on_event and events:
                trigger = True

            if cfg.narrow_trigger_on_count_change:
                if prev_raw_count is None:
                    prev_raw_count = raw_count
                if abs(raw_count - prev_raw_count) >= cfg.narrow_trigger_count_delta:
                    trigger = True
                prev_raw_count = raw_count

            if trigger:
                narrow_trigger_until_ms = max(narrow_trigger_until_ms, tnow + cfg.narrow_trigger_window_ms)

            allow_narrow_window = (cfg.enable_dual_fov and (tnow <= narrow_trigger_until_ms))

            ran_any = False

            # (A) narrow
            if allow_narrow_window and (narrow_id in last_frame):
                if (tnow - last_narrow_run_ms) >= int(1000.0 / max(0.1, cfg.narrow_fps_cap)):
                    if (tnow - last_narrow_run_ms) >= cfg.narrow_trigger_min_interval_ms:
                        if sched.can_run(narrow_id, t_ms):
                            frame_n, ts_n = last_frame[narrow_id]
                            dets_n, infer_ms_n, _ = yolo.infer_person(frame_n, cfg.infer_w, cfg.infer_h)
                            last_dets[narrow_id] = dets_n
                            last_infer_ms[narrow_id] = infer_ms_n
                            last_infer_t[narrow_id] = now_ms()
                            sched.consume(narrow_id, t_ms, min_infer_ms=args.min_infer_ms)
                            last_narrow_run_ms = tnow
                            ran_any = True

            # (B) wide
            if not ran_any:
                if (tnow - last_wide_run_ms) >= int(1000.0 / max(0.1, cfg.wide_fps_cap)):
                    if sched.can_run(wide_id, t_ms):
                        frame_w, ts_w = last_frame[wide_id]
                        dets_w2, infer_ms_w2, _ = yolo.infer_person(frame_w, cfg.infer_w, cfg.infer_h)
                        last_dets[wide_id] = dets_w2
                        last_infer_ms[wide_id] = infer_ms_w2
                        last_infer_t[wide_id] = now_ms()
                        sched.consume(wide_id, t_ms, min_infer_ms=args.min_infer_ms)
                        last_wide_run_ms = tnow
                        ran_any = True

            # ---------------------------
            # 3) density accumulation (wide only)
            # ---------------------------
            decay_per_sec = 0.85
            if grid_accum is None or grid_accum.shape != grid.shape:
                grid_accum = grid.astype(np.float32)
                last_grid_t_ms = tnow
            else:
                dt = max(0.0, (tnow - (last_grid_t_ms or tnow)) / 1000.0)
                last_grid_t_ms = tnow
                decay = float(decay_per_sec ** dt)
                grid_accum = grid_accum * decay + grid.astype(np.float32)

            # 保存干净的帧用于裁剪（在任何叠加之前）
            raw_canvas = canvas.copy()

            canvas = draw_grid_overlay(canvas, grid_accum, alpha=0.35)

            # ---------------------------
            # 4) narrow evidence (2nd-order) ONLY
            # ---------------------------
            roi = last_frame.get(f"{sources[0].name}:narrow_roi", None)

            confirm_flags = {}
            narrow_has_person = False

            confirm_window_ms = cfg.narrow_trigger_window_ms
            if cfg.enable_dual_fov and (narrow_id in last_dets) and (narrow_id in last_infer_t):
                narrow_recent = (tnow - last_infer_t[narrow_id]) <= confirm_window_ms
                narrow_has_person = narrow_recent and (len(last_dets.get(narrow_id, [])) > 0)

            for tr in tracks:
                tid = tr["id"]
                x1, y1, x2, y2 = tr["xyxy"]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                in_roi = False
                if roi is not None:
                    rx1, ry1, rx2, ry2 = roi
                    in_roi = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

                if roi is not None and in_roi:
                    confirm_flags[tid] = bool(narrow_has_person)
                else:
                    confirm_flags[tid] = True

            # ---------------------------
            # 5) confirmed_count: policy-based (not hard drop)
            # ---------------------------
            confirmed_count = 0
            effective_count = 0.0

            for tr in tracks:
                if tr.get("hits", 0) < cfg.track_min_hits:
                    continue

                tid = tr["id"]
                in_confirm = confirm_flags.get(tid, True)

                if cfg.confirm_policy == 0:
                    confirmed_count += 1
                    effective_count += 1.0
                elif cfg.confirm_policy == 1:
                    confirmed_count += 1
                    effective_count += 1.0
                elif cfg.confirm_policy == 2:
                    w = cfg.roi_confirm_weight if in_confirm else cfg.roi_unconfirmed_weight
                    effective_count += float(w)
                    if in_confirm:
                        confirmed_count += 1
                else:
                    if in_confirm:
                        confirmed_count += 1
                        effective_count += 1.0

            # ---------------------------
            # 6) visualization (wide only + ROI + tracks)
            # ---------------------------
            canvas = draw_tracks(
                canvas,
                tracks,
                min_hits=cfg.track_min_hits,
                labels=labels,
                confirm_flags=confirm_flags,
            )
            video_canvas = canvas.copy()
            if cfg.enable_dual_fov and roi is not None:
                rx1, ry1, rx2, ry2 = roi
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)

            infer_ms_w = last_infer_ms.get(wide_id, -1)
            infer_ms_n = last_infer_ms.get(narrow_id, -1)
            latency_ms = tnow - ts_cap

            # ---------------------------
            # 7) show narrow view OUTSIDE (side-by-side)
            # ---------------------------
            if cfg.enable_dual_fov and (narrow_id in last_frame):
                narrow_frame, _ = last_frame[narrow_id]
                narrow_vis = narrow_frame.copy()

                for d in last_dets.get(narrow_id, []):
                    x1, y1, x2, y2 = map(int, d["xyxy"])
                    cv2.rectangle(narrow_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                final_canvas = side_by_side(canvas, narrow_vis, gap=10)
            else:
                final_canvas = canvas
            web_canvas = final_canvas.copy()
            # ---------------------------
            # KPI computation (for panel)
            # ---------------------------
            # congestion (single-node)
            try:
                gw, gh = cfg.grid_w, cfg.grid_h
                congest_cells = int((grid_accum >= 2.0).sum()) if grid_accum is not None else 0
                congest_rate = (congest_cells / float(gw * gh)) if (gw * gh) > 0 else 0.0
            except Exception:
                congest_rate = 0.0

            # avg distance (pixel median)
            avg_dist_px = 0
            pts = []
            for tr in tracks:
                if tr.get("hits", 0) < cfg.track_min_hits:
                    continue
                x1, y1, x2, y2 = tr["xyxy"]
                pts.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5))

            if len(pts) >= 2:
                dists = []
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dx = pts[i][0] - pts[j][0]
                        dy = pts[i][1] - pts[j][1]
                        dists.append((dx * dx + dy * dy) ** 0.5)
                if dists:
                    dists.sort()
                    avg_dist_px = int(dists[len(dists) // 2])

            # last event (wide only)
            last_event_str = "--"
            if events:
                e = events[-1]
                last_event_str = f"{e.get('type', '?')}/tid={e.get('track_id', '?')}"

            # build cards (ONLY ONCE)
            cards = [
                {
                    "title": "COUNT",
                    "value": str(int(confirmed_count)),
                    "subtitle": f"raw={int(raw_count)}  eff={effective_count:.2f}"
                },
                {
                    "title": "LATENCY",
                    "value": str(int(latency_ms)),
                    "unit": "ms",
                    "subtitle": f"posture={posture_main}"
                },
                {
                    "title": "INFER (W/N)",
                    "value": f"{infer_ms_w:.0f}/{infer_ms_n:.0f}",
                    "unit": "ms",
                    "subtitle": "serial budget"
                },
                {
                    "title": "LAST EVENT",
                    "value": "",
                    "subtitle": last_event_str
                },
            ]

            # ---------------------------
            # UI layout: video + bottom panel (outside video)
            # ---------------------------
            video = final_canvas
            vh, vw = video.shape[:2]

            panel_h = 120
            ui = np.zeros((vh + panel_h, vw, 3), dtype=np.uint8)
            ui[:] = (245, 245, 245)

            ui[0:vh, 0:vw] = video

            panel = ui[vh:vh + panel_h, 0:vw]
            panel = draw_kpi_cards_on_panel(panel, cards, card_h=92)
            ui[vh:vh + panel_h, 0:vw] = panel

            final_canvas = ui

            # ---------------------------
            # 8) logging
            # ---------------------------
            m2 = dict(m)
            m2.update({
                "raw_count": raw_count,
                "confirmed_count": int(confirmed_count),
                "effective_count": float(effective_count),
                "infer_ms_wide": float(infer_ms_w) if infer_ms_w != -1 else -1,
                "infer_ms_narrow": float(infer_ms_n) if infer_ms_n != -1 else -1,
                "latency_ms": int(latency_ms),
                "drops": drops.get(sources[0].name, 0),
                "reconnect": reconnects.get(sources[0].name, 0),
                "narrow_trigger_active": int(allow_narrow_window),
                "narrow_has_person": int(narrow_has_person),
                "congest_rate": float(congest_rate),
                "avg_dist_px": int(avg_dist_px),
            })

            logger.log_metrics(m2)
            logger.log_events(events)

            # ---- heat grid payload (0..1) ----
            grid_norm = None
            try:
                if grid_accum is not None:
                    g = grid_accum.astype(np.float32)
                    gmax = float(g.max()) if float(g.max()) > 1e-6 else 1.0
                    grid_norm = (g / gmax).clip(0, 1).tolist()
            except Exception:
                grid_norm = None

            # ---- track centers + Re-ID features for cross-cam matching ----
            tracks_xy = []
            try:
                for tr in tracks:
                    if tr.get("hits", 0) < cfg.track_min_hits:
                        continue

                    tid = int(tr["id"])
                    x1, y1, x2, y2 = tr["xyxy"]
                    cx = float((x1 + x2) * 0.5)
                    cy = float((y1 + y2) * 0.5)

                    h, w = raw_canvas.shape[:2]
                    px1, py1 = max(0, int(x1)), max(0, int(y1))
                    px2, py2 = min(w, int(x2)), min(h, int(y2))

                    if use_reid and reid_extractor:
                        # Re-ID with caching
                        cached = reid_cache.get(tid)
                        feat = None

                        if cached is not None and (tnow - cached[1]) <= REID_UPDATE_INTERVAL_MS:
                            feat = cached[0]  # 使用缓存
                        elif px2 > px1 + 10 and py2 > py1 + 20:
                            # 需要提取新特征
                            crop = raw_canvas[py1:py2, px1:px2]
                            feat = reid_extractor.extract(crop)
                            reid_cache[tid] = (feat, tnow)

                        if feat is not None:
                            tracks_xy.append({
                                "tid": tid, "cx": cx, "cy": cy,
                                "reid_feat": [round(float(v), 6) for v in feat],
                                "feature_type": "reid"
                            })
                    else:
                        # Histogram fallback
                        if px2 > px1 + 10 and py2 > py1 + 20:
                            crop = raw_canvas[py1:py2, px1:px2]
                            hc, wc = crop.shape[:2]

                            def hist_region(region):
                                lab = cv2.cvtColor(region, cv2.COLOR_BGR2Lab)
                                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                                parts = []
                                for ch in range(3):
                                    h_lab = cv2.calcHist([lab], [ch], None, [32], [0, 256])
                                    parts.append(cv2.normalize(h_lab, h_lab).flatten())
                                h_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])
                                h_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
                                parts.extend([cv2.normalize(h_h, h_h).flatten(),
                                             cv2.normalize(h_s, h_s).flatten()])
                                return np.concatenate(parts)

                            hist_parts = []
                            upper = crop[int(hc*0.1):int(hc*0.45), int(wc*0.1):int(wc*0.9)]
                            lower = crop[int(hc*0.5):int(hc*0.9), int(wc*0.1):int(wc*0.9)]
                            if upper.size > 100: hist_parts.append(hist_region(upper))
                            if lower.size > 100: hist_parts.append(hist_region(lower))

                            color_hist = None
                            if hist_parts:
                                combined = np.concatenate(hist_parts)
                                total = combined.sum()
                                if total > 1e-6: combined = combined / total
                                color_hist = [round(float(v), 4) for v in combined]

                            tracks_xy.append({
                                "tid": tid, "cx": cx, "cy": cy,
                                "color_hist": color_hist,
                                "feature_type": "histogram"
                            })

                # Clean expired cache
                expired = [k for k, v in reid_cache.items() if (tnow - v[1]) > REID_CACHE_TTL_MS]
                for k in expired:
                    del reid_cache[k]
            except Exception as e:
                tracks_xy = []
            tracks_box = []
            try:
                for tr in tracks:
                    if tr.get("hits", 0) < cfg.track_min_hits:
                        continue
                    tid = int(tr["id"])
                    x1, y1, x2, y2 = map(float, tr["xyxy"])
                    tracks_box.append({
                        "tid": tid,
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2
                    })
            except Exception:
                tracks_box = []
            # ✅ MUST create m2_web BEFORE update
            m2_web = dict(m2)
            m2_web.update({
                "last_event": last_event_str,
                "paused": int(get_paused()),
                "grid_norm": grid_norm,
                "grid_w": int(cfg.grid_w),
                "grid_h": int(cfg.grid_h),
                "name": args.name,
                "ts_ms": int(tnow),

                # for matcher
                "tracks_xy": tracks_xy,
                "tracks_box": tracks_box,  # ✅ 新增这一行

                "frame_w": int(video_canvas.shape[1]),
                "frame_h": int(video_canvas.shape[0]),

            })
            update_metrics(m2_web)
            # ---- track boxes for pair-only visualization (UI only) ----


            # ---------------------------
            # 9) render / web output (throttled)
            # ---------------------------
            if not headless:
                cv2.imshow(win, final_canvas)
            last_render = final_canvas

            tnow2 = now_ms()
            if (tnow2 - last_web_push_ms) >= int(1000.0 / max(1.0, web_fps)):
                update_frame(web_canvas, raw_canvas)  # 第二个参数用于裁剪（干净无叠加）
                last_web_push_ms = tnow2

            if not headless:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    break
            else:
                time.sleep(0.001)  # headless 模式下让出 CPU


    finally:
        logger.close()
        for s in sources:
            s.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
