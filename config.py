from dataclasses import dataclass

@dataclass
class JetsonLikeConfig:
    # 强制推理输入尺寸（模拟Jetson常用低分辨率）
    infer_w: int = 320
    infer_h: int = 320

    # 渲染上限（看起来像边缘设备）
    render_fps_cap: float = 12.0
    infer_hz_total: float = 8.0

    # 单路时建议：12Hz；双路：各6Hz；三路：各4Hz
    # 这里由 scheduler 用 token bucket 控制

    # 人检测阈值
    conf_thres: float = 0.45     # 提高置信度，减少椅子等误检
    iou_thres_nms: float = 0.50

    # 追踪参数（简易IOU tracker）
    track_iou_match: float = 0.3
    track_max_age: int = 40      # 允许丢检帧数（越大越"粘"）
    track_min_hits: int = 3      # hits>=3 才计入人数/密度（抑制误报）

    # ROI密度网格
    grid_w: int = 16
    grid_h: int = 9

    # 异常检测频率（启发式，低频）
    anomaly_hz: float = 10.0
    loiter_seconds: float = 2.0  # 滞留阈值

    # “跌倒/蹲伏”粗判阈值（非常粗）
    fall_aspect_drop: float = 0.55   # 高宽比下降比例
    fall_speed_spike: float = 25.0   # 像素/秒速度突变阈值（按你的画面调整）

    # 断流重连
    reconnect_base_sleep: float = 1.0
    reconnect_max_sleep: float = 5.0

    # 软件双视场（Narrow crop）
    enable_dual_fov: bool = True
    # Narrow ROI：相对宽高（0~1），居中裁剪
    narrow_rel_w: float = 0.45
    narrow_rel_h: float = 0.45
    narrow_infer_share: float = 0.35  # 推理预算分给Narrow的比例（剩下给Wide）
    # --- posture heuristic (no pose model) ---
    posture_enable: bool = True
    posture_update_hz: float = 6.0   # 状态更新频率（不必每帧）
    posture_debounce: int = 5        # 连续满足多少次才切换

    stand_h_ratio: float = 0.3
    stand_aspect_min: float = 1.10

    sit_h_ratio_min: float = 0.12
    sit_h_ratio_max: float = 0.28
    sit_aspect_min: float = 0.60
    sit_aspect_max: float = 1.20
    sit_speed_max: float = 30.0

    fall_aspect_max: float = 0.75
    fall_speed_min: float = 80.0
    fall_hold_seconds: float = 2.0  # FALL? 状态保持时间（防抖）
    jetson_like: bool = True
    infer_w: int = 320
    infer_h: int = 320

    # wide 渲染/推理 cap（10~15fps）
    wide_fps_cap: float = 12.0

    # narrow 推理 cap（1~5fps），且触发后允许短时间加速
    narrow_fps_cap: float = 2.0
    narrow_trigger_window_ms: int = 1200   # 触发后这段时间内允许跑 narrow
    narrow_trigger_min_interval_ms: int = 400  # 防止触发期疯狂跑（<=5fps）

    # scheduler 总预算仍然用你原来的 infer_hz_total（例如 8Hz）
    # 但这里再加硬上限：单 loop 禁止并行（你代码已有 break）

    # ===== confirmation strategy (not hard drop) =====
    # 0: none（不影响计数）
    # 1: soft（只标记/降低置信，不减少count）
    # 2: weighted（ROI内未confirm给较低权重，得到“effective_count”）
    # 3: strict（ROI内未confirm不计入confirmed_count）
    confirm_policy: int = 2

    roi_unconfirmed_weight: float = 0.5   # policy=2 时使用
    roi_confirm_weight: float = 1.0

    # 触发 narrow 的条件（最小实现：有候选事件 或 count变化）
    narrow_trigger_on_event: bool = True
    narrow_trigger_on_count_change: bool = True
    narrow_trigger_count_delta: int = 1   # raw_count变化>=1 触发