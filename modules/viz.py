import cv2
import numpy as np
import time
def draw_kpi_cards_on_panel(panel, cards, *, margin=16, card_h=92, gap=14):
    """
    panel: 一张纯 UI 面板图（比如白底或黑底），只在 panel 内画卡片
    cards: [{"title":..., "value":..., "unit":..., "subtitle":...}]
    """
    H, W = panel.shape[:2]
    n = max(0, len(cards))
    if n == 0:
        return panel

    n_show = min(n, 4)
    total_gap = gap * (n_show - 1)
    card_w = int((W - 2 * margin - total_gap) / n_show)

    # 卡片垂直居中放在 panel 内
    y1 = int((H - card_h) / 2)
    y2 = y1 + card_h

    out = panel.copy()

    for i in range(n_show):
        c = cards[i]
        x1 = margin + i * (card_w + gap)
        x2 = x1 + card_w

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(out, (x1, y1), (x2, y2), (220, 220, 220), 2)

        title = c.get("title", "")
        value = c.get("value", "")
        unit = c.get("unit", "")
        subtitle = c.get("subtitle", "")

        cv2.putText(out, title, (x1 + 12, y1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 2, cv2.LINE_AA)

        big = f"{value}{unit}"
        cv2.putText(out, big, (x1 + 12, y1 + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, (20, 20, 20), 3, cv2.LINE_AA)

        if subtitle:
            cv2.putText(out, subtitle, (x1 + 12, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    return out


def draw_grid_overlay(img, grid, alpha=0.35, vmax=None, colormap=cv2.COLORMAP_TURBO):
    """
    彩色热力图叠加：
    - grid: (gh, gw) 可以是 int/float（例如 grid_accum）
    - vmax: 手动指定最大热度（更稳定）；None 则用 grid.max()
    - alpha: 叠加透明度
    """
    if grid is None:
        return img

    h, w = img.shape[:2]
    g = grid.astype(np.float32)

    gmax = float(g.max())
    if vmax is not None:
        gmax = float(max(1e-6, vmax))
    if gmax <= 1e-6:
        return img  # 没热度就不画

    # 归一化到 0~255
    g_norm = np.clip(g / gmax, 0.0, 1.0)
    g_u8 = (g_norm * 255.0).astype(np.uint8)

    # 放大到整张图
    heat_small = cv2.resize(g_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    # 颜色映射
    heat_color = cv2.applyColorMap(heat_small, colormap)

    # 让低热度更“隐形”：可选（避免背景被染色）
    # 例如：只有 >= 5% 才开始可见
    mask = (heat_small >= 13).astype(np.uint8)  # 13/255≈5%
    if mask.sum() == 0:
        return img
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)

    out = img.copy()
    out = np.where(mask3 > 0,
                   cv2.addWeighted(img, 1 - alpha, heat_color, alpha, 0),
                   img)
    return out

def side_by_side(img1, img2, gap=10, bg=(0,0,0)):
    h = max(img1.shape[0], img2.shape[0])
    def pad_to_h(img):
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        top = pad // 2
        bottom = pad - top
        return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=bg)

    a = pad_to_h(img1)
    b = pad_to_h(img2)

    gap_col = np.full((h, gap, 3), bg, dtype=np.uint8)
    return np.concatenate([a, gap_col, b], axis=1)

def draw_tracks(img, tracks, min_hits=3, labels=None, confirm_flags=None):
    if labels is None:
        labels = {}
    if confirm_flags is None:
        confirm_flags = {}

    # ---- make overlays readable under MJPEG + browser scaling ----
    h, w = img.shape[:2]
    th = max(2, int(min(h, w) / 240))          # line thickness (720p ~3-4)
    fs = max(0.7, min(1.2, min(h, w) / 650))   # font scale
    text_th = max(2, th)                        # text thickness

    for t in tracks:
        if t["hits"] < min_hits:
            continue

        tid = t["id"]
        x1, y1, x2, y2 = map(int, t["xyxy"])

        confirmed = bool(confirm_flags.get(tid, True))
        color = (0, 255, 0) if confirmed else (160, 160, 160)  # 绿 / 灰

        # bbox (AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th, lineType=cv2.LINE_AA)

        # label with outline (black shadow -> white text)
        lab = labels.get(tid, "")
        txt = f"id:{tid}"
        if lab:
            txt += f" {lab}"
        org = (x1, max(0, y1 - 8))

        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    text_th + 2, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    text_th, cv2.LINE_AA)

        # trajectory (AA, slightly thinner than bbox)
        pts = [(int(x), int(y)) for (_, x, y) in t["traj"][-20:]]
        traj_th = max(1, th - 1)
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], (0, 255, 255), traj_th, lineType=cv2.LINE_AA)

    return img





def draw_status(img, lines):
    x, y = 10, 20
    for s in lines:
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 22
    return img

def crop_center(frame, rel_w, rel_h):
    H, W = frame.shape[:2]
    cw = int(W * rel_w)
    ch = int(H * rel_h)
    x1 = max(0, (W - cw)//2)
    y1 = max(0, (H - ch)//2)
    x2 = min(W, x1 + cw)
    y2 = min(H, y1 + ch)
    return frame[y1:y2, x1:x2], (x1,y1,x2,y2)

def paste_small_view(canvas, small, position="bottom_right", scale=0.35, margin=12):
    """
    position: 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right'
    """
    H, W = canvas.shape[:2]
    h, w = small.shape[:2]

    sw = int(w * scale)
    sh = int(h * scale)
    small_r = cv2.resize(small, (sw, sh))

    if position == "top_left":
        x1, y1 = margin, margin
    elif position == "top_right":
        x1, y1 = W - sw - margin, margin
    elif position == "bottom_left":
        x1, y1 = margin, H - sh - margin
    else:  # bottom_right
        x1, y1 = W - sw - margin, H - sh - margin

    x2, y2 = x1 + sw, y1 + sh

    # 防止越界
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    canvas[y1:y2, x1:x2] = small_r[0:(y2-y1), 0:(x2-x1)]
    return canvas

