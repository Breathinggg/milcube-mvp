import numpy as np
import cv2
import onnxruntime as ort
import time

def letterbox(img, new_shape=(640, 360), color=(114,114,114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / w, new_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = new_w - nw
    pad_h = new_h - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    meta = {"scale": scale, "pad": (left, top), "orig_shape": (h, w), "new_shape": (new_h, new_w)}
    return img_padded, meta

def nms_xyxy(boxes, scores, iou_thres=0.5):
    # boxes: Nx4, scores: N
    if len(boxes) == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        iou = iou_xyxy(boxes[i], boxes[rest])
        idxs = rest[iou <= iou_thres]
    return keep

def iou_xyxy(box, boxes):
    # box: (4,), boxes: (M,4)
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    union = np.maximum(1e-6, area1 + area2 - inter)
    return inter / union

class YOLOv8ONNX:
    """
    兼容常见 yolov8 onnx 输出：
    - 有的输出 shape: (1, N, 84) (xywh + obj + 80cls)
    - 有的输出 shape: (1, 84, N) 需要转置
    本实现只取 person 类（COCO: 0）
    """
    def __init__(self, onnx_path: str, conf_thres=0.35, iou_thres=0.5, providers=None):
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        if providers is None:
            providers = ort.get_available_providers()
        print("ORT available:", ort.get_available_providers())
        print("ORT requested:", providers)
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

        print("ORT session providers:", self.sess.get_providers())
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        print("ORT providers:", self.sess.get_providers())

    def infer_person(self, frame_bgr, infer_w, infer_h):
        t0 = time.time()
        img, meta = letterbox(frame_bgr, (infer_w, infer_h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # 1x3xHxW

        out = self.sess.run([self.out_name], {self.in_name: inp})[0]

        # normalize to (1, N, C)
        if out.ndim == 3 and out.shape[1] < out.shape[2]:
            out = np.transpose(out, (0, 2, 1))
        pred = out[0]  # (N, C)

        C = pred.shape[1]
        if C not in (84, 85):
            # 兜底：打印一次方便定位
            print("[YOLO ONNX] Unexpected output shape:", out.shape)
            return [], int((time.time() - t0) * 1000), meta

        xywh = pred[:, 0:4]

        # ✅ 兼容两种导出：84(无obj) / 85(有obj)
        if C == 85:
            obj = pred[:, 4:5]
            cls = pred[:, 5:]
            person_scores = (obj * cls[:, [0]]).squeeze(1)
        else:  # C == 84
            cls = pred[:, 4:]
            person_scores = cls[:, 0]  # person=0

        # Debug：你第一次跑建议先把阈值放低一点确认出框
        keep = person_scores >= self.conf_thres
        if not np.any(keep):
            # 打印一下最大分数，确认不是一直为0
            # （只打印很少，避免刷屏）
            # 你也可以注释掉
            mx = float(person_scores.max()) if person_scores.size else -1.0
            # print("[YOLO ONNX] no det, max person score:", mx)
            return [], int((time.time() - t0) * 1000), meta

        xywh = xywh[keep]
        scores = person_scores[keep]

        # xywh (center) -> xyxy in letterbox space
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep_idx = nms_xyxy(boxes, scores, self.iou_thres)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        # map back to original frame coords
        scale = meta["scale"]
        pad_x, pad_y = meta["pad"]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale

        H, W = meta["orig_shape"]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

        dets = [{"xyxy": b.tolist(), "conf": float(s)} for b, s in zip(boxes, scores)]
        return dets, int((time.time() - t0) * 1000), meta
