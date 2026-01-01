import cv2, numpy as np, time, os, sys

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "cam5.mp4"

backends = [
    ("FFMPEG", cv2.CAP_FFMPEG),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]

for name, api in backends:
    cap = cv2.VideoCapture(VIDEO, api)
    print(f"\n=== backend {name} ({api}) ===")
    print("opened:", cap.isOpened())

    ok, frame = cap.read()
    print("first_read:", ok, "frame_none:", frame is None)
    if ok and frame is not None:
        print("shape:", frame.shape, "dtype:", frame.dtype,
              "min/max/mean:", int(frame.min()), int(frame.max()), float(frame.mean()))
        # 保存第一帧到磁盘，肉眼确认是否黑
        out = f"__first_{name}.jpg"
        cv2.imwrite(out, frame)
        print("saved:", out)

        # 显示 2 秒
        cv2.imshow(f"smoke_{name}", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    cap.release()

print("\nDONE")
