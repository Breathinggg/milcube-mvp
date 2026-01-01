"""
从视频生成 demo GIF
用法: python make_demo_gif.py
"""
import cv2
import subprocess
import os

# 配置
VIDEO_SRC = "6p-c1.avi"  # 源视频
OUTPUT_GIF = "docs/demo.gif"
START_SEC = 5            # 从第几秒开始
DURATION_SEC = 8         # 录多少秒
FPS = 10                 # GIF 帧率
WIDTH = 480              # GIF 宽度 (高度自动)

def main():
    os.makedirs("docs", exist_ok=True)

    # 先用 ffmpeg 生成（如果有的话）
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(START_SEC),
            "-t", str(DURATION_SEC),
            "-i", VIDEO_SRC,
            "-vf", f"fps={FPS},scale={WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
            OUTPUT_GIF
        ]
        subprocess.run(cmd, check=True)
        print(f"Done! Saved to {OUTPUT_GIF}")
        return
    except FileNotFoundError:
        print("ffmpeg not found, using OpenCV fallback...")

    # OpenCV fallback (质量稍差)
    cap = cv2.VideoCapture(VIDEO_SRC)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25

    # 跳到起始位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_SEC * fps_src))

    frames = []
    frame_interval = int(fps_src / FPS)
    total_frames = int(DURATION_SEC * fps_src)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            h, w = frame.shape[:2]
            new_h = int(h * WIDTH / w)
            frame = cv2.resize(frame, (WIDTH, new_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if not frames:
        print("No frames captured!")
        return

    # 用 imageio 保存 GIF
    try:
        import imageio
        imageio.mimsave(OUTPUT_GIF, frames, fps=FPS, loop=0)
        print(f"Done! Saved to {OUTPUT_GIF}")
    except ImportError:
        print("请安装 imageio: pip install imageio")
        print("或者安装 ffmpeg: https://ffmpeg.org/download.html")

if __name__ == "__main__":
    main()
