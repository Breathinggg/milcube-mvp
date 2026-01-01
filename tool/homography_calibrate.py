"""
Homography 标定工具
在两个视频中点击对应点，计算变换矩阵

使用方法：
  python tool/homography_calibrate.py --cam5 cam5.mp4 --cam6 cam6.mp4

操作：
  - 在左边(cam5)点击一个点，然后在右边(cam6)点击对应的点
  - 至少需要 4 对点
  - 按 'c' 计算并保存 Homography
  - 按 'r' 重置所有点
  - 按 'u' 撤销最后一个点
  - 按 'q' 退出
"""

import argparse
import cv2
import numpy as np
import json
import os

class HomographyCalibrator:
    def __init__(self, cam5_path, cam6_path, output_path="homography_config.json"):
        self.cam5_path = cam5_path
        self.cam6_path = cam6_path
        self.output_path = output_path

        # 读取第一帧
        cap5 = cv2.VideoCapture(cam5_path)
        ret5, self.frame5 = cap5.read()
        cap5.release()

        cap6 = cv2.VideoCapture(cam6_path)
        ret6, self.frame6 = cap6.read()
        cap6.release()

        if not ret5 or not ret6:
            raise ValueError("无法读取视频文件")

        # 调整大小以便显示
        self.display_h = 540
        self.scale5 = self.display_h / self.frame5.shape[0]
        self.scale6 = self.display_h / self.frame6.shape[0]

        self.frame5_display = cv2.resize(self.frame5, None, fx=self.scale5, fy=self.scale5)
        self.frame6_display = cv2.resize(self.frame6, None, fx=self.scale6, fy=self.scale6)

        # 存储点（原始坐标）
        self.points5 = []
        self.points6 = []

        # 当前状态：等待 cam5 点击 或 cam6 点击
        self.waiting_for = "cam5"

        # 窗口和画布
        self.window_name = "Homography Calibration"
        self.gap = 20

    def draw(self):
        """绘制当前状态"""
        f5 = self.frame5_display.copy()
        f6 = self.frame6_display.copy()

        # 绘制已有的点
        colors = [
            (0, 255, 0),    # 绿
            (255, 0, 0),    # 蓝
            (0, 0, 255),    # 红
            (255, 255, 0),  # 青
            (255, 0, 255),  # 洋红
            (0, 255, 255),  # 黄
            (128, 255, 0),
            (255, 128, 0),
        ]

        for i, (p5, p6) in enumerate(zip(self.points5, self.points6)):
            color = colors[i % len(colors)]
            # cam5 上的点
            dp5 = (int(p5[0] * self.scale5), int(p5[1] * self.scale5))
            cv2.circle(f5, dp5, 8, color, -1)
            cv2.putText(f5, str(i+1), (dp5[0]+10, dp5[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # cam6 上的点
            dp6 = (int(p6[0] * self.scale6), int(p6[1] * self.scale6))
            cv2.circle(f6, dp6, 8, color, -1)
            cv2.putText(f6, str(i+1), (dp6[0]+10, dp6[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 如果有未配对的 cam5 点
        if len(self.points5) > len(self.points6):
            p5 = self.points5[-1]
            dp5 = (int(p5[0] * self.scale5), int(p5[1] * self.scale5))
            cv2.circle(f5, dp5, 8, (0, 165, 255), -1)  # 橙色表示未配对
            cv2.circle(f5, dp5, 12, (0, 165, 255), 2)

        # 状态提示
        status = f"Waiting for: {self.waiting_for.upper()} | Points: {min(len(self.points5), len(self.points6))}/4+"
        cv2.putText(f5, "CAM5", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(f6, "CAM6", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 拼接
        gap_img = np.zeros((f5.shape[0], self.gap, 3), dtype=np.uint8)
        canvas = np.hstack([f5, gap_img, f6])

        # 底部状态栏
        bar_h = 50
        bar = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bar, "[C]calc [R]reset [U]undo [Q]quit", (canvas.shape[1]-400, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        canvas = np.vstack([canvas, bar])

        return canvas

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 判断点击在哪个区域
        f5_w = self.frame5_display.shape[1]
        f6_w = self.frame6_display.shape[1]

        if x < f5_w:
            # 点击在 cam5 区域
            if self.waiting_for == "cam5":
                # 转换为原始坐标
                orig_x = x / self.scale5
                orig_y = y / self.scale5
                self.points5.append([orig_x, orig_y])
                self.waiting_for = "cam6"
                print(f"CAM5 点 {len(self.points5)}: ({orig_x:.1f}, {orig_y:.1f})")
        elif x > f5_w + self.gap:
            # 点击在 cam6 区域
            if self.waiting_for == "cam6":
                local_x = x - f5_w - self.gap
                orig_x = local_x / self.scale6
                orig_y = y / self.scale6
                self.points6.append([orig_x, orig_y])
                self.waiting_for = "cam5"
                print(f"CAM6 点 {len(self.points6)}: ({orig_x:.1f}, {orig_y:.1f})")

    def compute_homography(self):
        """计算 Homography 矩阵"""
        n = min(len(self.points5), len(self.points6))
        if n < 4:
            print(f"至少需要 4 对点，当前只有 {n} 对")
            return None

        pts5 = np.array(self.points5[:n], dtype=np.float32)
        pts6 = np.array(self.points6[:n], dtype=np.float32)

        # 计算 cam5 -> cam6 的 Homography
        H_5to6, mask = cv2.findHomography(pts5, pts6, cv2.RANSAC, 5.0)

        # 计算 cam6 -> cam5 的 Homography
        H_6to5, mask = cv2.findHomography(pts6, pts5, cv2.RANSAC, 5.0)

        return H_5to6, H_6to5

    def save_config(self, H_5to6, H_6to5):
        """保存配置"""
        config = {
            "cam5_to_cam6": H_5to6.tolist(),
            "cam6_to_cam5": H_6to5.tolist(),
            "points_cam5": self.points5[:min(len(self.points5), len(self.points6))],
            "points_cam6": self.points6[:min(len(self.points5), len(self.points6))],
            "cam5_shape": [self.frame5.shape[1], self.frame5.shape[0]],
            "cam6_shape": [self.frame6.shape[1], self.frame6.shape[0]],
        }

        with open(self.output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n配置已保存到: {self.output_path}")
        print(f"H (cam5->cam6):\n{H_5to6}")

    def run(self):
        """运行标定工具"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n=== Homography 标定工具 ===")
        print("1. 在 CAM5 中点击一个点")
        print("2. 在 CAM6 中点击对应的同一个位置")
        print("3. 重复至少 4 次")
        print("4. 按 C 计算并保存\n")

        while True:
            canvas = self.draw()
            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                result = self.compute_homography()
                if result:
                    H_5to6, H_6to5 = result
                    self.save_config(H_5to6, H_6to5)
                    print("\n按任意键退出...")
                    cv2.waitKey(0)
                    break
            elif key == ord('r'):
                self.points5 = []
                self.points6 = []
                self.waiting_for = "cam5"
                print("已重置所有点")
            elif key == ord('u'):
                if self.waiting_for == "cam6" and self.points5:
                    self.points5.pop()
                    self.waiting_for = "cam5"
                    print("撤销 cam5 点")
                elif self.waiting_for == "cam5" and self.points6:
                    self.points6.pop()
                    self.waiting_for = "cam6"
                    print("撤销 cam6 点")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Homography 标定工具")
    parser.add_argument("--cam5", default="cam5.mp4", help="cam5 视频路径")
    parser.add_argument("--cam6", default="cam6.mp4", help="cam6 视频路径")
    parser.add_argument("--output", default="homography_config.json", help="输出配置文件路径")
    args = parser.parse_args()

    calibrator = HomographyCalibrator(args.cam5, args.cam6, args.output)
    calibrator.run()


if __name__ == "__main__":
    main()
