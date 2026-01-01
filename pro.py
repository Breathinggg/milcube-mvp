import cv2

input_video = "cam5_6min.mp4"
output_video = "cam5.mp4"

cap = cv2.VideoCapture(input_video)

orig_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = 10
frame_interval = int(round(orig_fps / target_fps))

width, height = 640, 360
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        frame_resized = cv2.resize(frame, (width, height))
        out.write(frame_resized)

    frame_id += 1

cap.release()
out.release()
