import cv2
import os

def extract_frames(video_path, output_dir, fps):
    if not os.path.exists(video_path):
        print("Video file not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    if fps > frame_rate:
        print("Desired FPS is higher than the video frame rate.")
        return

    interval = max(1, frame_rate // fps)

    print(f"Total frames: {frame_count}, Frame Rate: {frame_rate}, Interval: {interval}")

    current_frame = 0
    while current_frame < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % interval == 0:
            filename = f"Ambar_blue_{current_frame:04d}.jpg" # change name of frame here
            success = cv2.imwrite(os.path.join(output_dir, filename), frame)
            if success:
                print(f"Saved frame {current_frame} as {filename}")
            else:
                print(f"Failed to save frame {current_frame}")

        current_frame += 1

    cap.release()

video_path = 'Ambar_blue.mp4'
output_dir = 'Frames'
desired_fps = 6

extract_frames(video_path, output_dir, desired_fps)
print("done")
