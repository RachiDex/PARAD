#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video frame extraction based on distance traveled.
"""

import cv2
import os
import glob
import sys

# Auto-detect video file
video_files = glob.glob("*.mp4")
if not video_files:
    print("No .mp4 video files found in current directory.")
    sys.exit(1)

video_path = video_files[0]  # Take first video found
print(f"Video detected: {video_path}")

# Parameters
output_folder = "frames_50cm"  # Output folder for frames
speed_mps = 4.16               # 15 km/h ≈ 4.16 m/s
target_distance_m = 0.5        # 50 cm = 0.5 m

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: cannot open video {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
distance_per_frame = speed_mps / fps
frames_for_target = target_distance_m / distance_per_frame

print(f"FPS: {fps:.2f}")
print(f"Each frame ≈ {distance_per_frame*100:.2f} cm")
print(f"Extracting every {frames_for_target:.2f} frames")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % int(frames_for_target) == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Extraction complete: {saved_count} frames saved out of {frame_count} processed.")