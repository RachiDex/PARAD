#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main pipeline for rail inspection system.
Executes the complete image processing workflow from video to anomaly detection.
"""

import cv2
import os
import json
from tqdm import tqdm
import glob

print("Step 1: Extracting frames from video...")
video_files = glob.glob("*.mp4")
if not video_files:
    raise FileNotFoundError("No .mp4 video files found.")
video_path = video_files[0]
print(f"Video detected: {video_path}")
os.system(f"python script_extraction_par_distance.py \"{video_path}\"")

# Step 2: Image cropping
print("Step 2: Cropping rail regions...")
os.system("python script_decoupage.py")

# Step 3: Normalization and Resize
print("Step 3: Normalizing images...")
os.system("python script_normalisation.py")

# Step 4: Homography alignment
print("Step 4: Aligning frames...")
os.system("python script_alignement_frame.py")

# User interaction for point selection
def get_alignment_points(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    points = []
    h = img.shape[0]

    def click_event(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(x)
            print(f"Point {len(points)} recorded: x = {x}")
            if len(points) == 2:
                cv2.destroyAllWindows()
    
    print("Click two points (top and bottom - only X coordinate is used)")
    cv2.imshow("Select two points", img)
    cv2.setMouseCallback("Select two points", click_event)
    
    while len(points) < 2:
        cv2.waitKey(1)
    
    point_a = (points[0], 0)
    point_b = (points[1], h - 1)
    
    print(f"Point A: {point_a}")
    print(f"Point B: {point_b}")
    return point_a, point_b

# Interactive call after alignment
aligned_image_path = os.path.join("rail_aligned_ecc", "frame_0000.jpg")
point_a, point_b = get_alignment_points(aligned_image_path)

# Store points in JSON for next script
with open("points_rotation.json", "w") as f:
    json.dump({"point_a": point_a, "point_b": point_b}, f)

# Step 5: Perspective transformation
print("Step 5: Applying perspective transformation...")
os.system("python script_rotation.py")
os.system("python script_cut.py")

# Step 6: Segmentation and filtering
print("Step 6: Rail segmentation...")
os.system("python script_filtre_homomorphique.py")

# Step 7: Anomaly detection with autoencoder
print("Step 7: Anomaly detection using autoencoder...")
os.system("python scan_ae_anomalie.py")

print("\nPipeline completed successfully.")