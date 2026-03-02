#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Homomorphic filtering for V channel extraction from HSV images.
"""

import cv2
import os

# Source folder containing images
input_folder = "rail_masked"

# Output folder for filtered images (V channel)
output_folder = "rail_seg_filtre"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process all files in source folder
for filename in os.listdir(input_folder):
    # Check if file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Full path to image
        img_path = os.path.join(input_folder, filename)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Unable to read image {filename}")
            continue
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract V channel (brightness)
        v_channel = hsv[:, :, 2]
        
        # Save V channel as grayscale image
        output_path = os.path.join(output_folder, filename.replace('.', '_V.'))
        cv2.imwrite(output_path, v_channel)

print("Processing complete.")