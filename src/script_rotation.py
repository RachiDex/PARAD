#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perspective transformation for rail images.
"""

import cv2
import numpy as np
import os
import json
from tqdm import tqdm

# Read points from JSON
with open("points_rotation.json", "r") as f:
    data = json.load(f)
point_a = tuple(data["point_a"])
point_b = tuple(data["point_b"])

def transform_image(image, point_a, point_b):
    """
    Apply perspective transformation to image.
    
    Args:
        image: Input image
        point_a: Source top point
        point_b: Source bottom point
        
    Returns:
        numpy.ndarray: Transformed image
    """
    # Define new positions for points A and B
    new_point_a = (0, 0)
    new_point_b = (0, image.shape[0] - 1)
    
    # Source points (A, B, and two other corners)
    points_src = np.float32([
        point_a, 
        point_b, 
        [image.shape[1] - 1, 0], 
        [image.shape[1] - 1, image.shape[0] - 1]
    ])
    
    # Destination points
    points_dst = np.float32([
        new_point_a, 
        new_point_b, 
        [image.shape[1] - 1, 0], 
        [image.shape[1] - 1, image.shape[0] - 1]
    ])
    
    # Calculate perspective transformation
    matrix = cv2.getPerspectiveTransform(points_src, points_dst)
    
    # Apply transformation
    transformed_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    
    return transformed_image

# Folders
input_folder = "rail_aligned_ecc"
output_folder = "rail_rotate"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all images in folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Apply transformation to each image with progress bar
for image_name in tqdm(image_files, desc="Processing images"):
    input_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name)

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error reading image: {input_path}")
        continue

    # Define A and B points for each image
    point_a = (50, 0)
    point_b = (150, image.shape[0] - 1)

    transformed_image = transform_image(image, point_a, point_b)

    # Save transformed image
    cv2.imwrite(output_path, transformed_image)

print("Processing complete.")