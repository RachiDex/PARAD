#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image masking utility using line-based polygon masking.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def mask_image_with_line(image, point_a, point_b):
    """
    Apply mask to image using a line-defined polygon.
    
    Args:
        image: Input image (BGR)
        point_a: Top point (x, y)
        point_b: Bottom point (x, y)
        
    Returns:
        numpy.ndarray: Masked image
    """
    height, width = image.shape[:2]

    # Create white mask (keep all by default)
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Define polygon for masking
    polygon = np.array([
        [point_a[0], point_a[1]],    # A
        [width - 1, point_a[1]],     # Right at A level
        [width - 1, point_b[1]],     # Right at B level
        [point_b[0], point_b[1]],    # B
    ])

    # Fill polygon with black (mask out)
    cv2.fillPoly(mask, [polygon], 0)

    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

# Folders
input_folder = "rail_rotate"
output_folder = "rail_masked"

os.makedirs(output_folder, exist_ok=True)

# Load all images
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define A and B points
point_a = (110, 0)    # x,y for A
point_b = (256, 120)  # x,y for B

for image_name in tqdm(image_files, desc="Applying masking"):
    input_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name)

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error loading: {input_path}")
        continue

    result = mask_image_with_line(image, point_a, point_b)
    cv2.imwrite(output_path, result)

print("Masking complete.")