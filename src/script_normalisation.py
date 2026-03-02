#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image normalization and resizing utility.
"""

import cv2
import os

def normalize_and_resize_images(input_folder, output_folder, size=(256, 256)):
    """
    Normalize and resize images to target size.
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder for normalized images
        size: Target size (width, height)
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading {img_path}")
                continue

            # Resize
            img_resized = cv2.resize(img, size)

            # Normalize: pixels between [0,1]
            img_normalized = img_resized.astype('float32') / 255.0

            # Save normalized images
            img_to_save = (img_normalized * 255).astype('uint8')
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, img_to_save)

    print(f"Normalized and resized images saved to {output_folder}")

# Example usage
if __name__ == "__main__":
    input_folder = 'rail_cropped'    # Manually cropped images
    output_folder = 'rail_normalized'  # Output folder
    size = (256, 256)  # Target size

    normalize_and_resize_images(input_folder, output_folder, size)