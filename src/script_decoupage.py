#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rail region cropping utility.
"""

import os
from PIL import Image

def crop_rail_region(input_folder, output_folder, crop_box=(10, 10, 5000, 3000)):
    """
    Crop rail region from images.
    
    Args:
        input_folder: Folder containing input images
        output_folder: Folder for cropped images
        crop_box: Crop coordinates (x1, y1, x2, y2)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('RGB')

            # Crop specified region
            img_cropped = img.crop(crop_box)

            # Save cropped image
            save_path = os.path.join(output_folder, filename)
            img_cropped.save(save_path)

    print(f"Rail region extracted and saved to {output_folder}")

# Function usage
if __name__ == "__main__":
    input_folder = 'frames_50cm'
    output_folder = 'rail_cropped'
    crop_box = (800, 400, 1300, 1080)

    crop_rail_region(input_folder, output_folder, crop_box)