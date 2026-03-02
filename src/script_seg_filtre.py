#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rail segmentation using homomorphic filtering and morphological operations.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

input_folder = "rail_masked"
output_folder = "rail_seg_filtre"
os.makedirs(output_folder, exist_ok=True)

def homomorphic_filter(image, gamma_low=0.7, gamma_high=1.5, c=1, d0=30):
    """
    Apply homomorphic filter to enhance image.
    
    Args:
        image: Input BGR image
        gamma_low: Low frequency gain
        gamma_high: High frequency gain
        c: Filter sharpness
        d0: Cutoff frequency
        
    Returns:
        numpy.ndarray: Filtered grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    log_img = np.log1p(gray)
    dft = cv2.dft(log_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - rows // 2, v - cols // 2, indexing='ij')
    d = np.sqrt(u**2 + v**2)
    h = (gamma_high - gamma_low) * (1 - np.exp(-c * (d ** 2) / (d0 ** 2))) + gamma_low
    h = h.astype(np.float32)
    h = np.repeat(h[:, :, np.newaxis], 2, axis=2)
    dft_filtered = dft_shift * h

    dft_ishift = np.fft.ifftshift(dft_filtered)
    img_back = cv2.idft(dft_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = np.expm1(img_back)
    img_back = np.clip(img_back, 0, 1)
    return (img_back * 255).astype(np.uint8)

def extract_rail_segments(img):
    """
    Extract rail segments from image.
    
    Args:
        img: Input BGR image
        
    Returns:
        numpy.ndarray: Binary mask of rail segments
    """
    filtered = homomorphic_filter(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(filtered)

    # Lower threshold
    _, thresh = cv2.threshold(contrast, 150, 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(clean)

    h, w = mask.shape
    min_area = 200  # More permissive
    max_area = 30000
    rails_found = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Wider vertical position
        if h * 0.2 < y + ch // 2 < h * 0.8:
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            rails_found += 1

    return mask

# Processing
for fname in tqdm([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]):
    path = os.path.join(input_folder, fname)
    img = cv2.imread(path)
    if img is None:
        continue

    result = extract_rail_segments(img)
    cv2.imwrite(os.path.join(output_folder, fname), result)

print("Relaxed processing complete.")