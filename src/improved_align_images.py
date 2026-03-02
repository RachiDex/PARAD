#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image alignment module with evaluation metrics.
Supports SIFT, ORB and ECC methods.
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import csv

def mse(img1, img2):
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        img1, img2: Grayscale images
        
    Returns:
        float: MSE value
    """
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def normalized_cross_correlation(img1, img2):
    """
    Calculate Normalized Cross-Correlation between two images.
    
    Args:
        img1, img2: Grayscale images
        
    Returns:
        float: NCC value between -1 and 1
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1_mean = img1 - np.mean(img1)
    img2_mean = img2 - np.mean(img2)
    numerator = np.sum(img1_mean * img2_mean)
    denominator = np.sqrt(np.sum(img1_mean ** 2) * np.sum(img2_mean ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator

def improved_align_images(ref_path, input_folder, output_folder, target_size=(256, 256), method='sift'):
    """
    Align images in a folder to a reference image.
    
    Args:
        ref_path: Path to reference image
        input_folder: Folder containing images to align
        output_folder: Output folder for aligned images
        target_size: Resize dimensions (width, height)
        method: Alignment method ('sift', 'orb', 'ecc')
    """
    os.makedirs(output_folder, exist_ok=True)

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise ValueError(f"Reference image {ref_path} not found")

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.equalizeHist(ref_gray)

    if method == 'sift':
        detector = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    elif method == 'orb':
        detector = cv2.ORB_create(1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method == 'ecc':
        warp_mode = cv2.MOTION_EUCLIDEAN
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-5)
    else:
        raise ValueError("Unknown method")

    if method in ['sift', 'orb']:
        kp_ref, desc_ref = detector.detectAndCompute(ref_gray, None)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results = []

    for filename in tqdm(image_files, desc=f"Alignment ({method})"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        try:
            if method == 'ecc':
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                _, warp_matrix = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, warp_mode, criteria)
                aligned = cv2.warpAffine(img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:
                kp_img, desc_img = detector.detectAndCompute(img_gray, None)
                if desc_img is None or len(kp_img) < 10:
                    results.append([filename, None, None, 0])
                    continue

                matches = matcher.match(desc_ref, desc_img) if method == 'orb' \
                    else matcher.knnMatch(desc_ref, desc_img, k=2)

                if method == 'sift':
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                else:
                    good_matches = matches

                if len(good_matches) < 10:
                    results.append([filename, None, None, len(good_matches)])
                    continue

                good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if H is None:
                    results.append([filename, None, None, len(good_matches)])
                    continue

                aligned = cv2.warpPerspective(img, H, (ref_img.shape[1], ref_img.shape[0]))

            aligned = cv2.resize(aligned, target_size)
            cv2.imwrite(os.path.join(output_folder, filename), aligned)

            aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            ref_resized = cv2.resize(ref_gray, (aligned_gray.shape[1], aligned_gray.shape[0]))

            error = mse(ref_resized, aligned_gray)
            corr = normalized_cross_correlation(ref_resized, aligned_gray)
            num_matches = len(good_matches) if method in ['sift', 'orb'] else -1

            results.append([filename, error, corr, num_matches])

        except Exception as e:
            print(f"Error on {filename}: {str(e)}")
            results.append([filename, None, None, None])
            continue

    # Save CSV results
    csv_path = os.path.join(output_folder, f"alignment_metrics_{method}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'MSE', 'NCC', 'NumGoodMatches'])
        writer.writerows(results)

    print(f"Metrics saved to {csv_path}")

# Example usage
if __name__ == "__main__":
    ref_image = "ref_image.jpg"
    input_folder = "rail_normalized"
    output_folder = "rail_aligned_sift"
    improved_align_images(ref_image, input_folder, output_folder, method='sift')