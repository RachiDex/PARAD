#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frame alignment using multiple methods (SIFT, ORB, ECC).
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

def improved_align_images(ref_path, input_folder, output_folder, target_size=(256, 256), method='sift'):
    """
    Improved image alignment with method selection and preprocessing.
    
    Args:
        ref_path: Path to reference image
        input_folder: Folder containing images to align
        output_folder: Output folder for aligned images
        target_size: Resize dimensions (width, height)
        method: Alignment method ('sift', 'orb', 'ecc')
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load reference image
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise ValueError(f"Reference image {ref_path} not found")
    
    # Preprocess reference image
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.equalizeHist(ref_gray)
    
    # Initialize based on method
    if method == 'sift':
        detector = cv2.SIFT_create()
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    elif method == 'orb':
        detector = cv2.ORB_create(1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method == 'ecc':
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-5)
    else:
        raise ValueError("Unknown method")
    
    if method in ['sift', 'orb']:
        kp_ref, desc_ref = detector.detectAndCompute(ref_gray, None)
    
    # Process images
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in tqdm(image_files, desc=f"Alignment ({method})"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Preprocess image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        
        try:
            if method == 'ecc':
                # ECC method (no keypoints)
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, img_gray, warp_matrix, warp_mode, criteria)
                aligned = cv2.warpAffine(img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                       flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:
                # SIFT/ORB methods
                kp_img, desc_img = detector.detectAndCompute(img_gray, None)
                
                if desc_img is None or len(kp_img) < 10:
                    continue
                
                matches = matcher.match(desc_ref, desc_img) if method == 'orb' \
                          else matcher.knnMatch(desc_ref, desc_img, k=2)
                
                # Filter matches
                if method == 'sift':
                    good_matches = [m for m,n in matches if m.distance < 0.7*n.distance]
                else:
                    good_matches = matches
                
                if len(good_matches) < 10:
                    continue
                
                # Sort best matches
                good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
                
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                
                H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                if H is None:
                    continue
                
                aligned = cv2.warpPerspective(img, H, (ref_img.shape[1], ref_img.shape[0]))
            
            # Post-processing
            aligned = cv2.resize(aligned, target_size)
            cv2.imwrite(os.path.join(output_folder, filename), aligned)
            
        except Exception as e:
            print(f"Error on {filename}: {str(e)}")
            continue

# Execution parameters
if __name__ == "__main__":
    # Paths
    ref_image = "ref_image.jpg"
    input_folder = "rail_normalized"
    output_folder = "rail_aligned"
    
    # Try different methods in order of robustness
    methods = ['ecc', 'orb', 'sift']  # Start with most robust
    
    for method in methods:
        output_folder_method = f"{output_folder}_{method}"
        print(f"\nTrying {method.upper()} method...")
        
        try:
            improved_align_images(
                ref_image,
                input_folder,
                output_folder_method,
                target_size=(256, 256),
                method=method
            )
            print(f"Success with {method}! Results in {output_folder_method}")
            break
        except Exception as e:
            print(f"Failed with {method}: {str(e)}")