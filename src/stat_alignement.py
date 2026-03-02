#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alignment method comparison and statistics.
"""

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Evaluation functions
def mse(img1, img2):
    """Calculate Mean Squared Error."""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def normalized_cross_correlation(img1, img2):
    """Calculate Normalized Cross-Correlation."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img1_mean = img1 - np.mean(img1)
    img2_mean = img2 - np.mean(img2)
    numerator = np.sum(img1_mean * img2_mean)
    denominator = np.sqrt(np.sum(img1_mean ** 2) * np.sum(img2_mean ** 2))
    return numerator / denominator if denominator != 0 else 0

# Main function
def align_and_evaluate(ref_path, input_folder, output_folder, method='sift', target_size=(256, 256)):
    """
    Align images and evaluate performance.
    
    Args:
        ref_path: Path to reference image
        input_folder: Folder containing images to align
        output_folder: Output folder for aligned images
        method: Alignment method ('sift', 'orb', 'ecc')
        target_size: Target image size
        
    Returns:
        pandas.DataFrame: Evaluation metrics
    """
    os.makedirs(output_folder, exist_ok=True)

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise ValueError("Reference image not found.")
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

    results = []
    for filename in tqdm(os.listdir(input_folder)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)

        try:
            if method == 'ecc':
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                cc, warp_matrix = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, warp_mode, criteria)
                aligned = cv2.warpAffine(img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                num_matches = -1
            else:
                kp_img, desc_img = detector.detectAndCompute(img_gray, None)
                if desc_img is None or len(kp_img) < 10:
                    results.append([filename, None, None, 0])
                    continue

                matches = matcher.match(desc_ref, desc_img) if method == 'orb' else matcher.knnMatch(desc_ref, desc_img, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance] if method == 'sift' else matches

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
                num_matches = len(good_matches)

            aligned = cv2.resize(aligned, target_size)
            ref_resized = cv2.resize(ref_gray, target_size)
            aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

            error = mse(ref_resized, aligned_gray)
            corr = normalized_cross_correlation(ref_resized, aligned_gray)
            results.append([filename, error, corr, num_matches])

            cv2.imwrite(os.path.join(output_folder, filename), aligned)
        except Exception as e:
            results.append([filename, None, None, None])
            continue

    df = pd.DataFrame(results, columns=["filename", "MSE", "NCC", "NumGoodMatches"])
    df["method"] = method
    csv_path = os.path.join(output_folder, f"alignment_metrics_{method}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved: {csv_path}")
    return df

# Main execution
if __name__ == "__main__":
    ref_image = "ref_image.jpg"
    input_folder = "rail_normalized"
    base_output = "rail_aligned"

    all_dfs = []
    for method in ["ecc", "orb", "sift"]:
        print(f"\n=== Alignment with {method.upper()} ===")
        df = align_and_evaluate(ref_image, input_folder, f"{base_output}_{method}", method=method)
        df = df.dropna()
        all_dfs.append(df)

    df_all = pd.concat(all_dfs)

    # Comparative plots
    sns.set(style="whitegrid")

    # MSE
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="method", y="MSE", data=df_all, palette="Set2")
    plt.title("Mean Squared Error (MSE) by Method")
    plt.tight_layout()
    plt.show()

    # NCC
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="method", y="NCC", data=df_all, palette="Set1")
    plt.title("Normalized Cross-Correlation (NCC) by Method")
    plt.tight_layout()
    plt.show()

    # NumGoodMatches (excluding ECC where it's -1)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="method", y="NumGoodMatches", data=df_all[df_all["NumGoodMatches"] >= 0], palette="Set3")
    plt.title("Number of Good Matches by Method")
    plt.tight_layout()
    plt.show()