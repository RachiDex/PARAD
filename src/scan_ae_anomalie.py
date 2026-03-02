#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anomaly detection using autoencoder reconstruction error.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from shutil import copy2

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 256
model_path = "autoencoder.pth"
input_dir = "rail_seg_filtre"
output_dir = "rail_ae_anomalies"
threshold = 0.015  # Anomaly threshold (adjust as needed)

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

# Model definition
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Load model
model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded from '{model_path}'")

def is_image_file(filename):
    """Check if file is an image based on extension."""
    return any(filename.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".bmp"])

def process_image(img_path):
    """
    Process single image and compute reconstruction error.
    
    Args:
        img_path: Path to image file
        
    Returns:
        float: Mean reconstruction error
    """
    img = Image.open(img_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img_tensor)
        error_map = torch.abs(recon - img_tensor).squeeze().cpu().numpy()
        mean_error = error_map.mean()
    return mean_error

# Process all images
print("Detecting anomalies...")
anomalies = []
for root, _, files in os.walk(input_dir):
    for file in files:
        if is_image_file(file):
            full_path = os.path.join(root, file)
            mean_error = process_image(full_path)

            if mean_error > threshold:
                print(f"Anomaly detected: {full_path} (error = {mean_error:.4f})")
                rel_path = os.path.relpath(full_path, input_dir)
                dest_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                copy2(full_path, dest_path)
                anomalies.append((full_path, mean_error))

print(f"\nProcessing complete. {len(anomalies)} anomalies found and copied to '{output_dir}'")