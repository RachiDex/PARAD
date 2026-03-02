# PARAD: Pipeline for Automated Rail Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of the paper:  
**"PARAD: A Pipeline for Automated Rail Anomaly Detection: From Video Frames to Autoencoder-Based Analysis"**  
*Mohammed Rachid Khatir, Yahia Lebbah*

## 📋 Description

PARAD is a two-stage framework for tramway rail inspection:
- **Main Step I**: Unsupervised preprocessing (geometric alignment, photometric normalization, frequency-based rail segmentation)
- **Main Step II**: Unsupervised anomaly detection using a convolutional autoencoder

## 🚀 Installation

```bash
git clone https://github.com/votreusername/PARAD.git
cd PARAD
pip install -r requirements.txt