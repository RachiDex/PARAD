#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio removal utility using FFmpeg.
"""

import subprocess
import os

def remove_audio(input_path, output_path):
    """
    Remove audio track from video file.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file without audio
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "copy",  # Copy video without re-encoding
        "-an",           # Remove audio
        output_path
    ]

    print(f"Removing audio: {input_path} -> {output_path}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("Error during re-encoding:", result.stderr.decode())
        raise RuntimeError("Re-encoding failed.")
    else:
        print("Audio-free video generated.")

# Example usage:
if __name__ == "__main__":
    video_in = "GX010621.mp4"
    video_out = "GX010621-no-audio.mp4"
    remove_audio(video_in, video_out)