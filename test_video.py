#!/usr/bin/env python3
"""
Simple YOLO Video Test Script
Clean and minimal - just input model and video path.
"""

from ultralytics import YOLO
import cv2
import os

def test_yolo_on_video(model_name, video_path):
    """Test a YOLO model on a video file."""
    # Load the model
    model = YOLO(model_name)
    
    # Run inference
    results = model(
        video_path, 
        save=True, 
        show=True,
        conf=0.3,
        classes=[0]  # Only detect people
    )
    
    return True


def main():

    # Run test
    test_yolo_on_video("yolo11n.pt", "uploads/bum.webm")

if __name__ == "__main__":
    main() 