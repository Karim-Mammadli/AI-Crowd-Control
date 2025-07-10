# src/detection/face_detector.py - MediaPipe Implementation
import cv2
import mediapipe as mp
import numpy as np
import os
from src.utils.config import MODEL_CONFIG

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'

class FaceDetector:
    """Face detection using MediaPipe."""
    
    def __init__(self):
        """Initialize the face detector with settings from config."""
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=MODEL_CONFIG['face']['mediapipe']['model_selection'],
            min_detection_confidence=MODEL_CONFIG['face']['mediapipe']['confidence_threshold']
        )
        print("✅ MediaPipe face detector initialized")
    
    def detect_faces(self, image):
        """
        Detect faces in the given image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries containing face detection results:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'center': (x, y)
            }
        """
        print(f"🔍 MediaPipe processing frame: {image.shape}")
        print(f"⚙️ Using MediaPipe confidence threshold: {MODEL_CONFIG['face']['mediapipe']['confidence_threshold']}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(image_rgb)
        
        detections = []
        if results.detections:
            height, width = image.shape[:2]
            print(f"📦 MediaPipe found {len(results.detections)} potential faces")
            
            for i, detection in enumerate(results.detections):
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * width)
                y1 = int(bbox.ymin * height)
                x2 = int((bbox.xmin + bbox.width) * width)
                y2 = int((bbox.ymin + bbox.height) * height)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                confidence = detection.score[0]
                if confidence >= MODEL_CONFIG['face']['mediapipe']['confidence_threshold']:
                    detections.append({
                    'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                    'center': (center_x, center_y)
                })
                    print(f"   ✅ Face {i}: conf={confidence:.3f}, bbox=[{x1}, {y1}, {x2}, {y2}]")
                else:
                    print(f"   ❌ Skipped face {i} (conf too low: {confidence:.3f})")
        else:
            print("📦 MediaPipe found no faces")
        
        print(f"🎯 MediaPipe final detections: {len(detections)}")
        return detections
    
    def __del__(self):
        """Clean up resources."""
        self.face_detection.close()

    def draw_detections(self, frame, detections):
        """Draw face bounding boxes with debugging."""
        print(f"🎨 Drawing {len(detections)} MediaPipe face detections")
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            
            print(f"   Drawing face {i}: bbox={bbox}, conf={conf}")
            
            # Draw bounding box (blue for faces)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            
            # Draw confidence score
            label = f"Face: {conf:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame