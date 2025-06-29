# src/detection/retinaface_detector.py
import cv2
import numpy as np
import os
import insightface
from src.utils.config import MODEL_CONFIG

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RetinaFaceDetector:
    """Face detection using RetinaFace (latest implementation)."""
    
    def __init__(self):
        """Initialize the RetinaFace detector with settings from config."""
        try:
            self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
            self.detector.prepare(ctx_id=0, nms=0.4)
            self.confidence_threshold = MODEL_CONFIG['face']['retinaface']['confidence_threshold']
            print("✅ RetinaFace detector initialized")
        except ImportError:
            print("❌ RetinaFace not available. Please install: pip install retinaface-pytorch")
            raise
    
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
        try:
            bboxes, landmarks = self.detector.detect(image, threshold=self.confidence_threshold)
            detections = []
            if bboxes is not None:
                for bbox in bboxes:
                    x1, y1, x2, y2, conf = bbox.astype(int)
                    if conf >= self.confidence_threshold:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'center': (center_x, center_y)
                        })
            return detections
            
        except Exception as e:
            print(f"❌ RetinaFace detection error: {e}")
            return []
    
    def __del__(self):
        """Clean up resources."""
        pass

    def draw_detections(self, frame, detections):
        """Draw face bounding boxes with debugging."""
        print(f"🎨 Drawing {len(detections)} RetinaFace detections")
        
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