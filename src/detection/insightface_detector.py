import cv2
import insightface
import numpy as np
import os
from src.utils.config import MODEL_CONFIG

class InsightFaceDetector:
    """Face detection using InsightFace (detection only for speed)."""

    def __init__(self):
        #use this for full model with recognition and gender/age prediction
        #self.model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        """Initialize the face detector with detection-only mode."""
        # Load only the detection model (SCRFD) directly
        self.detector = insightface.app.FaceAnalysis(
            providers=['CPUExecutionProvider']
        )
        # Force only detection module by clearing other models
        detection_model = self.detector.models.get('detection')
        self.detector.models = {}
        if detection_model:
            self.detector.models['detection'] = detection_model
        
        self.detector.prepare(ctx_id=0, det_size=(640, 640))
        self.confidence_threshold = MODEL_CONFIG['face']['confidence_threshold']
        print("✅ InsightFace detector initialized (detection-only mode)")

    def detect_faces(self, image):
        """
        Detect faces in the given image using InsightFace.

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
        print(f"🔍 InsightFace processing frame: {image.shape}")
        print(f"⚙️ Using InsightFace confidence threshold: {self.confidence_threshold}")
        
        faces = self.detector.get(image)
        detections = []
        height, width = image.shape[:2]

        print(f"📦 InsightFace found {len(faces)} potential faces")
        
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            confidence = face.det_score
            if confidence >= self.confidence_threshold:
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'center': (int(center_x), int(center_y))
                })
                print(f"   ✅ Face {i}: conf={confidence:.3f}, bbox=[{x1}, {y1}, {x2}, {y2}]")
            else:
                print(f"   ❌ Skipped face {i} (conf too low: {confidence:.3f})")

        print(f"🎯 InsightFace final detections: {len(detections)}")
        return detections

    def draw_detections(self, frame, detections):
        """Draw face bounding boxes with debugging."""
        print(f"🎨 Drawing {len(detections)} face detections")

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