 # src/detection/face_detector.py - Debug Version
import cv2
import mediapipe as mp
import numpy as np
import os

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'

class FaceDetector:
    def __init__(self, confidence_threshold=0.3):
        """Initialize MediaPipe face detector with debugging."""
        print(f"🔄 Initializing MediaPipe face detector (confidence: {confidence_threshold})")
        
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.confidence_threshold = confidence_threshold
            
            # Initialize with optimized settings
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,   # 0 = short-range (faster)   1 = long-range (slower 5 meters)
                min_detection_confidence=confidence_threshold
            )
            
            print(f"✅ MediaPipe Face Detection initialized successfully")
            
        except Exception as e:
            print(f"❌ MediaPipe initialization error: {e}")
            raise e
    
    def detect_faces(self, frame):
        """Detect faces with extensive debugging."""
        try:
            if frame is None:
                print("⚠️ MediaPipe received None frame")
                return []
            
            print(f"👤 MediaPipe processing frame: {frame.shape}")
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"👤 Converted to RGB: {rgb_frame.shape}")
            
            # Run face detection
            results = self.face_detection.process(rgb_frame)
            print(f"👤 MediaPipe processing complete")
            
            detections = []
            if results.detections:
                h, w = frame.shape[:2]
                print(f"👤 Found {len(results.detections)} face detection(s)")
                
                for i, detection in enumerate(results.detections):
                    # Get confidence score
                    confidence = detection.score[0]
                    print(f"   Face {i}: confidence={confidence:.3f}")
                    
                    if confidence > self.confidence_threshold:
                        # Get bounding box (relative coordinates)
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert to absolute coordinates
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Ensure coordinates are within bounds
                        x = max(0, x)
                        y = max(0, y)
                        x2 = min(x + width, w)
                        y2 = min(y + height, h)
                        
                        face_detection = {
                            'bbox': [x, y, x2, y2],
                            'confidence': float(confidence),
                            'class': 'face'
                        }
                        
                        detections.append(face_detection)
                        print(f"   ✅ Added face {i}: bbox=[{x},{y},{x2},{y2}], conf={confidence:.3f}")
                    else:
                        print(f"   ❌ Skipped face {i} (conf too low: {confidence:.3f})")
            else:
                print("👤 No faces detected by MediaPipe")
            
            print(f"🎯 MediaPipe final detections: {len(detections)}")
            return detections
            
        except Exception as e:
            print(f"❌ MediaPipe detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
            print("🧹 MediaPipe Face Detection cleaned up")