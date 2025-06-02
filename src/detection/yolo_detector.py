 # src/detection/yolo_detector.py - Debug Version
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize YOLO detector with debugging."""
        print(f"🔄 Initializing YOLO detector with model: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"✅ YOLO model loaded successfully on {self.device}")
            
            # Test the model with a dummy image
            print("🧪 Testing YOLO model with dummy image...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = self.model(dummy_img, verbose=False)
            print(f"✅ YOLO model test successful - {len(test_results)} result(s)")
            
        except Exception as e:
            print(f"❌ YOLO initialization error: {e}")
            raise e
    
    def detect_persons(self, frame):
        """Detect persons in frame with extensive debugging."""
        try:
            if frame is None:
                print("⚠️ YOLO received None frame")
                return []
            
            print(f"🔍 YOLO processing frame: {frame.shape}")
            
            # Run detection with specific parameters
            results = self.model(
                frame, 
                device=self.device, 
                classes=[0],  # Only person class
                conf=0.3,     # Lower confidence threshold
                verbose=False
            )
            
            print(f"📊 YOLO raw results: {len(results)} result objects")
            
            detections = []
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    
                    print(f"📦 YOLO found {len(boxes)} potential detections")
                    
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        print(f"   Detection {i}: confidence={score:.3f}, box={box}")
                        
                        if score > 0.3:  # Lower threshold for debugging
                            detections.append({
                                'bbox': [int(x) for x in box],
                                'confidence': float(score),
                                'class': 'person'
                            })
                            print(f"   ✅ Added detection {i} (conf: {score:.3f})")
                        else:
                            print(f"   ❌ Skipped detection {i} (conf too low: {score:.3f})")
                else:
                    print("📦 YOLO result has no boxes")
            else:
                print("📦 YOLO returned no results")
            
            print(f"🎯 YOLO final detections: {len(detections)}")
            for i, det in enumerate(detections):
                print(f"   Final {i}: {det}")
            
            return detections
            
        except Exception as e:
            print(f"❌ YOLO detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame with debugging."""
        print(f"🎨 Drawing {len(detections)} YOLO detections")
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            
            print(f"   Drawing detection {i}: bbox={bbox}, conf={conf}")
            
            # Draw bounding box (green)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Person: {conf:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame