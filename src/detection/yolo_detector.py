# src/detection/yolo_detector.py - Debug Version
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from src.utils.config import MODEL_CONFIG
import os

# class YOLOFaceDetector:
#     def __init__(self, model_name='yolov8n.pt'):
#         print(f"🔄 Initializing YOLO face detector with model: {model_name}")
#         try:
#             # Use the standard YOLOv8 model which is available for download
#             self.model = YOLO(model_name)
#             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#             print(f"✅ YOLO face model loaded on {self.device}")
            
#             # Test the model with a dummy image
#             print("🧪 Testing YOLO face model with dummy image...")
#             dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
#             test_results = self.model(dummy_img, verbose=False)
#             print(f"✅ YOLO face model test successful - {len(test_results)} result(s)")
            
#         except Exception as e:
#             print(f"❌ Model loading error: {e}")
#             print("Attempting to download model...")
#             try:
#                 # Try to download the model explicitly
#                 from ultralytics.utils.downloads import download
#                 download(model_name)
#                 self.model = YOLO(model_name)
#                 print("✅ Model downloaded and loaded successfully")
#             except Exception as download_error:
#                 print(f"❌ Failed to download model: {download_error}")
#                 raise download_error
    
#     def detect_faces(self, frame):
#         try:
#             if frame is None:
#                 print("⚠️ YOLO face detector received None frame")
#                 return []
            
#             print(f"🔍 YOLO face detector processing frame: {frame.shape}")
            
#             # Run detection with specific parameters for face detection
#             results = self.model(
#                 frame, 
#                 device=self.device,
#                 classes=[0],  # Person class
#                 conf=0.3,     # Confidence threshold
#                 verbose=False
#             )
            
#             print(f"📊 YOLO face detector raw results: {len(results)} result objects")
            
#             detections = []
#             if results and len(results) > 0:
#                 result = results[0]
#                 if result.boxes is not None and len(result.boxes) > 0:
#                     boxes = result.boxes.xyxy.cpu().numpy()
#                     scores = result.boxes.conf.cpu().numpy()
                    
#                     print(f"📦 YOLO face detector found {len(boxes)} potential faces")
                    
#                     for i, (box, score) in enumerate(zip(boxes, scores)):
#                         # Filter for face-like detections based on aspect ratio
#                         width = box[2] - box[0]
#                         height = box[3] - box[1]
#                         aspect_ratio = width / height
                        
#                         # Faces typically have an aspect ratio close to 1
#                         if 0.5 <= aspect_ratio <= 1.5:
#                             print(f"   Face {i}: confidence={score:.3f}, box={box}, aspect_ratio={aspect_ratio:.2f}")
                            
#                             if score > 0.3:
#                                 # Create a smaller box for the face (upper portion of person detection)
#                                 face_height = height * 0.3  # Face typically takes up top 30% of person height
#                                 face_box = [
#                                     int(box[0] + width * 0.1),  # Slightly inset from left
#                                     int(box[1]),                # Start from top of person
#                                     int(box[2] - width * 0.1),  # Slightly inset from right
#                                     int(box[1] + face_height)   # Only take top portion
#                                 ]
                                
#                                 detections.append({
#                                     'bbox': face_box,
#                                     'confidence': float(score),
#                                     'class': 'face'
#                                 })
#                                 print(f"   ✅ Added face {i} (conf: {score:.3f})")
#                             else:
#                                 print(f"   ❌ Skipped face {i} (conf too low: {score:.3f})")
#                         else:
#                             print(f"   ❌ Skipped detection {i} (not face-like aspect ratio: {aspect_ratio:.2f})")
#                 else:
#                     print("📦 YOLO face detector result has no boxes")
#             else:
#                 print("📦 YOLO face detector returned no results")
            
#             print(f"🎯 YOLO face detector final detections: {len(detections)}")
#             return detections
            
#         except Exception as e:
#             print(f"❌ YOLO face detection error: {e}")
#             import traceback
#             traceback.print_exc()
#             return []

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize YOLO detector with debugging."""
        print(f"🔄 Initializing YOLO detector with model: {model_name}")
        
        # Store the model name for later use
        self.model_name = model_name
        
        try:
            # Handle different model types
            if model_name == 'yolov11m':
                model_path = MODEL_CONFIG['yolo']['yolov11m']['model_path']
                print(f"🚀 Loading YOLOv11m model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLOv11m model not found at {model_path}")
            elif model_name == 'yolov11l':
                model_path = MODEL_CONFIG['yolo']['yolov11l']['model_path']
                print(f"🚀 Loading YOLOv11l model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLOv11l model not found at {model_path}")
            elif model_name == 'yolov11x':
                model_path = MODEL_CONFIG['yolo']['yolov11x']['model_path']
                print(f"🚀 Loading YOLOv11x model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLOv11x model not found at {model_path}")
            elif model_name == 'yolov11n':
                model_path = MODEL_CONFIG['yolo']['yolov11n']['model_path']
                print(f"🚀 Loading YOLOv11n model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLOv11n model not found at {model_path}")
            elif model_name == 'yolov11s':
                model_path = MODEL_CONFIG['yolo']['yolov11s']['model_path']
                print(f"🚀 Loading YOLOv11s model: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"YOLOv11s model not found at {model_path}")
            elif model_name == 'yolov8n':
                # Use default YOLOv8n model
                model_path = MODEL_CONFIG['yolo']['model_path']
                print(f"🚀 Loading YOLOv8n model: {model_path}")
            else:
                # Use provided model name directly
                model_path = model_name
                print(f"🚀 Loading custom model: {model_path}")
            
            # Try to load the model
            try:
                self.model = YOLO(model_path)
                print(f"✅ YOLO model loaded successfully")
            except Exception as model_error:
                print(f"⚠️ Model loading failed, attempting to download: {model_error}")
                # Try to download the model if it doesn't exist
                try:
                    print(f"📥 Downloading model: {model_path}")
                    self.model = YOLO(model_path)  # This should trigger download
                    print(f"✅ Model downloaded and loaded successfully")
                except Exception as download_error:
                    print(f"❌ Failed to download model: {download_error}")
                    # Fallback to YOLOv8n if YOLOv11 models fail
                    if model_name in ['yolov11m', 'yolov11l', 'yolov11x', 'yolov11n', 'yolov11s']:
                        print(f"🔄 Falling back to YOLOv8n model...")
                        fallback_path = MODEL_CONFIG['yolo']['model_path']
                        self.model = YOLO(fallback_path)
                        self.model_name = 'yolov8n'  # Update model name to reflect fallback
                        print(f"✅ Fallback to YOLOv8n successful")
                    else:
                        raise download_error
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"✅ YOLO model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ YOLO initialization error: {e}")
            raise e
    
    def detect_persons(self, frame):
        """Detect persons in frame with cleaner debugging."""
        try:
            if frame is None:
                print("⚠️ YOLO received None frame")
                return []
            
            print(f"🔍 YOLO processing frame: {frame.shape}")
            print(f"🤖 Active model: {self.model_name}")
            
            # Get confidence threshold from config based on model type
            if hasattr(self, 'model_name') and self.model_name == 'yolov11m':
                conf_threshold = MODEL_CONFIG['yolo']['yolov11m']['confidence_threshold']
                print(f"⚙️ Using YOLOv11m confidence threshold: {conf_threshold}")
            elif hasattr(self, 'model_name') and self.model_name == 'yolov11l':
                conf_threshold = MODEL_CONFIG['yolo']['yolov11l']['confidence_threshold']
                print(f"⚙️ Using YOLOv11l confidence threshold: {conf_threshold}")
            elif hasattr(self, 'model_name') and self.model_name == 'yolov11x':
                conf_threshold = MODEL_CONFIG['yolo']['yolov11x']['confidence_threshold']
                print(f"⚙️ Using YOLOv11x confidence threshold: {conf_threshold}")
            elif hasattr(self, 'model_name') and self.model_name == 'yolov11n':
                conf_threshold = MODEL_CONFIG['yolo']['yolov11n']['confidence_threshold']
                print(f"⚙️ Using YOLOv11n confidence threshold: {conf_threshold}")
            elif hasattr(self, 'model_name') and self.model_name == 'yolov11s':
                conf_threshold = MODEL_CONFIG['yolo']['yolov11s']['confidence_threshold']
                print(f"⚙️ Using YOLOv11s confidence threshold: {conf_threshold}")
            else:
                conf_threshold = MODEL_CONFIG['yolo']['confidence_threshold']
                print(f"⚙️ Using YOLOv8n confidence threshold: {conf_threshold}")
            
            # Run detection with specific parameters
            results = self.model(
                frame, 
                device=self.device, 
                classes=[0],  # Only person class
                conf=conf_threshold,  # Use config threshold
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
                    
                    # Process detections with cleaner output
                    for i, (box, score) in enumerate(zip(boxes, scores)):
                        if score > conf_threshold:  # Use config threshold
                            detections.append({
                                'bbox': [int(x) for x in box],
                                'confidence': float(score),
                                'class': 'person'
                            })
                            print(f"   ✅ Detection {i}: conf={score:.3f}, bbox=[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
                        else:
                            print(f"   ❌ Skipped detection {i} (conf too low: {score:.3f})")
                else:
                    print("📦 YOLO result has no boxes")
            else:
                print("📦 YOLO returned no results")
            
            print(f"🎯 YOLO final detections: {len(detections)} using {self.model_name}")
            
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

    def get_model_info(self):
        """Get information about the current model."""
        # Get confidence threshold based on model type
        if self.model_name == 'yolov11m':
            conf_threshold = MODEL_CONFIG['yolo']['yolov11m']['confidence_threshold']
        elif self.model_name == 'yolov11l':
            conf_threshold = MODEL_CONFIG['yolo']['yolov11l']['confidence_threshold']
        elif self.model_name == 'yolov11x':
            conf_threshold = MODEL_CONFIG['yolo']['yolov11x']['confidence_threshold']
        elif self.model_name == 'yolov11n':
            conf_threshold = MODEL_CONFIG['yolo']['yolov11n']['confidence_threshold']
        elif self.model_name == 'yolov11s':
            conf_threshold = MODEL_CONFIG['yolo']['yolov11s']['confidence_threshold']
        else:
            conf_threshold = MODEL_CONFIG['yolo']['confidence_threshold']
        
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'model_path': getattr(self.model, 'ckpt_path', 'Unknown'),
            'confidence_threshold': conf_threshold
        }
        return info
    
    def print_model_info(self):
        """Print detailed model information to console."""
        info = self.get_model_info()
        print("🤖 Model Information:")
        print(f"   📋 Model Name: {info['model_name']}")
        print(f"   🔧 Device: {info['device']}")
        print(f"   📁 Model Path: {info['model_path']}")
        print(f"   ⚙️ Confidence Threshold: {info['confidence_threshold']}")
        
        # Determine model type description
        if info['model_name'] == 'yolov11m':
            model_type = 'YOLOv11m (Medium)'
        elif info['model_name'] == 'yolov11l':
            model_type = 'YOLOv11l (Large)'
        elif info['model_name'] == 'yolov11x':
            model_type = 'YOLOv11x (X Large)'
        elif info['model_name'] == 'yolov11n':
            model_type = 'YOLOv11n (Nano)'
        elif info['model_name'] == 'yolov11s':
            model_type = 'YOLOv11s (Small)'
        elif info['model_name'] == 'yolov8n':
            model_type = 'YOLOv8n (Popular)'
        else:
            model_type = 'Custom Model'
        
        print(f"   🎯 Model Type: {model_type}")

# face_detector = YOLOFaceDetector()  # This will use yolov8n.pt which is available for download
