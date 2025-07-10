# src/detection/face_detector_manager.py
import os
from src.utils.config import MODEL_CONFIG

class FaceDetectorManager:
    """Simple manager for face detectors."""
    
    @staticmethod
    def create_detector(detector_type):
        """
        Create a face detector based on the specified type.
        
        Args:
            detector_type (str): Type of detectors ('mediapipe', 'insightface')
            
        Returns:
            Face detector instance
        """
        detector_type = detector_type.lower()
        
        if detector_type == 'mediapipe':
            from src.detection.face_detector import FaceDetector
            return FaceDetector()
        elif detector_type == 'insightface':
            from src.detection.insightface_detector import InsightFaceDetector
            return InsightFaceDetector()
        else:
            print(f"⚠️ Unknown detector type: {detector_type}, falling back to MediaPipe")
            from src.detection.face_detector import FaceDetector
            return FaceDetector()
    
    @staticmethod
    def get_available_detectors():
        """Get list of available detector types."""
        detectors = ['mediapipe', 'insightface']
        return detectors 