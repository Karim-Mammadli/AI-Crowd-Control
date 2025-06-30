from typing import List
import numpy as np
from mtcnn import MTCNN
from ..components import FaceDetector
from ..datastructures import BBox

# Import the other detector implementations to be wrapped
from src.detection.face_detector import FaceDetector as MediaPipeFaceDetector
from src.detection.insightface_detector import InsightFaceDetector as InsightFaceDetectorImpl

class MTCNNDetector(FaceDetector):
    """
    A face detector that uses the MTCNN (Multi-task Cascaded Convolutional Networks) model.
    """
    def __init__(self):
        """
        Initializes the MTCNN detector.
        """
        self.detector = MTCNN()

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Detects faces in a frame and returns a list of bounding boxes.
        """
        detections = self.detector.detect_faces(frame)
        bboxes = []
        for detection in detections:
            x1, y1, width, height = detection['box']
            x2, y2 = x1 + width, y1 + height
            bboxes.append((x1, y1, x2, y2))
        return bboxes

class MediaPipeDetectorWrapper(FaceDetector):
    """
    Wrapper for the MediaPipe face detector to conform to the FaceDetector interface.
    """
    def __init__(self):
        self.detector = MediaPipeFaceDetector()

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Detects faces and returns a list of bounding boxes.
        """
        detections = self.detector.detect_faces(frame)
        return [tuple(det['bbox']) for det in detections]

class InsightFaceDetectorWrapper(FaceDetector):
    """
    Wrapper for the InsightFace detector to conform to the FaceDetector interface.
    """
    def __init__(self):
        self.detector = InsightFaceDetectorImpl()

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Detects faces and returns a list of bounding boxes.
        """
        detections = self.detector.detect_faces(frame)
        return [tuple(det['bbox']) for det in detections]

def create_facestream_detector(detector_type: str) -> FaceDetector:
    """
    Factory function to create a face detector compatible with FaceStreamProcessor.
    Args:
        detector_type (str): The type of detector to create ('mtcnn', 'mediapipe', 'insightface').
    Returns:
        An instance of a FaceDetector.
    """
    detector_type = detector_type.lower()
    print(f"Creating facestream detector of type: {detector_type}")
    if detector_type == 'mtcnn':
        return MTCNNDetector()
    elif detector_type == 'mediapipe':
        return MediaPipeDetectorWrapper()
    elif detector_type == 'insightface':
        return InsightFaceDetectorWrapper()
    else:
        print(f"⚠️ Unknown facestream detector type: {detector_type}, falling back to MTCNN.")
        return MTCNNDetector()