from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
from .datastructures import BBox

class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[BBox]:
        """Detects faces in a frame and returns a list of bounding boxes."""
        pass

class FaceRecognizer(ABC):
    @abstractmethod
    def compute_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """Computes a feature vector (embedding) for a face image."""
        pass

class QualityAssessor(ABC):
    @abstractmethod
    def assess(self, face_crop: np.ndarray) -> float:
        """Assesses the quality of a face crop, returning a score (e.g., 0-1)."""
        pass