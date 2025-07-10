import cv2
import numpy as np
from ..components import QualityAssessor

class LaplacianVarianceQualityAssessor(QualityAssessor):
    """
    A quality assessor that uses the variance of the Laplacian to measure
    the blurriness of a face crop.
    """
    def __init__(self, blur_threshold: float = 100.0):
        """
        Initializes the Laplacian variance quality assessor.

        Args:
            blur_threshold: The threshold below which an image is considered
                blurry.
        """
        self.blur_threshold = blur_threshold

    def assess(self, face_crop: np.ndarray) -> float:
        """
        Assesses the quality of a face crop, returning a score (e.g., 0-1).

        Args:
            face_crop: The face image as a NumPy array.

        Returns:
            A quality score, where a higher score indicates better quality.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize the variance to a 0-1 score
        # This is a simple normalization, a more sophisticated one could be used.
        score = 1.0 if variance > self.blur_threshold else variance / self.blur_threshold
        return score