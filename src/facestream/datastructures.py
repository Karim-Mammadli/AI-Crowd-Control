from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

# Bounding box type alias
BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

@dataclass
class TrackedFace:
    """The final output object for a single face in a frame."""
    bbox: BBox
    track_id: int
    identity: Optional[str] = None
    confidence: float = 0.0

@dataclass
class TrackState:
    """Internal state for managing a single tracked person."""
    track_id: int
    tracker: object  # The specific tracker instance (e.g., from dlib or OpenCV)
    bbox: BBox
    aggregated_embedding: Optional[np.ndarray] = None
    frames_since_update: int = 0
    misses: int = 0
    identity: Optional[str] = None # Add identity to TrackState