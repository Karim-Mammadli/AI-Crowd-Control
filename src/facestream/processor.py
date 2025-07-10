from typing import List, Dict, Tuple
import numpy as np
import dlib # Import dlib
from .components import FaceDetector, FaceRecognizer, QualityAssessor
from .datastructures import TrackedFace, TrackState, BBox
import cv2

def iou(bbox1: BBox, bbox2: BBox) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)

    # Calculate the area of both bounding boxes
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the IoU
    union_area = bbox1_area + bbox2_area - intersection_area
    if union_area == 0:
        return 0
    iou = intersection_area / union_area
    return iou

class FaceStreamProcessor:
    """
    Processes a video stream to perform efficient facial recognition.
    """
    def __init__(
        self,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        quality_assessor: QualityAssessor,
        known_faces: Dict[str, np.ndarray],
        tracker_factory, # e.g., a function that returns a new dlib.correlation_tracker
        tracker_confidence_threshold: float = 7.0, # Confidence threshold for dlib trackers
        keyframe_interval: int = 10,
        quality_threshold: float = 0.6,
        recognition_threshold: float = 0.8,
        max_misses: int = 5,
        embedding_alpha: float = 0.1 # For exponential moving average
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.quality_assessor = quality_assessor
        self.known_faces = known_faces
        self.tracker_factory = tracker_factory
        
        # Configuration
        self.tracker_confidence_threshold = tracker_confidence_threshold
        self.keyframe_interval = keyframe_interval
        self.quality_threshold = quality_threshold
        self.recognition_threshold = recognition_threshold
        self.max_misses = max_misses
        self.embedding_alpha = embedding_alpha
        
        # State
        self.active_tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> List[TrackedFace]:
        """
        Main processing loop for a single video frame.
        """
        # 1. Update existing trackers
        self._update_trackers(frame)
        
        # 2. On keyframes, run full detection to find new faces & correct drift
        if self.frame_count % self.keyframe_interval == 0:
            self._detect_and_match(frame)
        
        # 3. Perform quality assessment and recognition on high-quality tracks
        self._recognize_faces(frame)
        
        # 4. Clean up lost tracks
        self._cleanup_tracks()
        
        self.frame_count += 1
        return self._get_current_output()

    def _update_trackers(self, frame: np.ndarray):
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            tracker = track.tracker
            confidence = tracker.update(frame)
            
            # Get the updated position
            rect = tracker.get_position()
            x1 = int(rect.left())
            y1 = int(rect.top())
            x2 = int(rect.right())
            y2 = int(rect.bottom())
            track.bbox = (x1, y1, x2, y2)
            track.frames_since_update = 0
            
            # If tracker confidence is low or the bbox is invalid, count it as a miss.
            if confidence < self.tracker_confidence_threshold or x1 >= x2 or y1 >= y2:
                track.misses += 1
                if track.misses > self.max_misses:
                    tracks_to_remove.append(track_id)
            else:
                # If the track is good, reset the miss count
                track.misses = 0
        
        for track_id in tracks_to_remove:
            self.active_tracks.pop(track_id)
        
    def _detect_and_match(self, frame: np.ndarray):
        detections = self.detector.detect(frame)
        
        # --- Match detections to existing tracks ---
        
        # Handle cases with no detections or no tracks
        if not detections:
            for track in self.active_tracks.values():
                track.misses += 1
            return
        if not self.active_tracks:
            for bbox in detections:
                self._create_new_track(frame, bbox)
            return

        # Create a list of potential matches (detection_idx, track_id, iou_score)
        potential_matches = []
        for i, detection in enumerate(detections):
            for track_id, track in self.active_tracks.items():
                current_iou = iou(detection, track.bbox)
                if current_iou > 0.3:  # Threshold for considering a match
                    potential_matches.append((i, track_id, current_iou))
        
        # Sort matches by IoU score (highest first) to greedily find the best pairs
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        matched_detection_indices = set()
        matched_track_ids = set()
        
        for det_idx, track_id, iou_score in potential_matches:
            # If detection or track is already matched, skip
            if det_idx in matched_detection_indices or track_id in matched_track_ids:
                continue
            
            # This is a good one-to-one match, accept it
            track = self.active_tracks[track_id]
            detection = detections[det_idx]
            
            # Update track with the new, more accurate detection
            track.bbox = detection
            track.misses = 0
            dlib_rect = dlib.rectangle(int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]))
            track.tracker.start_track(frame, dlib_rect)
            
            matched_detection_indices.add(det_idx)
            matched_track_ids.add(track_id)
            
            # Suppress other detections that are very similar to this one
            for i, other_detection in enumerate(detections):
                if i not in matched_detection_indices and iou(detection, other_detection) > 0.9:
                    matched_detection_indices.add(i)

        # --- Update state based on matches ---

        # Create new tracks for any detections that were not matched
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                self._create_new_track(frame, detection)
        
        # Increment miss count for any tracks that were not matched
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_track_ids:
                track.misses += 1
        
    def _recognize_faces(self, frame: np.ndarray):
        for track_id, track in self.active_tracks.items():
            face_crop = self._crop_face(frame, track.bbox)
            
            # Skip if crop is empty
            if face_crop.size == 0:
                continue
                
            if self.quality_assessor.assess(face_crop) < self.quality_threshold:
                continue

            # Compute embedding
            current_embedding = self.recognizer.compute_embedding(face_crop)
            
            # Aggregate using exponential moving average
            if track.aggregated_embedding is None:
                track.aggregated_embedding = current_embedding
            else:
                track.aggregated_embedding = (
                    self.embedding_alpha * current_embedding +
                    (1 - self.embedding_alpha) * track.aggregated_embedding
                )

            # Match against known faces
            best_match = None
            best_confidence = 0.0
            for identity, known_embedding in self.known_faces.items():
                confidence = np.dot(track.aggregated_embedding, known_embedding) / (np.linalg.norm(track.aggregated_embedding) * np.linalg.norm(known_embedding))
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = identity

            if best_confidence > self.recognition_threshold:
                track.identity = best_match

    def _create_new_track(self, frame: np.ndarray, bbox: BBox):
        tracker = self.tracker_factory()
        
        # Convert bbox to dlib rectangle
        dlib_rect = dlib.rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        tracker.start_track(frame, dlib_rect)
        
        new_track = TrackState(
            track_id=self.next_track_id,
            tracker=tracker,
            bbox=bbox,
            aggregated_embedding=None,
            frames_since_update=0,
            misses=0
        )
        self.active_tracks[self.next_track_id] = new_track
        self.next_track_id += 1
        
    def _cleanup_tracks(self):
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track.misses > self.max_misses:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.active_tracks.pop(track_id)
        
    def _get_current_output(self) -> List[TrackedFace]:
        tracked_faces = []
        for track in self.active_tracks.values():
            tracked_face = TrackedFace(
                bbox=track.bbox,
                track_id=track.track_id,
                identity=track.identity,
                confidence=0.0 #TODO: Expose confidence
            )
            tracked_faces.append(tracked_face)
        return tracked_faces

    def _crop_face(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Crops a face from the frame given a bounding box."""
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]