import sys
import cv2
from typing import List # Import List
from facestream.processor import FaceStreamProcessor
from facestream.datastructures import TrackedFace

# Assume these implementations exist and are pre-trained
from facestream.implementations.detectors import MTCNNDetector
from facestream.implementations.recognizers import InsightFaceRecognizer
from facestream.implementations.quality import LaplacianVarianceQualityAssessor
import dlib # For the tracker

# 1. Load your known faces database
# This would typically be loaded from disk
known_faces_db = {
}

# 2. Instantiate components
detector = MTCNNDetector()
recognizer = InsightFaceRecognizer()
quality_assessor = LaplacianVarianceQualityAssessor(blur_threshold=100.0)

# 3. Create the main processor
processor = FaceStreamProcessor(
    detector=detector,
    recognizer=recognizer,
    quality_assessor=quality_assessor,
    known_faces=known_faces_db,
    tracker_factory=dlib.correlation_tracker 
)

# 4. Process a video stream
cap = cv2.VideoCapture(sys.argv[1])
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # The magic happens here!
    tracked_faces: List[TrackedFace] = processor.process_frame(frame)

    # Draw results on the frame
    for face in tracked_faces:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = face.identity or f"Track ID: {face.track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('FaceStream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()