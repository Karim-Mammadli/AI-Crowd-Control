"""
This module contains the core logic for the Crowd Monitoring System.
"""

import os
import cv2
import dlib
import base64
import threading
from datetime import datetime
from typing import List

# Import utility and configuration
from src.utils.config import MODEL_CONFIG, PATHS

# Import detection modules dynamically to avoid loading heavy libraries upfront
def load_detection_modules():
    """
    Dynamically loads and returns the necessary AI/ML model classes.
    This approach helps in reducing initial memory footprint and load time.
    """
    try:
        from src.detection.yolo_detector import YOLODetector
        from src.detection.face_detector_manager import FaceDetectorManager
        from src.utils.video_processor import VideoProcessor
        from src.facestream.processor import FaceStreamProcessor
        from src.facestream.implementations.detectors import create_facestream_detector
        from src.facestream.implementations.recognizers import InsightFaceRecognizer
        from src.facestream.implementations.quality import LaplacianVarianceQualityAssessor
        from src.facestream.datastructures import TrackedFace
        return (YOLODetector, FaceDetectorManager, VideoProcessor, FaceStreamProcessor,
                create_facestream_detector, InsightFaceRecognizer, LaplacianVarianceQualityAssessor, TrackedFace)
    except ImportError as e:
        print(f"Error loading detection modules: {e}")
        return (None, None, None, None, None, None, None, None)

class CrowdMonitoringSystem:
    """
    Manages the state and operations of the crowd monitoring system, including
    model initialization, video/image processing, and statistics calculation.
    """
    def __init__(self, socketio):
        """
        Initializes the CrowdMonitoringSystem.

        Args:
            socketio: The Flask-SocketIO instance for real-time communication.
        """
        self.socketio = socketio
        self.is_monitoring = False
        self.is_initializing = False
        self.models_loaded = False

        # Processing state
        self.processing_mode = 'camera'  # Default mode: 'camera', 'video', 'image'

        # AI Models - initialized to None
        self.video_processor = None
        self.yolo_detector = None
        self.face_detector = None
        self.facestream_processor = None
        self.mtcnn_detector = None
        self.insightface_recognizer = None
        self.laplacian_quality_assessor = None
        self.known_faces_db = {}  # In a real app, this would be loaded from a persistent database

        # Threading control
        self.processing_thread = None
        self._stop_event = threading.Event()
        self._initialization_lock = threading.Lock()

        # File paths for current processing tasks
        self.current_video_path = None
        self.current_image_path = None

        # Real-time statistics
        self.stats = {
            'person_count': 0,
            'face_count': 0,
            'crowd_density': 'EMPTY',
            'alert_level': 'NORMAL',
            'last_activity': 'System ready - upload a file or use camera',
            'system_status': 'Ready'
        }

        # Current model selections
        self.current_person_model = 'yolov8'
        self.current_face_model = 'insightface'

    def update_progress(self, step, total, message):
        """Sends model loading progress updates to the frontend."""
        progress = int((step / total) * 100)
        self.socketio.emit('loading_progress', {
            'step': step,
            'total': total,
            'progress': progress,
            'message': message
        })
        print(f"📊 Progress: {progress}% - {message}")

    def initialize_models(self, person_model='yolov8', face_model='insightface'):
        """
        Initializes the AI models required for person and face detection.
        This function is thread-safe.
        """
        with self._initialization_lock:
            if self.models_loaded and self.current_person_model == person_model and self.current_face_model == face_model:
                print("✅ Models already loaded with the same configuration.")
                return True

            if self.is_initializing:
                print("⏳ Model initialization is already in progress.")
                return False

            self.is_initializing = True

            try:
                print(f"📥 Starting AI model initialization... Person: {person_model}, Face: {face_model}")
                total_steps = 4

                # Step 1: Import detection modules
                self.update_progress(1, total_steps, "Importing detection modules...")
                modules = load_detection_modules()
                if any(m is None for m in modules):
                    raise ImportError("Failed to load one or more detection modules.")
                (YOLODetector, FaceDetectorManager, VideoProcessor, FaceStreamProcessor,
                 create_facestream_detector, InsightFaceRecognizer, LaplacianVarianceQualityAssessor, _) = modules

                # Step 2: Initialize video processor
                self.update_progress(2, total_steps, "Initializing video processor...")
                self.video_processor = VideoProcessor()

                # Step 3: Load YOLO model for person detection
                self.update_progress(3, total_steps, f"Loading {person_model.upper()} person detection model...")
                self.yolo_detector = YOLODetector(person_model)
                print(f"✅ {person_model.upper()} model loaded: {self.yolo_detector.model_name}")

                # Notify frontend if a fallback model was used
                if self.yolo_detector.model_name != person_model:
                    self.socketio.emit('model_fallback', {
                        'requested_model': person_model,
                        'actual_model': self.yolo_detector.model_name,
                        'message': f'Model {person_model} not available, using {self.yolo_detector.model_name} instead.'
                    })

                # Step 4: Load face detection model
                self.update_progress(4, total_steps, f"Loading {face_model.upper()} face detection model...")
                # Step 4: Load face detection model
                self.update_progress(4, total_steps, f"Loading {face_model.upper()} face detection model...")
                
                # Use the new factory to create the selected detector
                selected_detector = create_facestream_detector(face_model)

                self.insightface_recognizer = InsightFaceRecognizer()
                self.laplacian_quality_assessor = LaplacianVarianceQualityAssessor(blur_threshold=100.0)
                
                self.facestream_processor = FaceStreamProcessor(
                    detector=selected_detector,
                    recognizer=self.insightface_recognizer,
                    quality_assessor=self.laplacian_quality_assessor,
                    known_faces=self.known_faces_db,
                    tracker_factory=dlib.correlation_tracker
                )
                self.face_detector = None  # FaceStreamProcessor handles its own detection

                # Finalize initialization
                self.update_progress(4, total_steps, f"All AI models loaded: {person_model.upper()} + {face_model.upper()} ready!")
                self.socketio.emit('system_status', {
                    'status': 'ready',
                    'message': f'All AI models loaded and ready to monitor!'
                })
                self.models_loaded = True
                self.current_person_model = person_model
                self.current_face_model = face_model
                return True

            except Exception as e:
                print(f"❌ Model loading error: {e}")
                self.socketio.emit('system_status', {
                    'status': 'error',
                    'message': f'Model loading failed: {str(e)}'
                })
                self.models_loaded = False
                return False
            finally:
                self.is_initializing = False

    def process_image(self, image_path, person_model='yolov8', face_model='insightface'):
        """Processes a single image for person and face detection."""
        if not self.initialize_models(person_model, face_model):
            return {'success': False, 'message': 'Failed to initialize models.'}

        try:
            print(f"🖼️ Processing image: {image_path}")
            frame = cv2.imread(image_path)
            if frame is None:
                return {'success': False, 'message': 'Could not load image.'}

            # Run detections
            person_detections = self.yolo_detector.detect_persons(frame)
            if self.facestream_processor:
                tracked_faces = self.facestream_processor.process_frame(frame)
                face_detections = [{'bbox': face.bbox, 'confidence': face.confidence, 'identity': face.identity} for face in tracked_faces]
            else:
                face_detections = self.face_detector.detect_faces(frame)

            print(f"👥 Found {len(person_detections)} people and {len(face_detections)} faces.")

            # Draw results on the image
            result_frame = self._draw_detections(frame, person_detections, face_detections)

            # Save processed image
            processed_filename, processed_path = self._save_processed_image(result_frame, image_path)

            # Encode image to base64 for frontend display
            _, buffer = cv2.imencode('.jpg', result_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Update and return stats
            self.stats.update({
                'person_count': len(person_detections),
                'face_count': len(face_detections),
                'crowd_density': self.calculate_crowd_density(len(person_detections)),
                'alert_level': self.calculate_alert_level(len(person_detections), len(face_detections)),
                'last_activity': f"Processed image: {len(person_detections)} people, {len(face_detections)} faces detected.",
                'timestamp': datetime.now().isoformat(),
                'system_status': 'Image Processed'
            })

            return {
                'success': True,
                'processed_image': img_base64,
                'processed_path': processed_path,
                'processed_filename': processed_filename,
                'stats': self.stats
            }

        except Exception as e:
            print(f"❌ Image processing error: {e}")
            return {'success': False, 'message': str(e)}

    def process_video(self, video_path, person_model='yolov8', face_model='insightface'):
        """Processes a video for person and face detection, emitting real-time updates."""
        if not self.initialize_models(person_model, face_model):
            self.socketio.emit('video_processing_complete', {'success': False, 'message': 'Failed to initialize models.'})
            return

        try:
            print(f"🎬 Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Could not open video file.")

            # Video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Prepare output video writer
            filename = os.path.basename(video_path)
            processed_path = os.path.join(PATHS['processed'], f"processed_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

            self.is_monitoring = True
            frame_num = 0
            person_detections, face_detections = [], []

            while self.is_monitoring:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1

                # Perform detection periodically (e.g., every 10 frames) for performance
                if frame_num % 10 == 0:
                    print(f"🔍 Processing batch at frame {frame_num}/{frame_count}")
                    person_detections = self.yolo_detector.detect_persons(frame)
                    if self.facestream_processor:
                        tracked_faces = self.facestream_processor.process_frame(frame)
                        face_detections = [{'bbox': face.bbox, 'confidence': face.confidence, 'identity': face.identity} for face in tracked_faces]
                    else:
                        face_detections = self.face_detector.detect_faces(frame)

                    # Emit progress and stats
                    # Draw detections on the frame to create the live feed image
                    result_frame = self._draw_detections(frame, person_detections, face_detections)

                    # Encode the frame with detections for live streaming
                    _, buffer = cv2.imencode('.jpg', result_frame)
                    live_frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit progress and stats, now including the live frame
                    self._emit_video_progress(frame_num, frame_count, person_detections, face_detections, live_frame_b64)

                # Draw the latest detections on every frame for the output video file
                result_frame_for_video = self._draw_detections(frame, person_detections, face_detections)
                out.write(result_frame_for_video)

            cap.release()
            out.release()
            print(f"✅ Video processing complete: {processed_path}")
            self.socketio.emit('video_processing_complete', {
                'success': True,
                'processed_path': processed_path,
                'total_frames': frame_count,
                'message': f'Video processed successfully: {frame_count} frames.'
            })

        except Exception as e:
            print(f"❌ Video processing error: {e}")
            self.socketio.emit('video_processing_complete', {'success': False, 'message': str(e)})
        finally:
            self.is_monitoring = False

    def _draw_detections(self, frame, person_detections, face_detections):
        """Helper function to draw detection bounding boxes on a frame."""
        result_frame = frame.copy()
        # Draw person boxes (green)
        for detection in person_detections:
            bbox, conf = detection['bbox'], detection['confidence']
            cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(result_frame, f"Person: {conf:.2f}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw face boxes (blue for unknown, yellow for known)
        for detection in face_detections:
            bbox, conf, identity = detection['bbox'], detection['confidence'], detection.get('identity')
            color = (0, 255, 255) if identity else (255, 0, 0)
            label = f"{identity} ({conf:.2f})" if identity else f"Face: {conf:.2f}"
            cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(result_frame, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return result_frame

    def _save_processed_image(self, frame, original_path):
        """Saves the processed image frame to disk."""
        base_filename = os.path.basename(original_path)
        name, ext = os.path.splitext(base_filename)
        ext = ext.lower() if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] else '.jpg'
        processed_filename = f"processed_{name}{ext}"
        processed_path = os.path.join(PATHS['processed'], processed_filename)

        # Use appropriate flags for different image formats
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(processed_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif ext == '.png':
            cv2.imwrite(processed_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        else:
            cv2.imwrite(processed_path, frame) # Default for BMP, WEBP etc.
        return processed_filename, processed_path

    def _emit_video_progress(self, frame_num, frame_count, person_detections, face_detections, live_frame_b64):
        """Emits a single, consolidated video update event via SocketIO."""
        progress = int((frame_num / frame_count) * 100)
        
        # Update internal stats
        self.stats.update({
            'person_count': len(person_detections),
            'face_count': len(face_detections),
            'crowd_density': self.calculate_crowd_density(len(person_detections)),
            'alert_level': self.calculate_alert_level(len(person_detections), len(face_detections)),
            'last_activity': f"Frame {frame_num}: {len(person_detections)} people, {len(face_detections)} faces",
            'timestamp': datetime.now().isoformat(),
            'system_status': 'Processing Video'
        })

        # Consolidate all data into a single payload
        update_payload = {
            'progress_data': {
                'progress': progress,
                'frame': frame_num,
                'total_frames': frame_count,
                'message': f"Processing batch at frame {frame_num}/{frame_count}"
            },
            'stats_data': self.stats.copy(), # Send a copy of the current stats
            'live_frame_b64': live_frame_b64
        }
        
        # Emit the single, combined event
        self.socketio.emit('video_update', update_payload)

    def calculate_crowd_density(self, person_count):
        """Calculates crowd density level based on person count."""
        thresholds = MODEL_CONFIG['crowd']['density_thresholds']
        if person_count == 0: return 'EMPTY'
        if person_count <= thresholds['low']: return 'LOW'
        if person_count <= thresholds['medium']: return 'MEDIUM'
        return 'HIGH'

    def calculate_alert_level(self, person_count, face_count):
        """Calculates an alert level based on person and face counts."""
        # A simple weighted calculation
        total = person_count + (face_count * 0.3)
        if total <= 3: return 'NORMAL'
        if total <= 6: return 'CAUTION'
        return 'ALERT'

    def stop_processing(self):
        """Stops any ongoing processing."""
        self.is_monitoring = False
        print("🛑 Processing stopped by request.")