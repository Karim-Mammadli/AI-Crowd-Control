import os
import sys
import warnings
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import tempfile


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')

from flask import Flask, send_from_directory, request, jsonify
from src.models.model_registry import CrowdMonitoringModelRegistry
from src.utils.config import MODEL_CONFIG, PATHS, ALLOWED_EXTENSIONS, MLFLOW_CONFIG
from src.utils.image_enhancer import QuickImageEnhancer
import atexit
from flask_socketio import SocketIO, emit
import threading
import time
import json
from datetime import datetime
import signal

# Import detection modules
def load_detection_modules():
    try:
        from src.detection.yolo_detector import YOLODetector
        from src.detection.face_detector import FaceDetector
        from src.utils.video_processor import VideoProcessor
        return YOLODetector, FaceDetector, VideoProcessor
    except Exception as e:
        print(f"Error loading detection modules: {e}")
        return None, None, None

app = Flask(__name__, static_folder='static')
# app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False, max_http_buffer_size=100*1024*1024)

# Create upload directories
os.makedirs(PATHS['uploads'], exist_ok=True)
os.makedirs(PATHS['processed'], exist_ok=True)

def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

class CrowdMonitoringSystem:
    def __init__(self):
        self.is_monitoring = False
        self.is_initializing = False
        self.models_loaded = False
        self.processing_mode = 'camera'  # 'camera', 'video', 'image'
        
        # AI Models
        self.video_processor = None
        self.yolo_detector = None
        self.face_detector = None
        
        # Threading
        self.processing_thread = None
        self._stop_event = threading.Event()
        self._initialization_lock = threading.Lock()

        
        self.image_enhancer = QuickImageEnhancer()

        
        # File processing
        self.current_video_path = None
        self.current_image_path = None

        self.model_registry = CrowdMonitoringModelRegistry(MLFLOW_CONFIG['experiment_name'])
        self.models_registered = False
        
        # Statistics
        self.stats = {
            'person_count': 0,
            'face_count': 0,
            'crowd_density': 'EMPTY',
            'alert_level': 'NORMAL',
            'last_activity': 'System ready - upload a file or use camera',
            'system_status': 'Ready'
        }
        
        print("🚀 AI Crowd Monitoring System initialized with file upload support")
    
    def update_progress(self, step, total, message):
        """Update loading progress bar."""
        progress = int((step / total) * 100)
        socketio.emit('loading_progress', {
            'step': step,
            'total': total,
            'progress': progress,
            'message': message
        })
        print(f"📊 Progress: {progress}% - {message}")
    
    def initialize_models(self):
        with self._initialization_lock:
            if self.models_loaded:
                print("✅ Models already loaded")
                return True
                
            if self.is_initializing:
                print("⏳ Models already initializing...")
                return False
                
            self.is_initializing = True
            
            try:
                print("📥 Starting AI model initialization...")
                total_steps = 5
                
                # Step 1: Import modules
                self.update_progress(1, total_steps, "Importing detection modules...")
                YOLODetector, FaceDetector, VideoProcessor = load_detection_modules()
                if None in [YOLODetector, FaceDetector, VideoProcessor]:
                    print("Failed to load one or more detection modules")
                    return False
                
                # Step 2: Initialize video processor (for camera if needed)
                self.update_progress(2, total_steps, "Initializing video processor...")
                print("🎥 Initializing video processor...")
                self.video_processor = VideoProcessor()
                
                # Step 3: Load YOLO model
                self.update_progress(3, total_steps, "Loading YOLOv8 person detection model...")
                print("🔄 Loading YOLOv8 model...")
                self.yolo_detector = YOLODetector(MODEL_CONFIG['yolo']['model_path'])
                
                # Step 4: Load MediaPipe model
                self.update_progress(4, total_steps, "Loading face detection model...")
                print("👤 Loading face detection...")
                self.face_detector = FaceDetector()
                
                # Step 5: Complete
                self.update_progress(5, total_steps, "All AI models loaded - ready for processing!")
                print("✅ All models loaded successfully!")
                
                socketio.emit('system_status', {
                    'status': 'ready', 
                    'message': 'All AI models loaded - ready to monitor!'
                })
                
                self.models_loaded = True
                self.is_initializing = False

                if self.models_loaded:
                    # Register models with MLflow for HP AI Studio deployment
                    threading.Thread(target=self.register_models_with_mlflow, daemon=True).start()

                return True
                
            except Exception as e:
                print(f"❌ Model loading error: {e}")
                socketio.emit('system_status', {
                    'status': 'error', 
                    'message': f'Model loading failed: {str(e)}'
                })
                self.is_initializing = False
                self.models_loaded = False
                return False
    
    def register_models_with_mlflow(self):
        """Register models with MLflow for HP AI Studio competition."""
        if not self.models_loaded or self.models_registered:
            return
        
        try:
            print("🏆 Registering for HP AI Studio Competition...")
            
            # Use the competition registration function
            run_id, model = self.model_registry.register_complete_system(
                self.yolo_detector,
                self.face_detector
            )
            
            if model:
                self.models_registered = True
                print("🎯 Competition models registered successfully!")
                print(f"   Model: {model.name} v{model.version}")
                print(f"   Demo folder created for HP AI Studio deployment")
                return True
            
        except Exception as e:
            print(f"❌ Competition registration error: {e}")
            return False
    
    # def register_models_with_mlflow(self):
    #     """Register models with MLflow after they're loaded."""
    #     if not self.models_loaded or self.models_registered:
    #         return
        
    #     try:
    #         print("📝 Registering models with MLflow...")
            
    #         # Register individual models
    #         yolo_run_id = self.model_registry.register_yolo_model(
    #             self.yolo_detector, 
    #             "crowd-yolo-detector-v1"
    #         )
            
    #         face_run_id = self.model_registry.register_face_model(
    #             self.face_detector,
    #             "crowd-face-detector-v1" 
    #         )
            
    #         # Register complete system
    #         system_run_id = self.model_registry.register_complete_system(
    #             self.yolo_detector,
    #             self.face_detector,
    #             "ai-crowd-monitoring-system-v1"
    #         )
            
    #         # Create demo artifacts for deployment
    #         demo_dir = self.model_registry.create_demo_artifacts("demo")
            
    #         if all([yolo_run_id, face_run_id, system_run_id, demo_dir]):
    #             self.models_registered = True
    #             print("✅ All models registered with MLflow successfully!")
                
    #             # List registered models
    #             self.model_registry.list_models()
                
    #         return True
        
        except Exception as e:
            print(f"❌ MLflow registration error: {e}")
            return False

    def process_image(self, image_path):
        """Process a single image and return results."""
        if not self.models_loaded:
            return {'success': False, 'message': 'Models not loaded'}
        
        try:
            print(f"🖼️ Processing image: {image_path}")
            
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                return {'success': False, 'message': 'Could not load image'}
            
            print(f"📷 Image shape: {frame.shape}")
            
            # Enhance image for better detection
            enhanced_frame = self.image_enhancer.enhance_for_detection(frame)
            
            # Run detections
            person_detections = self.yolo_detector.detect_persons(enhanced_frame)
            face_detections = self.face_detector.detect_faces(enhanced_frame)
            
            print(f"👥 Found {len(person_detections)} people")
            print(f"👤 Found {len(face_detections)} faces")
            
            # Draw detections on image
            result_frame = frame.copy()
            
            # Draw person boxes (green)
            for detection in person_detections:
                bbox = detection['bbox']
                conf = detection['confidence']
                cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                label = f"Person: {conf:.2f}"
                cv2.putText(result_frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw face boxes (blue)
            for detection in face_detections:
                bbox = detection['bbox']
                conf = detection['confidence']
                cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                label = f"Face: {conf:.2f}"
                cv2.putText(result_frame, label, (bbox[0], bbox[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Save processed image in the same format as the original upload
            base_filename = os.path.basename(image_path)
            if base_filename.startswith('processed_'):
                base_filename = base_filename[len('processed_'):]
            name, ext = os.path.splitext(base_filename)
            ext = ext.lower() if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] else '.jpg'
            processed_filename = f"processed_{name}{ext}"
            processed_path = os.path.join(PATHS['processed'], processed_filename)
            # Choose correct encoding for OpenCV
            if ext in ['.jpg', '.jpeg']:
                cv2.imwrite(processed_path, result_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            elif ext == '.png':
                cv2.imwrite(processed_path, result_frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            elif ext == '.bmp':
                cv2.imwrite(processed_path, result_frame)
            elif ext == '.webp':
                cv2.imwrite(processed_path, result_frame, [int(cv2.IMWRITE_WEBP_QUALITY), 90])
            else:
                cv2.imwrite(processed_path, result_frame)
            
            # Convert to base64 for frontend display
            _, buffer = cv2.imencode('.jpg', result_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Update stats
            self.stats.update({
                'person_count': len(person_detections),
                'face_count': len(face_detections),
                'crowd_density': self.calculate_crowd_density(len(person_detections)),
                'alert_level': self.calculate_alert_level(len(person_detections), len(face_detections)),
                'last_activity': f"Processed image: {len(person_detections)} people, {len(face_detections)} faces detected",
                'timestamp': datetime.now().isoformat(),
                'person_detections': person_detections,
                'face_detections': face_detections,
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
    
    def process_video(self, video_path):
        """Process video and emit real-time updates."""
        if not self.models_loaded:
            return {'success': False, 'message': 'Models not loaded'}
        
        try:
            print(f"🎬 Processing video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'message': 'Could not open video'}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Video: {width}x{height}, {fps} FPS, {frame_count} frames")
            
            # Prepare output video
            filename = os.path.basename(video_path)
            processed_path = os.path.join(PATHS['processed'], f"processed_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))
            
            self.is_monitoring = True
            frame_num = 0
            
            while self.is_monitoring:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Process frames in batches of 10
                if frame_num % 10 == 0:
                    print(f"🔍 Processing batch at frame {frame_num}/{frame_count}")
                    
                    # Run detections with improved parameters
                    person_detections = self.yolo_detector.detect_persons(frame)
                    face_detections = self.face_detector.detect_faces(frame)
                    
                    # Get timestamp in seconds
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                    # Emit detection results for this frame
                    socketio.emit('video_detection', {
                        'frame_index': frame_num,
                        'timestamp': timestamp,
                        'person_detections': person_detections,
                        'face_detections': face_detections
                    })
                    
                    # Update progress
                    progress = int((frame_num / frame_count) * 100)
                    socketio.emit('video_progress', {
                        'progress': progress,
                        'frame': frame_num,
                        'total_frames': frame_count,
                        'message': f"Processing batch at frame {frame_num}/{frame_count}"
                    })
                    
                    # Update stats and emit to frontend
                    self.stats.update({
                        'person_count': len(person_detections),
                        'face_count': len(face_detections),
                        'crowd_density': self.calculate_crowd_density(len(person_detections)),
                        'alert_level': self.calculate_alert_level(len(person_detections), len(face_detections)),
                        'last_activity': f"Frame {frame_num}: {len(person_detections)} people, {len(face_detections)} faces",
                        'timestamp': datetime.now().isoformat(),
                        'person_detections': person_detections,
                        'face_detections': face_detections,
                        'system_status': 'Processing Video'
                    })
                    
                    socketio.emit('detection_update', self.stats)
                
                # Draw detections on frame (for every frame)
                result_frame = frame.copy()
                
                # Draw person boxes
                for detection in person_detections if frame_num % 10 == 0 else []:
                    bbox = detection['bbox']
                    conf = detection['confidence']
                    cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(result_frame, label, (bbox[0], bbox[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw face boxes
                for detection in face_detections if frame_num % 10 == 0 else []:
                    bbox = detection['bbox']
                    conf = detection['confidence']
                    cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                    label = f"Face: {conf:.2f}"
                    cv2.putText(result_frame, label, (bbox[0], bbox[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Write frame to output video
                out.write(result_frame)
                
                # Small delay for real-time feel
                time.sleep(0.033)  # ~30 FPS
            
            cap.release()
            out.release()
            
            print(f"✅ Video processing complete: {processed_path}")
            
            return {
                'success': True,
                'processed_path': processed_path,
                'total_frames': frame_count,
                'message': f'Video processed successfully: {frame_count} frames'
            }
            
        except Exception as e:
            print(f"❌ Video processing error: {e}")
            return {'success': False, 'message': str(e)}
    
    def calculate_crowd_density(self, person_count):
        if person_count == 0:
            return 'EMPTY'
        elif person_count <= 2:
            return 'LOW'
        elif person_count <= 5:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def calculate_alert_level(self, person_count, face_count):
        total = person_count + (face_count * 0.3)
        if total == 0:
            return 'NORMAL'
        elif total <= 3:
            return 'NORMAL'
        elif total <= 6:
            return 'CAUTION'
        else:
            return 'ALERT'
    
    def stop_processing(self):
        """Stop any ongoing processing."""
        self.is_monitoring = False
        print("🛑 Processing stopped")

# Global system
monitor_system = CrowdMonitoringSystem()

# Routes for serving static files
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/css/<path:filename>')
def css_files(filename):
    return send_from_directory('static/css', filename)

@app.route('/js/<path:filename>')
def js_files(filename):
    return send_from_directory('static/js', filename)

# File upload routes
@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({'success': False, 'message': 'Invalid file type. Use JPG, PNG, BMP, or WEBP'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(PATHS['uploads'], filename)
        file.save(file_path)
        
        print(f"📁 Image uploaded: {file_path}")
        
        # Initialize models if needed
        if not monitor_system.models_loaded:
            if not monitor_system.initialize_models():
                return jsonify({'success': False, 'message': 'Failed to initialize AI models'})
        
        # Process image
        result = monitor_system.process_image(file_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'Image processed successfully',
                'processed_image': result['processed_image'],
                'stats': result['stats'],
                'processed_path': result['processed_path'],
                'processed_filename': result['processed_filename']
            })
        else:
            return jsonify(result)
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if not allowed_file(file.filename, 'video'):
            return jsonify({'success': False, 'message': 'Invalid file type. Use MP4, AVI, MOV, MKV, or WEBM'})
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(PATHS['uploads'], filename)
        file.save(file_path)
        
        print(f"📁 Video uploaded: {file_path}")
        
        # Initialize models if needed
        if not monitor_system.models_loaded:
            if not monitor_system.initialize_models():
                return jsonify({'success': False, 'message': 'Failed to initialize AI models'})
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully - processing will start',
            'file_path': file_path
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/download/<filename>')
def download_file(filename):
    """Download a processed file."""
    try:
        # Ensure the filename is secure and exists in the processed folder
        if not os.path.exists(os.path.join(PATHS['processed'], filename)):
            return jsonify({'success': False, 'message': 'File not found'}), 404
        
        return send_from_directory(PATHS['processed'], filename, as_attachment=True)
    except Exception as e:
        print(f"❌ Download error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# WebSocket handlers
@socketio.on('start_video_processing')
def handle_start_video_processing(data):
    video_path = data.get('file_path')
    print(f"📨 Starting video processing: {video_path}")
    
    # Process video in background thread
    def process_video_background():
        result = monitor_system.process_video(video_path)
        emit('video_processing_complete', result)
    
    thread = threading.Thread(target=process_video_background, daemon=True)
    thread.start()

@socketio.on('stop_processing')
def handle_stop_processing():
    print("📨 Stop processing requested")
    monitor_system.stop_processing()
    emit('processing_stopped', {'message': 'Processing stopped'})

@socketio.on('connect')
def handle_connect():
    print("🔗 WebSocket: Client connected")
    emit('status', {'message': 'Connected to AI Crowd Monitor'})

@socketio.on('disconnect')
def handle_disconnect():
    print("🔌 WebSocket: Client disconnected")

if __name__ == '__main__':
    print("🚀 Starting AI Crowd Monitoring System with File Upload...")
    print("📋 Open browser to: http://localhost:5000")
    print("📁 Upload videos/images instead of using camera")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, log_output=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        monitor_system.stop_processing()