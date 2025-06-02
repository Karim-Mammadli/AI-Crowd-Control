import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')

from flask import Flask, send_from_directory
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
    except ImportError as e:
        print(f"Warning: Could not import detection modules: {e}")
        return None, None, None

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

class CrowdMonitoringSystem:
    def __init__(self):
        self.is_monitoring = False
        self.is_initializing = False
        self.models_loaded = False
        
        # AI Models
        self.video_processor = None
        self.yolo_detector = None
        self.face_detector = None
        
        # Threading
        self.processing_thread = None
        self._stop_event = threading.Event()
        self._initialization_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'person_count': 0,
            'face_count': 0,
            'crowd_density': 'EMPTY',
            'alert_level': 'NORMAL',
            'last_activity': 'System ready - click Start to begin monitoring',
            'system_status': 'Ready'
        }
        
        print("🚀 AI Crowd Monitoring System initialized")
    
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
                    raise ImportError("Detection modules not available")
                
                # Step 2: Initialize video processor
                self.update_progress(2, total_steps, "Initializing camera system...")
                print("🎥 Initializing video processor...")
                self.video_processor = VideoProcessor()
                
                # Step 3: Load YOLO model
                self.update_progress(3, total_steps, "Loading YOLOv8 person detection model...")
                print("🔄 Loading YOLOv8 model...")
                self.yolo_detector = YOLODetector('yolov8n.pt')
                
                # Step 4: Load MediaPipe model
                self.update_progress(4, total_steps, "Loading MediaPipe face detection model...")
                print("👤 Loading MediaPipe face detection...")
                self.face_detector = FaceDetector(confidence_threshold=0.5)
                
                # Step 5: Complete
                self.update_progress(5, total_steps, "All AI models loaded successfully!")
                print("✅ All models loaded successfully!")
                
                socketio.emit('system_status', {
                    'status': 'ready', 
                    'message': 'All AI models loaded - ready to monitor!'
                })
                
                self.models_loaded = True
                self.is_initializing = False
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
    
    def start_monitoring(self):
        print(f"🔄 Start monitoring requested")
        
        if self.is_monitoring:
            return {'success': False, 'message': 'Already monitoring'}
        
        if self.is_initializing:
            return {'success': False, 'message': 'Still initializing...'}
        
        # Initialize models if needed
        if not self.models_loaded:
            socketio.emit('system_status', {'status': 'loading', 'message': 'Loading AI models...'})
            if not self.initialize_models():
                return {'success': False, 'message': 'Model initialization failed'}
        
        # Start video capture
        print("🎥 Starting video capture...")
        if not self.video_processor.start_capture():
            return {'success': False, 'message': 'Camera access failed - check camera permissions'}
        
        # Verify camera is working
        print("🧪 Testing camera frames...")
        test_attempts = 0
        while test_attempts < 10:
            test_frame = self.video_processor.get_frame()
            if test_frame is not None:
                print(f"✅ Camera test successful - frame shape: {test_frame.shape}")
                break
            test_attempts += 1
            time.sleep(0.1)
        
        if test_attempts >= 10:
            self.video_processor.stop_capture()
            return {'success': False, 'message': 'Camera test failed - no frames received'}
        
        # Start monitoring
        self._stop_event.clear()
        self.is_monitoring = True
        self.processing_thread = threading.Thread(target=self.process_video_loop, daemon=True)
        self.processing_thread.start()
        
        print("✅ Monitoring started successfully")
        return {'success': True, 'message': 'Monitoring started - AI detection active'}
    
    def stop_monitoring(self):
        print(f"🛑 Stop monitoring requested")
        
        if not self.is_monitoring:
            return {'success': True, 'message': 'Not monitoring'}
        
        self._stop_event.set()
        self.is_monitoring = False
        
        if self.video_processor:
            self.video_processor.stop_capture()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)
        
        # Reset stats
        self.stats.update({
            'person_count': 0,
            'face_count': 0,
            'crowd_density': 'EMPTY',
            'alert_level': 'NORMAL',
            'last_activity': 'Monitoring stopped',
            'system_status': 'Ready',
            'person_detections': [],
            'face_detections': []
        })
        
        socketio.emit('detection_update', self.stats)
        print("✅ Monitoring stopped")
        return {'success': True, 'message': 'Monitoring stopped'}
    
    def process_video_loop(self):
        frame_count = 0
        last_emit_time = time.time()
        consecutive_failures = 0
        
        print("🎥 === VIDEO PROCESSING LOOP STARTED ===")
        
        while self.is_monitoring and not self._stop_event.is_set():
            try:
                # Get frame
                frame = self.video_processor.get_frame()
                
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures % 50 == 0:  # Log every 50 failures
                        print(f"⚠️ No frames for {consecutive_failures} attempts")
                    time.sleep(0.05)
                    continue
                
                # Reset failure counter on success
                if consecutive_failures > 0:
                    print(f"✅ Frame capture resumed after {consecutive_failures} failures")
                    consecutive_failures = 0
                
                frame_count += 1
                
                # Process every 10th frame for debugging (3 FPS effective)
                if frame_count % 10 == 0:
                    print(f"\n🔍 === PROCESSING FRAME {frame_count} ===")
                    print(f"📷 Frame shape: {frame.shape}")
                    
                    # Run YOLO detection
                    print("🔄 Running YOLO detection...")
                    person_detections = self.yolo_detector.detect_persons(frame)
                    print(f"👥 YOLO result: {len(person_detections)} people")
                    
                    # Run MediaPipe detection
                    print("🔄 Running MediaPipe detection...")
                    face_detections = self.face_detector.detect_faces(frame)
                    print(f"👤 MediaPipe result: {len(face_detections)} faces")
                    
                    person_count = len(person_detections)
                    face_count = len(face_detections)
                    
                    # Update stats
                    self.stats.update({
                        'person_count': person_count,
                        'face_count': face_count,
                        'crowd_density': self.calculate_crowd_density(person_count),
                        'alert_level': self.calculate_alert_level(person_count, face_count),
                        'last_activity': self.generate_activity_description(person_count, face_count),
                        'timestamp': datetime.now().isoformat(),
                        'person_detections': person_detections,
                        'face_detections': face_detections,
                        'system_status': 'Monitoring Active'
                    })
                    
                    print(f"📊 Final stats: people={person_count}, faces={face_count}")
                    
                    # Emit to frontend
                    current_time = time.time()
                    if current_time - last_emit_time >= 1.0:  # 1 FPS for UI updates
                        socketio.emit('detection_update', self.stats)
                        print(f"✅ Data sent to frontend")
                        last_emit_time = current_time
                    
                    print(f"🔍 === FRAME {frame_count} COMPLETE ===\n")
                
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(0.1)
            
            time.sleep(0.033)  # ~30 FPS capture rate
        
        print("🏁 === VIDEO PROCESSING LOOP ENDED ===")
    
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
    
    def generate_activity_description(self, person_count, face_count):
        if person_count == 0 and face_count == 0:
            return "No detections - area appears empty"
        elif person_count == 0 and face_count > 0:
            return f"{face_count} face(s) detected - partial person visibility"
        elif person_count > 0 and face_count == 0:
            return f"{person_count} person(s) detected - faces not clearly visible"
        else:
            return f"{person_count} person(s), {face_count} face(s) detected - good visibility"
    
    def cleanup(self):
        print("🧹 Cleaning up...")
        self.stop_monitoring()
        if self.face_detector:
            self.face_detector.cleanup()

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

# WebSocket handlers
@socketio.on('start_monitoring')
def handle_start_monitoring():
    print("📨 WebSocket: start_monitoring received")
    result = monitor_system.start_monitoring()
    emit('monitoring_status', {
        'active': result['success'], 
        'message': result['message']
    })

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    print("📨 WebSocket: stop_monitoring received")
    result = monitor_system.stop_monitoring()
    emit('monitoring_status', {
        'active': False, 
        'message': result['message']
    })

@socketio.on('connect')
def handle_connect():
    print("🔗 WebSocket: Client connected")
    emit('status', {'message': 'Connected to AI Crowd Monitor'})

@socketio.on('disconnect')
def handle_disconnect():
    print("🔌 WebSocket: Client disconnected")

if __name__ == '__main__':
    print("🚀 Starting AI Crowd Monitoring System...")
    print("🔧 Backend only - separated frontend files")
    print("📋 Open browser to: http://localhost:5000")
    print("🔧 Watch console for detection logs")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, log_output=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        monitor_system.cleanup()