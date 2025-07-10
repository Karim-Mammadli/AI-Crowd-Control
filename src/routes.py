"""
This module defines the web routes and WebSocket event handlers for the application.
"""

import os
from flask import request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import threading

from src.utils.config import PATHS, ALLOWED_EXTENSIONS

def register_routes(app, socketio, monitor_system):
    """
    Registers all Flask and SocketIO routes for the application.

    Args:
        app: The Flask application instance.
        socketio: The Flask-SocketIO instance.
        monitor_system: The instance of the CrowdMonitoringSystem.
    """

    def allowed_file(filename, file_type):
        """Checks if the uploaded file has an allowed extension."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

    # --- Static File Routes ---
    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')

    @app.route('/css/<path:filename>')
    def css_files(filename):
        return send_from_directory('static/css', filename)

    @app.route('/js/<path:filename>')
    def js_files(filename):
        return send_from_directory('static/js', filename)

    @app.route('/download/<filename>')
    def download_file(filename):
        """Provides a route to download processed files."""
        try:
            processed_dir = PATHS['processed']
            if not os.path.exists(os.path.join(processed_dir, filename)):
                return jsonify({'success': False, 'message': 'File not found'}), 404
            return send_from_directory(processed_dir, filename, as_attachment=True)
        except Exception as e:
            print(f"❌ Download error: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500

    # --- File Upload Routes ---
    @app.route('/upload_image', methods=['POST'])
    def upload_image():
        """Handles image uploads, processes them, and returns the result."""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'message': 'No file part in the request.'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'message': 'No file selected.'})

            if not allowed_file(file.filename, 'image'):
                return jsonify({'success': False, 'message': 'Invalid image file type.'})

            person_model = request.form.get('person_model', 'yolov8')
            face_model = request.form.get('face_model', 'insightface')

            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(PATHS['uploads'], filename)
            file.save(file_path)
            
            print(f"📁 Image uploaded: {file_path}")
            
            result = monitor_system.process_image(file_path, person_model, face_model)
            return jsonify(result)

        except Exception as e:
            print(f"❌ Image upload error: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500

    @app.route('/upload_video', methods=['POST'])
    def upload_video():
        """Handles video uploads and returns a confirmation to start processing via WebSocket."""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'message': 'No file part in the request.'})

            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'message': 'No file selected.'})

            if not allowed_file(file.filename, 'video'):
                return jsonify({'success': False, 'message': 'Invalid video file type.'})

            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(PATHS['uploads'], filename)
            file.save(file_path)

            print(f"📁 Video uploaded: {file_path}")

            # The client will trigger the processing via WebSocket
            return jsonify({
                'success': True,
                'message': 'Video uploaded successfully. Processing will start via WebSocket.',
                'file_path': file_path
            })

        except Exception as e:
            print(f"❌ Video upload error: {e}")
            return jsonify({'success': False, 'message': str(e)}), 500

    # --- WebSocket Event Handlers ---
    @socketio.on('connect')
    def handle_connect():
        print("🔗 WebSocket: Client connected")
        socketio.emit('status', {'message': 'Connected to AI Crowd Monitor'})

    @socketio.on('disconnect')
    def handle_disconnect():
        print("🔌 WebSocket: Client disconnected")

    @socketio.on('start_video_processing')
    def handle_start_video_processing(data):
        """Starts video processing in a background thread."""
        video_path = data.get('file_path')
        person_model = data.get('person_model', 'yolov8')
        face_model = data.get('face_model', 'insightface')

        if not video_path or not os.path.exists(video_path):
            socketio.emit('video_processing_complete', {'success': False, 'message': 'Video file not found.'})
            return

        print(f"📨 Starting video processing via WebSocket: {video_path}")

        # Run processing in a background thread to keep the server responsive
        def process_video_background():
            with app.app_context():
                monitor_system.process_video(video_path, person_model, face_model)
        
        thread = threading.Thread(target=process_video_background, daemon=True)
        thread.start()

    @socketio.on('stop_processing')
    def handle_stop_processing():
        """Stops any ongoing video processing."""
        print("📨 Stop processing request received via WebSocket.")
        monitor_system.stop_processing()
        socketio.emit('processing_stopped', {'message': 'Processing stopped by user.'})