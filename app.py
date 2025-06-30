"""
Main application file for the AI Crowd Monitoring System.
This file initializes the Flask application, SocketIO, and registers the routes.
"""

import os
import warnings
from flask import Flask
from flask_socketio import SocketIO

# --- Pre-initialization Setup ---

# 1. Suppress TensorFlow and other warnings for a cleaner console output.
# This is done before importing other modules that might trigger these warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')

# --- Application Imports ---

# Import configurations and core components
from src.utils.config import PATHS
from src.crowd_monitor import CrowdMonitoringSystem
from src.routes import register_routes

# --- Application Initialization ---

# 1. Initialize the Flask application.
# The static_folder is set to 'static' to serve frontend files like index.html, CSS, and JS.
app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Set max file upload size to 100MB

# 2. Initialize Flask-SocketIO for real-time communication with the frontend.
# CORS is allowed from all origins for development flexibility.
# Logging is disabled for a cleaner console, and buffer size is increased.
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False, max_http_buffer_size=100*1024*1024)

# 3. Create necessary directories for file uploads and processed outputs.
# These directories are defined in the config.py file.
os.makedirs(PATHS['uploads'], exist_ok=True)
os.makedirs(PATHS['processed'], exist_ok=True)

# --- System Components Setup ---

# 1. Create an instance of the main monitoring system.
# The socketio instance is passed to it, allowing it to send real-time updates.
monitor_system = CrowdMonitoringSystem(socketio)

# 2. Register all web routes (Flask) and event handlers (SocketIO).
# This keeps the main file clean by delegating route definitions to the routes.py module.
register_routes(app, socketio, monitor_system)

# --- Application Execution ---

if __name__ == '__main__':
    """
    Main entry point for running the Flask application.
    """
    print("🚀 Starting AI Crowd Monitoring System...")
    print("📋 Open your browser and navigate to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server.")
    
    # Run the application using the SocketIO server.
    # debug=False is recommended for production or when using background threads.
    # host='0.0.0.0' makes the server accessible on the local network.
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, log_output=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        monitor_system.stop_processing()