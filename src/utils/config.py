"""
Configuration settings for the AI Crowd Control system.
Centralizes model parameters and settings for easy maintenance.
"""

# Model Settings
MODEL_CONFIG = {
    # YOLO Model Settings
    'yolo': {
        'model_path': 'yolov8n.pt',  # Path to YOLO model file
        'confidence_threshold': 0.3,  # Confidence threshold for person detection
        'batch_size': 10,            # Number of frames to process in batch
    },
    
    # Face Detection Settings
    'face': {
        'confidence_threshold': 0.1,  # Confidence threshold for face detection
        # 'min_face_size': 20,         # Minimum face size to detect
    },
    
    # Video Processing Settings
    'video': {
        'fps': 30,                   # Target FPS for video processing
        'max_frame_size': (1920, 1080),  # Maximum frame size to process
        'batch_size': 10,            # Number of frames to process in batch
    },
    
    # Crowd Analysis Settings
    'crowd': {
        'density_thresholds': {
            'low': 2,                # Number of people for LOW density
            'medium': 5,             # Number of people for MEDIUM density
            'high': 10,              # Number of people for HIGH density
        },
        'proximity_threshold': 0.15,  # Normalized distance for close interactions
    }
}

# File Paths
PATHS = {
    'uploads': 'uploads',
    'processed': 'processed',
    'demo': 'demo',
}

# Allowed File Extensions
ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm'},
    'image': {'jpg', 'jpeg', 'png', 'bmp', 'webp'},
}

# MLflow Settings
MLFLOW_CONFIG = {
    'experiment_name': 'ai-crowd-monitoring-hackathon',
    'model_name': 'ai-crowd-monitoring-system',
    'version': '1.0.0',
} 