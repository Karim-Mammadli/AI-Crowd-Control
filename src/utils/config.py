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
        # General face detection settings
        'confidence_threshold': 0.1,  # Default confidence threshold
        
        # MediaPipe specific settings
        'mediapipe': {
            'confidence_threshold': 0.3,  # Lower confidence threshold for better face detection in crowds
            'model_selection': 1,         # 0 for short-range, 1 for full-range
        },
        
        # RetinaFace specific settings
        'retinaface': {
            'confidence_threshold': 0.5,  # Lower confidence threshold for RetinaFace
            'quality': 'normal',          # 'normal', 'high', 'low'
        },
    },
    
    # Video Processing Settings
    'video': {
        'fps': 30,                   # Target FPS for video processing
        'max_frame_size': (1920, 1080),  # Maximum frame size to process
        'batch_size': 10,            # Number of frames to process in batch
    },
    
    # Crowd Analysis Settings - Updated for larger crowds
    'crowd': {
        'density_thresholds': {
            'low': 5,                 # Increased: Number of people for LOW density
            'medium': 15,             # Increased: Number of people for MEDIUM density
            'high': 30,               # Increased: Number of people for HIGH density
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