"""
Configuration settings for face recognition attendance system
"""

# Camera Configuration
CAMERA = {
    'DEVICE_ID': 0,  # Default camera
    'RESOLUTION': (640, 480),
    'FPS': 30
}

# Face Detection Settings (Haar Cascade)
FACE_DETECTION = {
    'SCALE_FACTOR': 1.1,
    'MIN_NEIGHBORS': 4,  # Lower = more detections
    'MIN_SIZE': (50, 50),  # Minimum face size
    'FLAGS': 'CASCADE_SCALE_IMAGE'
}

# LBPH Face Recognition Settings
LBPH_SETTINGS = {
    'RADIUS': 2,
    'NEIGHBORS': 16,
    'GRID_X': 8,
    'GRID_Y': 8,
    'THRESHOLD': 150.0,  # More lenient threshold
    'CONFIDENCE_MINIMUM': 30.0  # Lower minimum
}

# DeepFace Settings
DEEPFACE_SETTINGS = {
    'MODEL_NAME': 'Facenet512',
    'DETECTOR_BACKEND': 'opencv',
    'DISTANCE_METRIC': 'cosine',
    'DISTANCE_THRESHOLD': 0.50,  # Optimized threshold
    'ENFORCE_DETECTION': True,
    'ALIGN': True
}

# Recognition Settings
RECOGNITION = {
    'CONFIDENCE_THRESHOLD': 40.0,  # Minimum confidence to recognize
    'RECOGNITION_BUFFER_SIZE': 5,  # Frames to buffer
    'RECOGNITION_CONSISTENCY': 2,  # Required consistent recognitions
    'AUTO_ATTENDANCE_CONFIDENCE': 60.0,  # Auto-mark threshold
    'COOLDOWN_SECONDS': 5  # Cooldown between marks
}

# File Paths
PATHS = {
    'DATA_DIR': 'data',
    'PHOTOS_DIR': 'data/student_photos',
    'ENCODINGS_DIR': 'data/encodings',
    'EXPORTS_DIR': 'webapp/exports',
    'STUDENTS_JSON': 'data/students.json',
    'ATTENDANCE_JSON': 'data/attendance.json',
    'TRAINED_MODEL': 'data/encodings/trained_model.yml',
    'DEEPFACE_MODEL': 'data/encodings/deepface_model.pkl',
    'LABELS_JSON': 'data/encodings/labels.json'
}

# Display Settings
DISPLAY = {
    'WINDOW_NAME': 'Face Recognition - Enhanced System',
    'FONT': 'FONT_HERSHEY_SIMPLEX',
    'RECOGNIZED_COLOR': (0, 255, 0),  # Green
    'UNKNOWN_COLOR': (0, 0, 255),  # Red
    'TEXT_COLOR': (255, 255, 255),  # White
    'BOX_THICKNESS': 3,
    'TEXT_THICKNESS': 2
}
