"""
config.py — Single Source of Truth for DDSH Hyperparameters

All hyperparameters and file paths are defined here to match the paper:
"Driver Drowsiness Shield (DDSH): A Real-Time Driver Drowsiness Detection System"
Published in ROBOMECH Journal (2025) by Bhanja et al.
DOI: 10.1186/s40648-025-00307-4

Every script imports from this file — never hardcode values.
"""

# ============================================================================
# MODEL ARCHITECTURE & TRAINING PARAMETERS (Paper-Exact)
# ============================================================================

# Image preprocessing pipeline
IMG_SIZE = 224  # Targets size for MobileNet input (pixels)
ORIGINAL_SIZE = 84  # MRL dataset native size (pixels)
INPUT_SHAPE = (224, 224, 3)  # Model input tensor shape (H, W, C)

# Training hyperparameters (exact from paper)
BATCH_SIZE = 32  # Training batch size per gradient update
EPOCHS = 5  # Total training epochs (paper uses 5)
LEARNING_RATE = 0.001  # Adam optimizer default learning rate
VALIDATION_SPLIT = 0.1  # 10% of training data reserved for validation
LOSS_FUNCTION = "mse"  # Mean Squared Error (regression-style, paper-specific)
OPTIMIZER = "adam"  # Adaptive Moment Estimation
MODEL_WEIGHTS = "imagenet"  # Pre-trained ImageNet weights for transfer learning
INCLUDE_TOP = False  # Exclude classification head from pre-trained model

# Real-time detection thresholds
CLOSED_EYE_FRAMES_THRESHOLD = 6  # ~2 seconds at 30fps (6 frames) before alarm
ALARM_COOLDOWN_SEC = 3  # Seconds to wait before alarm can trigger again
FPS_TARGET = 30  # Target frames per second for webcam capture

# ============================================================================
# FILE PATHS & DIRECTORIES
# ============================================================================

# Dataset directories
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Model and checkpoint paths
MODEL_PATH = "model/ddsh_mobilenet.keras"

# Audio alert file
ALARM_PATH = "assets/alarm.wav"

# Haar cascade classifiers (built-in OpenCV files)
HAAR_FACE = "haarcascades/haarcascade_frontalface_default.xml"
HAAR_EYE = "haarcascades/haarcascade_eye.xml"

# Output directory for evaluation plots and results
OUTPUT_DIR = "outputs"

# ============================================================================
# HAAR CASCADE PARAMETERS (Face & Eye Detection)
# ============================================================================

# Face detection parameters
FACE_SCALE_FACTOR = 1.1  # Scale reduction per detector pass
FACE_MIN_NEIGHBORS = 5  # Minimum detection clusters before acceptance
FACE_MIN_SIZE = (30, 30)  # Minimum face bounding box size (pixels)

# Eye detection parameters
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 5
EYE_MIN_SIZE = (20, 20)

# ============================================================================
# MODEL INFERENCE PARAMETERS
# ============================================================================

# Drowsiness score threshold (model output range: [0, 1])
# Scores > threshold indicate drowsiness (closed eyes)
DROWSINESS_THRESHOLD = 0.5

# ============================================================================
# DISPLAY & UI PARAMETERS
# ============================================================================

# Webcam display resolution (real-time detection window)
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Font and color settings for OpenCV display
FONT_FACE = 2  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
FONT_COLOR_RGB = (0, 255, 0)  # Green (BGR in OpenCV: (0, 255, 0))
ALARM_COLOR_BGR = (0, 0, 255)  # Red (for alarm state)
NORMAL_COLOR_BGR = (0, 255, 0)  # Green (for normal state)

# ============================================================================
# DEMO MODE (for risk-free presentations)
# ============================================================================

# Set to True to use a pre-recorded video instead of live webcam
DEMO_MODE = False
DEMO_VIDEO_PATH = "demo.mp4"  # Path to pre-recorded video for demo

# ============================================================================
# LOGGING & DEBUG
# ============================================================================

# Enable verbose logging during training/inference
DEBUG_VERBOSE = True

# Print inference statistics every N frames
STATS_PRINT_FREQ = 30  # Print stats every 30 frames (~1 sec at 30fps)

# ============================================================================
# EXPECTED RESULTS FROM PAPER (Sanity Check Values)
# ============================================================================

# These are the target metrics from Bhanja et al. (2025)
# Use these to verify training convergence and catch issues early
PAPER_ACCURACY = 0.9000  # 90%
PAPER_PRECISION = 1.0000  # 100%
PAPER_RECALL = 0.8330  # 83.3%
PAPER_F1_SCORE = 0.9090  # F1 = 2*(P*R)/(P+R)

# Confusion matrix from paper (micro test set)
PAPER_TP = 5  # True Positives (closed eyes correctly detected)
PAPER_TN = 4  # True Negatives (open eyes correctly detected)
PAPER_FP = 0  # False Positives
PAPER_FN = 1  # False Negatives

print("""
╔════════════════════════════════════════════════════════════════╗
║       DDSH Configuration Loaded Successfully                   ║
║  Driver Drowsiness Shield — Bhanja et al. (ROBOMECH 2025)     ║
║  Model: MobileNet | Loss: MSE | Epochs: 5 | Batch: 32        ║
╚════════════════════════════════════════════════════════════════╝
""")
