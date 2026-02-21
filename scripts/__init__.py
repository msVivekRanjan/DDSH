"""
DDSH Package — Driver Drowsiness Shield Detection System

Submodules:
- preprocess: Data loading and preprocessing pipeline
- train: MobileNet model training
- evaluate: Model evaluation and metrics
- detect: Real-time webcam detection with alarm

Usage:
    from scripts.preprocess import load_dataset_from_directory
    from scripts.train import create_model, train_model
    from scripts.evaluate import predict_on_dataset
    from scripts.detect import DrowsinessDetector
"""

__version__ = "1.0.0"
__author__ = "Vivek Ranjan Sahoo"
__email__ = "vivek@example.com"
__description__ = "Driver Drowsiness Shield — Real-Time Detection System (ROBOMECH 2025)"

__all__ = [
    "preprocess",
    "train",
    "evaluate",
    "detect",
]
