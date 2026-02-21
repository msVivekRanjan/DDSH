import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
detect.py — Real-Time Driver Drowsiness Detection with Alarm System

Live webcam-based detection system that:
1. Detects faces using Haar Cascade
2. Detects eyes within faces using Haar Cascade
3. Preprocesses detected eye regions (grayscale → resize → RGB → normalize)
4. Classifies eye state (open/closed) using trained model
5. Counts consecutive closed frames
6. Triggers alarm when threshold exceeded
7. Displays real-time status overlay on webcam feed

Reference: Bhanja et al., ROBOMECH Journal (2025)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import pygame
import config


class DrowsinessDetector:
    """
    Real-time drowsiness detection system using pre-trained MobileNet model
    and OpenCV Haar Cascades for face/eye detection.
    """

    def __init__(self, model_path: str = None, alarm_path: str = None):
        """
        Initialize detector with model and cascade classifiers.

        Args:
            model_path (str, optional): Path to trained model. Defaults to config.MODEL_PATH.
            alarm_path (str, optional): Path to alarm audio file. Defaults to config.ALARM_PATH.

        Raises:
            FileNotFoundError: If model or cascade files don't exist.
        """

        print("\n🔄 Initializing Drowsiness Detector...")
        print("=" * 60)

        self.model_path = model_path or config.MODEL_PATH
        self.alarm_path = alarm_path or config.ALARM_PATH

        # Load trained model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=config.LEARNING_RATE), loss=config.LOSS_FUNCTION, metrics=["accuracy"])
        print(f"  ✓ Model loaded from {self.model_path}")

        # Load Haar Cascade classifiers for face detection
        if not os.path.exists(config.HAAR_FACE):
            raise FileNotFoundError(f"Haar cascade not found: {config.HAAR_FACE}")
        self.face_cascade = cv2.CascadeClassifier(config.HAAR_FACE)
        print(f"  ✓ Face cascade loaded")

        # Load Haar Cascade classifiers for eye detection
        if not os.path.exists(config.HAAR_EYE):
            raise FileNotFoundError(f"Haar cascade not found: {config.HAAR_EYE}")
        self.eye_cascade = cv2.CascadeClassifier(config.HAAR_EYE)
        print(f"  ✓ Eye cascade loaded")

        # State tracking variables
        self.closed_frame_count = 0  # Consecutive frames with closed eyes
        self.alarm_active = False  # Is alarm currently playing?
        self.last_alarm_time = None  # Time when alarm was last triggered
        self.frame_count = 0  # Total frames processed
        self.fps_list = []  # For FPS calculation

        # Initialize Pygame for audio playback
        try:
            pygame.mixer.init()
            print(f"  ✓ Audio system initialized")
        except Exception as e:
            print(f"  ⚠️  Warning: Audio initialization failed: {e}")

        print("=" * 60 + "\n")

    def preprocess_eye_image(self, eye_roi: np.ndarray) -> np.ndarray:
        """
        Preprocess detected eye region for model inference.

        Pipeline:
        1. Convert to grayscale
        2. Resize to 224×224
        3. Convert back to RGB
        4. Normalize to [0, 1]

        Args:
            eye_roi (np.ndarray): Cropped eye region from frame (H, W, 3).

        Returns:
            np.ndarray: Preprocessed image ready for model (1, 224, 224, 3).
        """

        # Convert BGR to grayscale
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # Resize to 224×224 (MobileNet input size)
        resized = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE))

        # Convert back to RGB (3 channels)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Add batch dimension for model input: (224, 224, 3) → (1, 224, 224, 3)
        batch_input = np.expand_dims(normalized, axis=0)

        return batch_input

    def predict_eye_state(self, eye_roi: np.ndarray) -> tuple:
        """
        Predict eye state (open/closed) for a detected eye region.

        The model outputs a drowsiness score: [0, 1]
        - Score < threshold → Open eyes (0)
        - Score ≥ threshold → Closed eyes (1)

        Args:
            eye_roi (np.ndarray): Cropped eye region (H, W, 3 in BGR).

        Returns:
            tuple: (state_label, score, threshold)
                - state_label (str): "OPEN" or "CLOSED"
                - score (float): Raw model output [0, 1]
                - threshold (float): Decision threshold
        """

        try:
            # Preprocess eye image
            input_batch = self.preprocess_eye_image(eye_roi)

            # Get model prediction
            score = self.model.predict(input_batch, verbose=0)[0][0]

            # Apply threshold
            threshold = config.DROWSINESS_THRESHOLD
            state = "CLOSED" if score >= threshold else "OPEN"

            return state, float(score), threshold

        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return "UNKNOWN", 0.5, config.DROWSINESS_THRESHOLD

    def trigger_alarm(self):
        """
        Play alarm sound and set alarm state.

        Implements cooldown to prevent alarm spam.
        """

        if not os.path.exists(self.alarm_path):
            print(f"⚠️  Alarm file not found: {self.alarm_path}")
            print("   Using visual alert instead")
            return

        try:
            # Check if alarm is on cooldown
            if self.last_alarm_time is not None:
                time_since_alarm = (datetime.now() - self.last_alarm_time).total_seconds()
                if time_since_alarm < config.ALARM_COOLDOWN_SEC:
                    return  # Still in cooldown
            # Play alarm sound
            sound = pygame.mixer.Sound(self.alarm_path)
            sound.play()
            self.alarm_active = True
            self.last_alarm_time = datetime.now()
            print(f"🚨 ALARM TRIGGERED at frame {self.frame_count}")

        except Exception as e:
            print(f"⚠️  Audio playback error: {e}")

    def detect_drowsiness(self, frame: np.ndarray) -> tuple:
        """
        Detect drowsiness in a frame.

        Pipeline:
        1. Detect faces in frame
        2. For each face, detect eyes
        3. Classify eye state
        4. Track consecutive closed frames
        5. Trigger alarm if threshold exceeded

        Args:
            frame (np.ndarray): Input video frame (H, W, 3 in BGR).

        Returns:
            tuple: (frame_annotated, is_drowsy, frame_data)
                - frame_annotated: Frame with detection overlays
                - is_drowsy (bool): True if current eye state is closed
                - frame_data (dict): Detailed detection results
        """

        frame_annotated = frame.copy()
        is_drowsy = False
        frame_data = {
            "faces_detected": 0,
            "eyes_detected": 0,
            "eye_states": [],
            "closed_frame_count": self.closed_frame_count,
        }

        # Convert frame to grayscale for Haar cascade detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE,
        )

        frame_data["faces_detected"] = len(faces)

        if len(faces) == 0:
            # No face detected
            cv2.putText(
                frame_annotated,
                "No face detected",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 165, 255),  # Orange
                2,
            )
            self.closed_frame_count = 0  # Reset counter when face is lost

        else:
            # Process first detected face (primary driver)
            x, y, w, h = faces[0]

            # Draw face bounding box
            cv2.rectangle(frame_annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Detect eyes within face region
            face_roi = gray_frame[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=config.EYE_SCALE_FACTOR,
                minNeighbors=config.EYE_MIN_NEIGHBORS,
                minSize=config.EYE_MIN_SIZE,
            )

            frame_data["eyes_detected"] = len(eyes)

            if len(eyes) > 0:
                # Process all detected eyes (typically left + right)
                states = []

                for ex, ey, ew, eh in eyes:
                    # Convert eye region coordinates to full frame
                    eye_x = x + ex
                    eye_y = y + ey

                    # Extract eye region from original frame
                    eye_roi = frame[eye_y : eye_y + eh, eye_x : eye_x + ew]

                    if eye_roi.size == 0:
                        continue

                    # Predict eye state
                    state, score, _ = self.predict_eye_state(eye_roi)
                    states.append((state, score, (eye_x, eye_y, ew, eh)))

                    # Draw eye bounding box and label
                    color = config.ALARM_COLOR_BGR if state == "CLOSED" else config.NORMAL_COLOR_BGR
                    cv2.rectangle(
                        frame_annotated,
                        (eye_x, eye_y),
                        (eye_x + ew, eye_y + eh),
                        color,
                        2,
                    )
                    cv2.putText(
                        frame_annotated,
                        f"{state} ({score:.2f})",
                        (eye_x, eye_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                frame_data["eye_states"] = states

                # Determine overall eye state (closed if ANY eye is closed)
                is_drowsy = any(state[0] == "CLOSED" for state in states)

                # Update closed frame counter
                if is_drowsy:
                    self.closed_frame_count += 1
                else:
                    self.closed_frame_count = 0

            else:
                # Eyes detected in initial face scan but not found during detailed detection
                self.closed_frame_count = 0

        # Update counter display
        frame_data["closed_frame_count"] = self.closed_frame_count

        # Check if alarm threshold exceeded
        if self.closed_frame_count >= config.CLOSED_EYE_FRAMES_THRESHOLD:
            self.trigger_alarm()
            # Draw prominent alarm indicator
            cv2.putText(
                frame_annotated,
                "DROWSINESS ALERT!",
                (frame.shape[1] // 2 - 200, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                config.ALARM_COLOR_BGR,
                3,
            )

        return frame_annotated, is_drowsy, frame_data

    def draw_status_bar(self, frame: np.ndarray, frame_data: dict) -> np.ndarray:
        """
        Draw status bar at bottom of frame showing:
        - Frame count and FPS
        - Faces and eyes detected
        - Closed eye frame count / threshold
        - Current eye state

        Args:
            frame (np.ndarray): Frame to annotate.
            frame_data (dict): Detection results.

        Returns:
            np.ndarray: Annotated frame.
        """

        h, w = frame.shape[:2]
        bar_height = 80
        bar_color = (40, 40, 40)  # Dark gray

        # Draw semi-transparent background for status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), bar_color, -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Prepare status text
        status_lines = [
            f"Frame: {self.frame_count} | Faces: {frame_data['faces_detected']} | Eyes: {frame_data['eyes_detected']}",
            f"Closed Frames: {frame_data['closed_frame_count']}/{config.CLOSED_EYE_FRAMES_THRESHOLD}",
        ]

        # Draw text
        y_offset = h - 40
        for i, line in enumerate(status_lines):
            cv2.putText(
                frame,
                line,
                (10, y_offset - i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )

        return frame

    def run_webcam_detection(self, use_demo_video: bool = False):
        """
        Run real-time drowsiness detection on webcam or demo video.

        Args:
            use_demo_video (bool): If True, use config.DEMO_VIDEO_PATH instead of webcam.
        """

        print("\n" + "=" * 60)
        print("🎬 Starting Real-Time Drowsiness Detection")
        print("=" * 60)

        # Open video source (webcam or demo video)
        if use_demo_video and os.path.exists(config.DEMO_VIDEO_PATH):
            print(f"📹 Using demo video: {config.DEMO_VIDEO_PATH}")
            cap = cv2.VideoCapture(config.DEMO_VIDEO_PATH)
        else:
            print("📷 Opening webcam...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm-up: give camera time to initialize
        import time
        time.sleep(2)
        # Discard first few frames
        for _ in range(5):
            cap.read()

        if not cap.isOpened():
            print("❌ Error: Could not open video source. Check webcam connection.")
            return

        print("✓ Video source opened successfully")
        print("\n▶️  Press 'q' to quit")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("\n⚠️  End of video or lost connection")
                    break

                self.frame_count += 1

                # Resize frame for processing (optional for speed)
                frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

                # Detect drowsiness
                frame_annotated, is_drowsy, frame_data = self.detect_drowsiness(frame)

                # Draw status bar
                frame_annotated = self.draw_status_bar(frame_annotated, frame_data)

                # Display frame
                cv2.imshow("DDSH — Driver Drowsiness Detection", frame_annotated)

                # Print stats periodically
                if self.frame_count % config.STATS_PRINT_FREQ == 0:
                    print(
                        f"Frame {self.frame_count} | Closed: {frame_data['closed_frame_count']} | "
                        f"Faces: {frame_data['faces_detected']} | Eyes: {frame_data['eyes_detected']}"
                    )

                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n👋 Quitting...")
                    break

        except KeyboardInterrupt:
            print("\n⚠️  Detection interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            print(f"\n✅ Detection complete. Total frames: {self.frame_count}")


def main():
    """
    Main entry point for real-time detection.
    """

    print("\n" + "=" * 60)
    print("DDSH — Real-Time Driver Drowsiness Detection System")
    print("Based on: Bhanja et al., ROBOMECH Journal (2025)")
    print("=" * 60)

    try:
        # Check if running in demo mode
        use_demo = config.DEMO_MODE

        # Initialize detector
        detector = DrowsinessDetector()

        # Run detection
        detector.run_webcam_detection(use_demo_video=use_demo)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n📌 First-time setup required:")
        print("   1. Train model: python scripts/train.py")
        print("   2. Place alarm sound at: assets/alarm.wav")
        print("   3. Ensure Haar cascades exist in: haarcascades/")

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
