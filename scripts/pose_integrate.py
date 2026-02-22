"""
pose_integrate.py — Combine DDSH drowsiness detection and whole-body pose

Runs both systems and provides multimodal alerts:
 - If eyes closed threshold exceeded AND pose shows slumping -> multi-modal alert
 - Flash red overlay on skeleton when drowsiness+posture detected

Include a DEMO CHECKLIST at top for showcase day.
"""

# DEMO CHECKLIST:
# 1. Ensure model/ddsh_mobilenet.keras exists (run scripts/train.py)
# 2. Ensure pose model is available (pose_models/ or mediapipe)
# 3. Place alarm.wav in assets/
# 4. Test: python scripts/pose_integrate.py

import os
import sys
import time
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from pose_model import ModelLoader, KeypointRegionSplitter

try:
    from scripts.detect import DrowsinessDetector
    _HAS_DDSH = True
except Exception:
    _HAS_DDSH = False


def is_slumped(body_kp: np.ndarray, threshold: float = 20.0) -> bool:
    """
    Simple heuristic: compare nose y to average hip y; if nose is lower (greater y)
    than hips by threshold pixels, consider slumped.

    Args:
        body_kp: (N,3) body keypoints
        threshold: pixel delta above which we flag slumping

    Returns:
        bool
    """
    try:
        nose = body_kp[config.BODY_REF_NOSE]
        left_hip = body_kp[config.BODY_REF_LEFT_HIP]
        right_hip = body_kp[config.BODY_REF_RIGHT_HIP]
        vals = [left_hip[1] if left_hip[1]>0 else np.nan, right_hip[1] if right_hip[1]>0 else np.nan]
        if all(np.isnan(vals)):
            return False
        hip_y = np.nanmean(vals)
        if np.isnan(hip_y) or nose[1] <= 0:
            return False
        return (nose[1] - hip_y) > threshold
    except Exception:
        return False


def main():
    print("\n" + "="*60)
    print("DDSH — Integrated Drowsiness + Pose System")
    print("Combine eye-based drowsiness and whole-body posture for robust alerts")
    print("="*60)

    # Initialize components
    pose_loader = ModelLoader()
    if pose_loader.backend is None:
        print("Pose backend not available; run pose_evaluate.py for synthetic outputs instead.")

    splitter = KeypointRegionSplitter()

    drowsy = None
    if _HAS_DDSH:
        try:
            drowsy = DrowsinessDetector()
            print("✓ DDSH detector ready")
        except Exception as e:
            print(f"⚠ DDSH init failed: {e}")
            drowsy = None

    cap = cv2.VideoCapture(config.POSE_DEMO_VIDEO if (config.DEMO_MODE and os.path.exists(config.POSE_DEMO_VIDEO)) else 0)
    cap = cv2.VideoCapture(config.POSE_DEMO_VIDEO if (config.DEMO_MODE and os.path.exists(config.POSE_DEMO_VIDEO)) else 0)
    if not cap.isOpened():
        print("❌ Could not open video source")
        return

    # Warm-up: give camera time to initialize on macOS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("⏳ Warming up camera...")
    time.sleep(2)
    for _ in range(10):
        cap.read()
    print("✓ Camera ready\n▶️  Press 'q' to quit")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
            frame_idx += 1

            # Run drowsiness every frame, pose maybe every POSE_RUN_STRIDE
            is_drowsy = False
            dd_overlay = None
            if drowsy is not None:
                try:
                    dd_frame, is_drowsy, _ = drowsy.detect_drowsiness(frame)
                    dd_overlay = dd_frame
                except Exception as e:
                    print(f"⚠ DDSH runtime error: {e}")

            kp = np.zeros((config.BODY_KP + config.FACE_KP + 2*config.HAND_KP + config.FEET_KP, 3))
            if pose_loader.backend is not None and (frame_idx % max(1, config.POSE_RUN_STRIDE) == 0):
                try:
                    pred = pose_loader.predict(frame)
                    if isinstance(pred, tuple):
                        kp, _ = pred
                    else:
                        kp = pred
                except Exception as e:
                    print(f"⚠ Pose prediction failed: {e}")

            body, face, lhand, rhand, feet = splitter.split(kp)

            # Check posture
            slump = is_slumped(body)

            # Multi-modal alert logic
            alert_msg = None
            if is_drowsy and slump:
                alert_msg = "DROWSY + POSTURE ALERT"
                # flash red overlay
                overlay = frame.copy()
                overlay[:] = (0,0,255)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            elif is_drowsy:
                alert_msg = "DROWSINESS ALERT"
            elif slump:
                alert_msg = "POSTURE ALERT"

            # Draw skeleton (simple)
            from pose_detect import draw_keypoints
            if body is not None:
                draw_keypoints(frame, body, (0,255,0))
            if face is not None:
                draw_keypoints(frame, face, (255,255,0))

            # Overlay DDSH frame if available
            if dd_overlay is not None:
                frame = cv2.addWeighted(frame, 0.7, dd_overlay, 0.3, 0)

            if alert_msg:
                cv2.putText(frame, alert_msg, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            cv2.imshow('DDSH + Pose Integrated', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
