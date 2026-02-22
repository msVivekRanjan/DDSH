"""
pose_detect.py — Real-time whole-body pose detection and visualization

Runs a pose estimator (mmpose/onnx/mediapipe) and draws color-coded skeletons.
Integration hook: if DDSH driver detector is present, overlay both outputs.

Keyboard controls:
 - q: quit
 - h: toggle heatmap
 - s: toggle skeleton
 - p: pause/unpause
 - v: toggle overlay/split view

All configuration read from config.py.
"""

import os
import sys
import time
from typing import Tuple
import cv2
import numpy as np

# Ensure project root on path so we can import scripts.detect (DDSH)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config
from pose_model import ModelLoader, KeypointRegionSplitter

# Try to import DDSH drowsiness detector for integration
try:
    from scripts.detect import DrowsinessDetector
    _HAS_DDSH = True
except Exception:
    _HAS_DDSH = False


def draw_keypoints(frame: np.ndarray, kps: np.ndarray, color: Tuple[int, int, int], label: str = "") -> None:
    """Draw keypoints (x,y,conf) on frame with small circles and optional label."""
    h, w = frame.shape[:2]
    for (x, y, c) in kps:
        if c >= config.POSE_CONFIDENCE_THRESHOLD and x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    if label:
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def overlay_heatmaps(frame: np.ndarray, heatmaps: list, alpha: float = 0.5) -> np.ndarray:
    """Overlay a normalized combined heatmap onto the frame."""
    if not heatmaps:
        return frame
    combined = np.sum(np.stack(heatmaps, axis=0), axis=0)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    heat = (combined * 255).astype('uint8')
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)
    return overlay


def main():
    print("\n" + "=" * 60)
    print("DDSH — Whole-Body Pose Detection Module")
    print("Based on: Wei et al., Complex & Intelligent Systems (2025)")
    print("=" * 60)

    # Initialize pose model
    loader = ModelLoader()
    if loader.backend is None:
        print("Pose model not available. See pose_models/README_DOWNLOAD.md for instructions.")
        return

    splitter = KeypointRegionSplitter()

    # Initialize DDSH detector if available
    if _HAS_DDSH:
        try:
            drowsy = DrowsinessDetector()
            print("✓ DDSH drowsiness detector integrated")
        except Exception as e:
            print(f"⚠ Could not initialize DDSH detector: {e}")
            drowsy = None
    else:
        drowsy = None

    # Video source
    use_demo = config.DEMO_MODE and os.path.exists(config.POSE_DEMO_VIDEO)
    cap = cv2.VideoCapture(config.POSE_DEMO_VIDEO if use_demo else 0)
    if not cap.isOpened():
        print("❌ Could not open video source for pose detection.")
        return

    paused = False
    show_heatmap = config.SHOW_HEATMAP
    show_skeleton = config.SHOW_SKELETON
    overlay_mode = True
    frame_idx = 0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ End of stream or camera error")
                    break
                frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                frame_idx += 1

                # Only run pose every POSE_RUN_STRIDE frames for speed
                results = None
                if frame_idx % max(1, config.POSE_RUN_STRIDE) == 0:
                    try:
                        pred = loader.predict(frame)
                        if isinstance(pred, tuple):
                            kps, results = pred
                        else:
                            kps = pred
                    except Exception as e:
                        print(f"⚠ Pose prediction error: {e}")
                        kps = np.zeros((config.BODY_KP + config.FACE_KP + 2*config.HAND_KP + config.FEET_KP, 3))
                else:
                    kps = np.zeros((config.BODY_KP + config.FACE_KP + 2*config.HAND_KP + config.FEET_KP, 3))

                # Split into regions
                body, face, lhand, rhand, feet = splitter.split(kps)

                # Draw overlays
                vis_frame = frame.copy()

                if show_skeleton:
                    # If we have raw MediaPipe results, prefer the drawing utilities
                    if results is not None:
                        try:
                            mp = getattr(loader, 'mp', None)
                            if mp is None:
                                import mediapipe as mp
                            dutils = mp.solutions.drawing_utils
                            # Body (green)
                            pose_spec_lm = dutils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                            pose_spec_conn = dutils.DrawingSpec(color=(0,180,0), thickness=2)
                            if results.pose_landmarks:
                                dutils.draw_landmarks(vis_frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                                                      landmark_drawing_spec=pose_spec_lm,
                                                      connection_drawing_spec=pose_spec_conn)
                            # Face (cyan)
                            face_spec = dutils.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1)
                            if results.face_landmarks:
                                try:
                                    dutils.draw_landmarks(vis_frame, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                                          landmark_drawing_spec=face_spec, connection_drawing_spec=face_spec)
                                except Exception:
                                    # fallback to simple points later
                                    pass
                            # Hands (yellow)
                            hand_spec = dutils.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                            hand_conn = dutils.DrawingSpec(color=(0,200,200), thickness=2)
                            if results.left_hand_landmarks:
                                dutils.draw_landmarks(vis_frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                                      landmark_drawing_spec=hand_spec, connection_drawing_spec=hand_conn)
                            if results.right_hand_landmarks:
                                dutils.draw_landmarks(vis_frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                                      landmark_drawing_spec=hand_spec, connection_drawing_spec=hand_conn)
                            # Feet: mediapipe pose includes ankle/foot; we keep circle markers via draw_keypoints for feet slice
                            draw_keypoints(vis_frame, feet, (0, 0, 255), label="Feet")
                        except Exception as e:
                            # On any drawing issues, fallback to simple draw_keypoints
                            draw_keypoints(vis_frame, body, (0, 255, 0), label="Body")
                            draw_keypoints(vis_frame, face, (255, 255, 0), label="Face")
                            draw_keypoints(vis_frame, lhand, (0, 255, 255), label="Left Hand")
                            draw_keypoints(vis_frame, rhand, (0, 255, 255), label="Right Hand")
                            draw_keypoints(vis_frame, feet, (0, 0, 255), label="Feet")
                    else:
                        draw_keypoints(vis_frame, body, (0, 255, 0), label="Body")
                        draw_keypoints(vis_frame, face, (255, 255, 0), label="Face")
                        draw_keypoints(vis_frame, lhand, (0, 255, 255), label="Left Hand")
                        draw_keypoints(vis_frame, rhand, (0, 255, 255), label="Right Hand")
                        draw_keypoints(vis_frame, feet, (0, 0, 255), label="Feet")

                # Optional heatmap overlay: generate coarse per-keypoint gaussian approximations
                heatmaps = []
                if show_heatmap:
                    from pose_model import AdaptiveGaussianGenerator
                    gen = AdaptiveGaussianGenerator((config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH))
                    for i in range(kps.shape[0]):
                        x, y, c = kps[i]
                        if c >= config.POSE_CONFIDENCE_THRESHOLD and x > 0 and y > 0:
                            sigma = config.SIGMA_BASE
                            heat = gen.generate_adaptive_heatmap((x, y), sigma)
                            heatmaps.append(heat)
                    vis_frame = overlay_heatmaps(vis_frame, heatmaps, alpha=0.45)

                # Integration with DDSH drowsiness detector
                if drowsy is not None:
                    try:
                        dd_frame_annotated, is_drowsy, frame_data = drowsy.detect_drowsiness(frame)
                        # Overlay skeleton on top of dd_frame_annotated for unified view
                        if overlay_mode:
                            # blend the two frames
                            vis_frame = cv2.addWeighted(vis_frame, 0.6, dd_frame_annotated, 0.4, 0)
                        else:
                            # split view: left = drowsy, right = pose
                            h, w = frame.shape[:2]
                            half = w // 2
                            left = cv2.resize(dd_frame_annotated, (half, h))
                            right = cv2.resize(vis_frame, (w - half, h))
                            vis_frame = np.hstack([left, right])
                    except Exception as e:
                        print(f"⚠ Integration error with DDSH: {e}")

                # Draw FPS and controls help
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.imshow("DDSH — Whole-Body Pose", vis_frame)

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('h'):
                show_heatmap = not show_heatmap
            if key == ord('s'):
                show_skeleton = not show_skeleton
            if key == ord('p'):
                paused = not paused
            if key == ord('v'):
                overlay_mode = not overlay_mode

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
