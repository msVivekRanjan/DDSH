"""
pose_detect.py — Real-time whole-body pose detection and visualization

Runs a pose estimator (mmpose/onnx/mediapipe) and draws color-coded skeletons.
Integration hook: if DDSH driver detector is present, overlay both outputs.
Features production-grade Telemetry UI Dashboard and Heuristic Tracking.

Keyboard controls:
 - q: quit
 - h: toggle heatmap
 - s: toggle skeleton
 - p: pause/unpause
 - v: toggle overlay/split view
"""

import os
import sys
import time
from typing import Tuple
from collections import deque
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


# --- HEURISTIC & UI FUNCTIONS ---

def analyze_posture(body_kp: np.ndarray, shrink_threshold: float = 20.0, lean_threshold: float = 40.0):
    try:
        nose = body_kp[config.BODY_REF_NOSE]
        left_hip = body_kp[config.BODY_REF_LEFT_HIP]
        right_hip = body_kp[config.BODY_REF_RIGHT_HIP]
        left_shoulder = body_kp[config.BODY_REF_LEFT_SHOULDER]
        right_shoulder = body_kp[config.BODY_REF_RIGHT_SHOULDER]

        hip_y_vals = [left_hip[1] if left_hip[1] > 0 else np.nan, right_hip[1] if right_hip[1] > 0 else np.nan]
        hip_y = np.nanmean(hip_y_vals)
        is_shrinking = False
        if not np.isnan(hip_y) and nose[1] > 0:
            is_shrinking = (nose[1] - hip_y) > shrink_threshold

        shoulder_x_vals = [left_shoulder[0] if left_shoulder[0] > 0 else np.nan, right_shoulder[0] if right_shoulder[0] > 0 else np.nan]
        center_x = np.nanmean(shoulder_x_vals)
        is_leaning = False
        if not np.isnan(center_x) and nose[0] > 0:
            is_leaning = abs(nose[0] - center_x) > lean_threshold

        return is_shrinking, is_leaning
    except Exception:
        return False, False

def check_hands_on_wheel(lhand: np.ndarray, rhand: np.ndarray, frame_height: int) -> bool:
    try:
        wheel_zone_y = frame_height * 0.4 
        l_visible = np.any(lhand[:, 1] > wheel_zone_y)
        r_visible = np.any(rhand[:, 1] > wheel_zone_y)
        return l_visible or r_visible
    except Exception:
        return True

def draw_telemetry_dashboard(frame, fps, latency, total_frames, alerts):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 140), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
    
    cv2.putText(frame, "POSE DIAGNOSTICS", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Latency: {latency:.1f} ms", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Frames: {total_frames}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    status_color = (0, 255, 0) if not alerts else (0, 0, 255)
    status_text = "SYSTEM NOMINAL" if not alerts else "BEHAVIOR DETECTED"
    cv2.putText(frame, status_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    if alerts:
        y_offset = 170
        for alert in alerts:
            cv2.rectangle(frame, (10, y_offset - 20), (320, y_offset + 5), (0, 0, 255), -1)
            cv2.putText(frame, alert, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 35
    return frame

# --- CORE VISUALIZATION ---

def draw_keypoints(frame: np.ndarray, kps: np.ndarray, color: Tuple[int, int, int], label: str = "") -> None:
    for (x, y, c) in kps:
        if c >= config.POSE_CONFIDENCE_THRESHOLD and x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    if label:
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def overlay_heatmaps(frame: np.ndarray, heatmaps: list, alpha: float = 0.5) -> np.ndarray:
    if not heatmaps: return frame
    combined = np.sum(np.stack(heatmaps, axis=0), axis=0)
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
    heat = (combined * 255).astype('uint8')
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)

def main():
    print("\n" + "=" * 60)
    print("DDSH — Whole-Body Pose Diagnostics [PRODUCTION UI]")
    print("=" * 60)

    loader = ModelLoader()
    if loader.backend is None:
        print("Pose model not available.")
        return

    splitter = KeypointRegionSplitter()

    if _HAS_DDSH:
        try:
            drowsy = DrowsinessDetector()
            print("✓ DDSH drowsiness detector integrated")
        except Exception:
            drowsy = None
    else:
        drowsy = None

    use_demo = config.DEMO_MODE and os.path.exists(config.POSE_DEMO_VIDEO)
    cap = cv2.VideoCapture(config.POSE_DEMO_VIDEO if use_demo else 0)
    if not cap.isOpened():
        print("❌ Could not open video source.")
        return

    paused = False
    show_heatmap = config.SHOW_HEATMAP
    show_skeleton = config.SHOW_SKELETON
    overlay_mode = True
    
    # Tracking metrics
    frame_idx = 0
    fps_smoothing = deque(maxlen=30)
    head_y_history = deque(maxlen=15)

    try:
        while True:
            if not paused:
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
                frame_idx += 1
                active_alerts = []

                results = None
                kps = np.zeros((config.BODY_KP + config.FACE_KP + 2*config.HAND_KP + config.FEET_KP, 3))
                if frame_idx % max(1, config.POSE_RUN_STRIDE) == 0:
                    try:
                        pred = loader.predict(frame)
                        if isinstance(pred, tuple):
                            kps, results = pred
                        else:
                            kps = pred
                    except Exception:
                        pass

                body, face, lhand, rhand, feet = splitter.split(kps)
                vis_frame = frame.copy()

                # Run Behavioral Heuristics for UI Alerts
                is_shrinking, is_leaning = analyze_posture(body)
                if is_shrinking: active_alerts.append("POSTURE: SHRINKING")
                if is_leaning: active_alerts.append("POSTURE: LEANING")

                try:
                    nose_y = body[config.BODY_REF_NOSE][1]
                    if nose_y > 0:
                        head_y_history.append(nose_y)
                        if len(head_y_history) == head_y_history.maxlen:
                            if max(head_y_history) - min(head_y_history) > 35:
                                active_alerts.append("DISTRACTION: HEAD BOBBING")
                except Exception:
                    pass

                if not check_hands_on_wheel(lhand, rhand, config.DISPLAY_HEIGHT):
                    active_alerts.append("HANDS OFF WHEEL")

                # Skeletons
                if show_skeleton:
                    if results is not None:
                        try:
                            mp = getattr(loader, 'mp', None)
                            if mp is None: import mediapipe as mp
                            dutils = mp.solutions.drawing_utils
                            
                            pose_spec_lm = dutils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                            pose_spec_conn = dutils.DrawingSpec(color=(0,180,0), thickness=2)
                            if results.pose_landmarks:
                                dutils.draw_landmarks(vis_frame, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS,
                                                      landmark_drawing_spec=pose_spec_lm, connection_drawing_spec=pose_spec_conn)
                            
                            hand_spec = dutils.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                            if results.left_hand_landmarks:
                                dutils.draw_landmarks(vis_frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                                      landmark_drawing_spec=hand_spec, connection_drawing_spec=hand_spec)
                            if results.right_hand_landmarks:
                                dutils.draw_landmarks(vis_frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS,
                                                      landmark_drawing_spec=hand_spec, connection_drawing_spec=hand_spec)
                            draw_keypoints(vis_frame, feet, (0, 0, 255))
                        except Exception:
                            draw_keypoints(vis_frame, body, (0, 255, 0))
                            draw_keypoints(vis_frame, lhand, (0, 255, 255))
                            draw_keypoints(vis_frame, rhand, (0, 255, 255))
                    else:
                        draw_keypoints(vis_frame, body, (0, 255, 0))
                        draw_keypoints(vis_frame, lhand, (0, 255, 255))
                        draw_keypoints(vis_frame, rhand, (0, 255, 255))

                # Heatmaps
                heatmaps = []
                if show_heatmap:
                    from pose_model import AdaptiveGaussianGenerator
                    gen = AdaptiveGaussianGenerator((config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH))
                    for i in range(kps.shape[0]):
                        x, y, c = kps[i]
                        if c >= config.POSE_CONFIDENCE_THRESHOLD and x > 0 and y > 0:
                            heatmaps.append(gen.generate_adaptive_heatmap((x, y), config.SIGMA_BASE))
                    vis_frame = overlay_heatmaps(vis_frame, heatmaps, alpha=0.45)

                # DDSH Integration
                if drowsy is not None:
                    try:
                        dd_frame_annotated, is_drowsy, _ = drowsy.detect_drowsiness(frame)
                        if is_drowsy: active_alerts.append("DROWSINESS DETECTED")
                        
                        if overlay_mode:
                            vis_frame = cv2.addWeighted(vis_frame, 0.6, dd_frame_annotated, 0.4, 0)
                        else:
                            h, w = frame.shape[:2]
                            half = w // 2
                            left = cv2.resize(dd_frame_annotated, (half, h))
                            right = cv2.resize(vis_frame, (w - half, h))
                            vis_frame = np.hstack([left, right])
                    except Exception:
                        pass

                # Performance Metrics & UI Dashboard
                loop_latency = (time.time() - loop_start) * 1000
                fps_smoothing.append(1000.0 / loop_latency if loop_latency > 0 else 0)
                avg_fps = sum(fps_smoothing) / len(fps_smoothing)

                vis_frame = draw_telemetry_dashboard(vis_frame, avg_fps, loop_latency, frame_idx, active_alerts)
                cv2.imshow("DDSH — Whole-Body Pose", vis_frame)

            # Keyboard handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('h'): show_heatmap = not show_heatmap
            if key == ord('s'): show_skeleton = not show_skeleton
            if key == ord('p'): paused = not paused
            if key == ord('v'): overlay_mode = not overlay_mode

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()