"""
pose_model.py — Model loader and adaptive Gaussian wrapper for whole-body pose

This module provides lightweight classes to:
- Load a whole-body pose estimation model (mmpose / onnx / mediapipe fallback)
- Generate adaptive Gaussian heatmaps per paper (Wei et al., 2025)
- Split keypoints into regions and extract reference points

All paths and hyperparameters are read from config.py.
"""

from typing import Tuple, List, Optional
import os
import sys
import math
import numpy as np
import cv2
import config

try:
    import onnxruntime as ort
    _HAS_ONNX = True
except Exception:
    _HAS_ONNX = False

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

# PAPER CONNECTION: We provide an Adaptive Gaussian implementation
# that follows the high-level equations in Wei et al. (2025).


class ModelLoader:
    """
    Load a pose model using one of three backends: mmpose, onnxruntime, mediapipe.

    Args:
        model_type (str): One of 'mmpose', 'onnx', 'mediapipe'. If None, uses
            config.POSE_MODEL_TYPE and falls back automatically.
        onnx_path (str): Path to ONNX model (used if model_type=='onnx').

    Returns:
        Instance with `.predict(frame)` method returning keypoints array
        shape (K, 3) where K is total keypoints and columns are (x, y, conf).
    """

    def __init__(self, model_type: Optional[str] = None, onnx_path: Optional[str] = None):
        self.model_type = model_type or config.POSE_MODEL_TYPE
        self.onnx_path = onnx_path or config.POSE_ONNX_PATH
        self.session = None
        self.backend = None

        # Try to initialize in order: mmpose -> onnx -> mediapipe
        if self.model_type == "mmpose":
            try:
                # Lazy import to avoid hard dependency
                import mmpose
                self.backend = "mmpose"
                # For showcase we expect user to download appropriate checkpoint
                # but we don't attempt to programmatically instantiate mmpose here
                print(f"✓ mmpose available — use mmpose backend (ensure weights in pose_models/)")
            except Exception:
                print("⚠ mmpose not available, trying ONNX backend")
                self._try_onnx_or_mediapipe()
        else:
            self._try_onnx_or_mediapipe()

    def _try_onnx_or_mediapipe(self):
        if _HAS_ONNX and os.path.exists(self.onnx_path):
            try:
                self.session = ort.InferenceSession(self.onnx_path)
                self.backend = "onnx"
                print(f"✓ ONNX Runtime backend loaded: {self.onnx_path}")
                return
            except Exception as e:
                print(f"⚠ Failed to load ONNX model: {e}")

        if _HAS_MEDIAPIPE:
            try:
                self.backend = "mediapipe"
                # initialize mediapipe holistic for combined pose+face+hands
                self.mp = mp
                self._mp_holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=True,
                )
                print("✓ MediaPipe Holistic backend initialized (fallback)")
                return
            except Exception as e:
                print(f"⚠ Failed to initialize MediaPipe: {e}")

        # If none available, mark as unavailable
        self.backend = None
        print("❌ No pose backend available. Install mmpose or provide ONNX or install mediapipe.")

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Run pose estimation on a single BGR frame.

        Returns:
            np.ndarray: keypoints array with shape (K,3) => (x, y, conf).
                        If backend is mediapipe, returns mapped subset.
                        If no model, returns empty array.
        """
        if self.backend is None:
            raise RuntimeError("No pose backend available. See pose_models/README_DOWNLOAD.md")

        if self.backend == "onnx":
            # Simple ONNX runtime flow: resize frame, normalize, run session
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                # Many ONNX models expect specific input size; for simplicity, resize to 384x384
                input_img = cv2.resize(img, (384, 384)).astype(np.float32) / 255.0
                input_img = np.transpose(input_img, (2, 0, 1))[None, :]
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: input_img})
                # Postprocessing depends on model; here we attempt to parse a common format
                # PAPER CONNECTION: In practice use model-specific postprocessing (DARK, CPN etc.)
                # For demo we will create placeholder zeros
                K = config.BODY_KP + config.FACE_KP + 2 * config.HAND_KP + config.FEET_KP
                result = np.zeros((K, 3), dtype=np.float32)
                return result
            except Exception as e:
                print(f"⚠ ONNX prediction failed: {e}")
                return np.zeros((0, 3), dtype=np.float32)

        if self.backend == "mediapipe":
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_holistic.process(img_rgb)
                h, w = frame.shape[:2]
                # Map mediapipe landmarks to a simplified keypoint array
                # We'll produce a Kx3 array with available landmarks and zeros otherwise
                K = config.BODY_KP + config.FACE_KP + 2 * config.HAND_KP + config.FEET_KP
                kp = np.zeros((K, 3), dtype=np.float32)

                # MEDIAPIPE BODY (33) -> place into first BODY_KP slots where possible
                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        if i >= config.BODY_KP:
                            break
                        kp[i, 0] = lm.x * w
                        kp[i, 1] = lm.y * h
                        kp[i, 2] = lm.visibility

                # MEDIAPIPE FACE (468) -> sample nose at face start
                if results.face_landmarks:
                    # nose tip is ~1 in mediapipe face mesh
                    nose = results.face_landmarks.landmark[1]
                    face_start = config.BODY_KP
                    kp[face_start + 0, 0] = nose.x * w
                    kp[face_start + 0, 1] = nose.y * h
                    kp[face_start + 0, 2] = 1.0
                    # rest remain zeros (demo)

                # Hands: place wrist landmarks if available
                if results.left_hand_landmarks:
                    left_wrist = results.left_hand_landmarks.landmark[0]
                    left_hand_start = config.BODY_KP + config.FACE_KP
                    kp[left_hand_start + 0, 0] = left_wrist.x * w
                    kp[left_hand_start + 0, 1] = left_wrist.y * h
                    kp[left_hand_start + 0, 2] = 1.0

                if results.right_hand_landmarks:
                    right_wrist = results.right_hand_landmarks.landmark[0]
                    right_hand_start = config.BODY_KP + config.FACE_KP + config.HAND_KP
                    kp[right_hand_start + 0, 0] = right_wrist.x * w
                    kp[right_hand_start + 0, 1] = right_wrist.y * h
                    kp[right_hand_start + 0, 2] = 1.0

                # store last raw results for downstream drawing if needed
                self.last_results = results
                return kp, results

            except Exception as e:
                print(f"⚠ MediaPipe prediction error: {e}")
                return np.zeros((0, 3), dtype=np.float32)

        # mmpose backend simplified message; actual integration requires mmpose APIs
        if self.backend == "mmpose":
            # In a real setup, instantiate mmpose Model and run inference here
            print("⚠ mmpose backend chosen but runtime wrapper not implemented in demo.")
            K = config.BODY_KP + config.FACE_KP + 2 * config.HAND_KP + config.FEET_KP
            return np.zeros((K, 3), dtype=np.float32)

        return np.zeros((0, 3), dtype=np.float32)


class AdaptiveGaussianGenerator:
    """
    Generate adaptive Gaussian heatmaps per keypoint following the
    high-level equations from Wei et al. (2025).

    Methods implement three factors:
      - part scale factor (SF_j)
      - density factor (σ_p)
      - visibility factor (η_vis)

    The final sigma is combined multiplicatively with SIGMA_BASE.
    """

    def __init__(self, img_shape: Tuple[int, int]):
        self.height, self.width = img_shape

    def compute_part_scale_factor(self, part_bbox: Tuple[int, int, int, int], global_bbox: Tuple[int, int, int, int]) -> float:
        """
        Part scale factor proportional to sqrt(area_part / area_global).

        Args:
            part_bbox: (x, y, w, h) for the part
            global_bbox: (x, y, w, h) for whole person

        Returns:
            float: scale multiplier (SF_j)
        """
        pw = max(1, part_bbox[2])
        ph = max(1, part_bbox[3])
        gw = max(1, global_bbox[2])
        gh = max(1, global_bbox[3])
        area_part = pw * ph
        area_global = gw * gh
        sf = math.sqrt(area_part / max(area_global, 1))
        return float(max(0.5, min(sf, 4.0)))

    def compute_density_factor(self, keypoints_in_part: int, beta_md: float = 1.0) -> float:
        """
        Density factor increases sigma for sparse regions and decreases
        for very dense regions.

        Args:
            keypoints_in_part: number of keypoints in that part (e.g., 68 face)
            beta_md: density multiplier (tunable)

        Returns:
            float: density-based multiplier
        """
        # simple heuristic: denser parts -> smaller factor
        if keypoints_in_part <= 0:
            return 1.0
        factor = math.sqrt((self.width * self.height) / (config.ORIGINAL_SIZE * config.ORIGINAL_SIZE)) * beta_md
        # normalize roughly: larger keypoint count -> reduce factor
        factor = factor * (1.0 / math.sqrt(keypoints_in_part))
        return float(max(0.5, min(factor, 4.0)))

    def compute_visibility_factor(self, visibility: float, alpha: float = 0.5) -> float:
        """
        Visibility robustness factor: increases sigma when visibility low.

        Args:
            visibility: in [0, 1] where 1 is fully visible
            alpha: sensitivity hyperparameter

        Returns:
            float: visibility multiplier η_vis
        """
        v = float(max(0.0, min(1.0, visibility)))
        eta_vis = 1.0 + alpha * (1.0 - v)
        return float(max(0.5, min(eta_vis, 2.0)))

    def generate_adaptive_heatmap(self, keypoint: Tuple[float, float], sigma: float) -> np.ndarray:
        """
        Generate a 2D Gaussian heatmap centered at keypoint with given sigma.

        Args:
            keypoint: (x, y) pixel coordinates
            sigma: standard deviation in pixels

        Returns:
            np.ndarray: heatmap (H, W) float32
        """
        x, y = keypoint
        W, H = self.width, self.height
        xv = np.arange(0, W, 1, dtype=np.float32)
        yv = np.arange(0, H, 1, dtype=np.float32)
        xx, yy = np.meshgrid(xv, yv)
        # 2D Gaussian
        if sigma <= 0:
            sigma = config.SIGMA_BASE
        exponent = ((xx - x) ** 2 + (yy - y) ** 2) / (2 * (sigma ** 2))
        heatmap = np.exp(-exponent)
        # normalize to [0,1]
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.astype(np.float32)


class KeypointRegionSplitter:
    """
    Utility to split a Kx3 keypoint array into region subarrays.

    Returns tuples for (body, face, left_hand, right_hand, feet) each as
    (N_region, 3) arrays. Missing data are zeros.
    """

    def __init__(self):
        # compute offsets using config counts
        self.body_start = 0
        self.body_end = self.body_start + config.BODY_KP
        self.face_start = self.body_end
        self.face_end = self.face_start + config.FACE_KP
        self.lhand_start = self.face_end
        self.lhand_end = self.lhand_start + config.HAND_KP
        self.rhand_start = self.lhand_end
        self.rhand_end = self.rhand_start + config.HAND_KP
        self.feet_start = self.rhand_end
        self.feet_end = self.feet_start + config.FEET_KP

    def split(self, kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        K = kp.shape[0]
        def safe_slice(s, e):
            if s >= K:
                return np.zeros((e - s, 3), dtype=np.float32)
            return kp[s:min(e, K), :]

        body = safe_slice(self.body_start, self.body_end)
        face = safe_slice(self.face_start, self.face_end)
        lhand = safe_slice(self.lhand_start, self.lhand_end)
        rhand = safe_slice(self.rhand_start, self.rhand_end)
        feet = safe_slice(self.feet_start, self.feet_end)
        return body, face, lhand, rhand, feet


class ReferencePointExtractor:
    """
    Extract region reference points using configured indices.
    """

    def __init__(self):
        pass

    def extract_body_refs(self, body_kp: np.ndarray) -> dict:
        """
        Return left hip, right hip, nose coordinates if available.
        """
        refs = {}
        try:
            if body_kp.shape[0] > config.BODY_REF_LEFT_HIP:
                refs['left_hip'] = body_kp[config.BODY_REF_LEFT_HIP][:3]
            if body_kp.shape[0] > config.BODY_REF_RIGHT_HIP:
                refs['right_hip'] = body_kp[config.BODY_REF_RIGHT_HIP][:3]
            if body_kp.shape[0] > config.BODY_REF_NOSE:
                refs['nose'] = body_kp[config.BODY_REF_NOSE][:3]
        except Exception:
            pass
        return refs

    def extract_face_ref(self, face_kp: np.ndarray) -> Optional[Tuple[float, float, float]]:
        if face_kp.shape[0] > config.FACE_REF_NOSE:
            return tuple(face_kp[config.FACE_REF_NOSE][:3])
        return None

    def extract_hand_ref(self, hand_kp: np.ndarray) -> Optional[Tuple[float, float, float]]:
        if hand_kp.shape[0] > config.HAND_REF_WRIST:
            return tuple(hand_kp[config.HAND_REF_WRIST][:3])
        return None


# End of pose_model.py
