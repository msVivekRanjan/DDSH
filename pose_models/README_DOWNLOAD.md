# Pose Models Download Instructions

Option A — MMPose (recommended)

1. Install OpenMIM and required libraries:

```bash
pip install openmim
mim install mmengine mmcv mmdet mmpose
```

2. Download a whole-body checkpoint into this folder (pose_models/):

```bash
# example (adjust to the exact config/checkpoint you want):
# mim download mmpose --config dwpose_l_wholebody --dest pose_models/
```

Option B — ONNX (simpler, no mmpose)

1. Install runtime:

```bash
pip install onnxruntime
```

2. Download an ONNX whole-body model (example):

- https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx

Place the downloaded `*.onnx` file into this directory and update
`config.POSE_ONNX_PATH` if necessary.

Option C — MediaPipe (fallback, limited keypoints)

```bash
pip install mediapipe
```

MediaPipe requires no additional downloads; it is bundled in the package.

---

Notes
- Use CPU-only weights for demo. If your system supports GPU and you
  want faster inference, follow the MMPose instructions and use CUDA-enabled
  builds, but this is NOT required for the showcase.
- Keep model files under `pose_models/` and do not commit large binaries to git.
# End of file
