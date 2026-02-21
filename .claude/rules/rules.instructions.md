# DDSH — Coding Guidelines for AI-Assisted Development

> These instructions define how AI should generate code, answer questions,
> and review changes for the **Driver Drowsiness Shield (DDSH)** project —
> a real-time drowsiness detection system built as a replica of the
> ROBOMECH Journal (2025) paper by Bhanja et al.
> This is a national project showcase submission by a B.Tech CSE student
> from ITER, SOA University, Bhubaneswar, Odisha, India.

---

## 1. Role & Context

- You are assisting a **final-year-level project developer** with intermediate
  Python and basic ML knowledge.
- The project uses **MobileNet Transfer Learning + OpenCV** to detect driver
  drowsiness via webcam in real time.
- All code must be **paper-accurate**, **demonstrable live**, and
  **explainable to judges** at a national showcase.
- Reference paper: *"Driver Drowsiness Shield (DDSH)"*, DOI:
  `10.1186/s40648-025-00307-4`

---

## 2. Language & Environment

- **Language:** Python 3.10+
- **Framework:** TensorFlow 2.x / Keras
- **CV Library:** OpenCV (`cv2`)
- **Target Hardware:** Standard laptop, 8GB RAM, CPU-compatible (no mandatory GPU)
- **IDE:** VS Code
- **Package Manager:** pip with `requirements.txt`

### Dependency Rules

- Always pin versions in `requirements.txt`. Example:
  ```
  tensorflow==2.13.0
  opencv-python==4.8.0.76
  numpy==1.24.3
  scikit-learn==1.3.0
  matplotlib==3.7.2
  seaborn==0.12.2
  pygame==2.5.2
  streamlit==1.27.0
  ```
- Never suggest conda-only packages unless a pip alternative exists.
- If a package requires a GPU, always provide a CPU fallback.

---

## 3. Project Structure Convention

All generated code must follow this exact folder structure:

```
ddsh/
├── config.py                  # All hyperparameters and paths
├── requirements.txt
├── README.md
├── data/
│   ├── train/
│   │   ├── Open_Eyes/
│   │   └── Closed_Eyes/
│   └── test/
│       ├── Open_Eyes/
│       └── Closed_Eyes/
├── model/
│   └── ddsh_mobilenet.keras   # Saved trained model
├── scripts/
│   ├── preprocess.py          # Dataset loading and preprocessing
│   ├── train.py               # Model training pipeline
│   ├── evaluate.py            # Metrics, confusion matrix, plots
│   └── detect.py              # Real-time webcam detection + alarm
├── assets/
│   └── alarm.wav              # Alert sound file
├── haarcascades/
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
└── outputs/
    ├── confusion_matrix.png
    ├── accuracy_plot.png
    └── loss_plot.png
```

- Never scatter files in the root directory.
- Never hardcode paths — always reference `config.py`.

---

## 4. config.py — Single Source of Truth

Every hyperparameter from the paper must live in `config.py`.
When generating any script, import from config — never inline magic numbers.

```python
# config.py — Paper-accurate hyperparameters (Bhanja et al., 2025)

IMG_SIZE          = 224          # Preprocessed image size (px)
ORIGINAL_SIZE     = 84           # MRL dataset original image size (px)
INPUT_SHAPE       = (224, 224, 3)
BATCH_SIZE        = 32
EPOCHS            = 5
LEARNING_RATE     = 0.001        # Adam default
VALIDATION_SPLIT  = 0.1
LOSS_FUNCTION     = "mse"        # Mean Squared Error as per paper
OPTIMIZER         = "adam"
MODEL_WEIGHTS     = "imagenet"
INCLUDE_TOP       = False

# Detection thresholds
CLOSED_EYE_FRAMES_THRESHOLD = 6     # ~2 sec at 30fps before alarm triggers
ALARM_COOLDOWN_SEC           = 3

# Paths
TRAIN_DIR   = "data/train"
TEST_DIR    = "data/test"
MODEL_PATH  = "model/ddsh_mobilenet.keras"
ALARM_PATH  = "assets/alarm.wav"
HAAR_FACE   = "haarcascades/haarcascade_frontalface_default.xml"
HAAR_EYE    = "haarcascades/haarcascade_eye.xml"
OUTPUT_DIR  = "outputs"
```

---

## 5. Code Style & Quality Rules

### General

- Follow **PEP 8** strictly.
- Max line length: **88 characters** (Black formatter standard).
- Use **type hints** on all function signatures.
- Every function must have a **docstring** explaining: what it does, args, returns.

```python
# ✅ Correct
def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load an image, convert to grayscale, resize to 224x224,
    convert back to RGB, and normalize pixel values to [0, 1].

    Args:
        img_path (str): Absolute path to the image file.

    Returns:
        np.ndarray: Preprocessed image array of shape (224, 224, 3).
    """

# ❌ Wrong
def prep(p):
    img = cv2.imread(p)
    return img
```

### Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Variables | snake_case | `closed_frame_count` |
| Functions | snake_case | `load_dataset()` |
| Classes | PascalCase | `DrowsinessDetector` |
| Constants | UPPER_SNAKE | `IMG_SIZE` |
| Files | snake_case | `train.py` |

---

## 6. Preprocessing Pipeline (Paper-Exact)

AI must always implement preprocessing in this exact sequence.
**Do not alter this order.** It is critical for MobileNet compatibility.

```
Raw Image (84×84, any format)
        ↓
1. Load with OpenCV
        ↓
2. Convert to Grayscale   → cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ↓
3. Resize to 224×224      → cv2.resize(gray, (224, 224))
        ↓
4. Convert back to RGB    → cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        ↓
5. Normalize to [0, 1]    → img_array / 255.0
        ↓
6. Expand dims            → np.expand_dims(img, axis=0) for inference
```

---

## 7. Model Architecture Rules

- Always use `tf.keras.applications.MobileNet` — **not MobileNetV2**.
- Always set `include_top=False` and `weights='imagenet'`.
- Always add a **Global Average Pooling** layer after the base model.
- Always use **Linear activation** on the output layer (regression-style score).
- Always use **MSE** as loss — not categorical crossentropy (paper-specific).
- Always freeze base model layers initially for faster convergence.

```python
# ✅ Correct model construction pattern
base_model = tf.keras.applications.MobileNet(
    weights=config.MODEL_WEIGHTS,
    include_top=config.INCLUDE_TOP,
    input_shape=config.INPUT_SHAPE
)
base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
```

---

## 8. Real-Time Detection Logic Rules

- Always check if `face_cascade.detectMultiScale()` returns results before proceeding.
- Always handle the **no-face-detected** case gracefully — display a status message, do not crash.
- Closed-eye detection must use a **frame counter**, not a simple boolean.
- Alarm must trigger only after `CLOSED_EYE_FRAMES_THRESHOLD` consecutive closed frames.
- Always implement a **cooldown** to prevent alarm looping.

```python
# ✅ Correct threshold logic
if eye_state == "CLOSED":
    closed_frame_count += 1
else:
    closed_frame_count = 0  # Reset on open eye

if closed_frame_count >= config.CLOSED_EYE_FRAMES_THRESHOLD:
    trigger_alarm()
    closed_frame_count = 0  # Reset after alarm
```

---

## 9. Error Handling Requirements

All scripts must handle:

| Scenario | Required Handling |
|---|---|
| Webcam not found | Raise clear error with message: `"Webcam not accessible. Check device connection."` |
| Model file missing | Raise `FileNotFoundError` with path hint |
| No face detected | Display `"No face detected"` on frame, continue loop |
| Dataset directory empty | Raise with directory path and expected structure |
| Low light / occlusion | Log warning, continue — do not crash |

- Always use `try/except` around webcam and file I/O operations.
- Never use bare `except:` — always catch specific exceptions.

---

## 10. Evaluation & Output Rules

After training, the evaluation script must produce:

- [ ] Confusion Matrix plot (saved to `outputs/confusion_matrix.png`)
- [ ] Accuracy vs Epochs line plot — Training + Validation (Fig. 7 replica)
- [ ] Loss vs Epochs line plot — Training + Validation (Fig. 8 replica)
- [ ] Accuracy per Epoch histogram (Fig. 9 replica)
- [ ] Console output of: Accuracy, Precision, Recall, F1-Score
- [ ] Sanity check: Print paper's expected values alongside actual values

```python
# ✅ Required console output format
print("=" * 40)
print("MODEL EVALUATION — DDSH (Bhanja et al., 2025)")
print("=" * 40)
print(f"Accuracy  : {accuracy:.4f}  | Paper: 0.9000")
print(f"Precision : {precision:.4f}  | Paper: 1.0000")
print(f"Recall    : {recall:.4f}  | Paper: 0.8330")
print(f"F1-Score  : {f1:.4f}  | Paper: 0.9090")
print("=" * 40)
```

---

## 11. Comments & Explainability

Since this is a **showcase project**, every non-trivial line needs a comment.
Judges will ask "what does this do?" — the code must answer that visually.

```python
# ✅ Good comment style for showcase
# Normalize pixel values to [0, 1] to stabilize gradient descent during training
img_array = img_array / 255.0

# Apply Haar Cascade to detect frontal faces in the current frame
# Returns a list of rectangles (x, y, w, h) for each detected face
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
```

- Do not comment obvious things like `# import numpy`.
- Do comment **why**, not just **what**.

---

## 12. Dataset Instructions

- **Source:** http://mrl.cs.vsb.cz/eyedataset (MRL Eyes 2018 — open source)
- **Classes:** `Open_Eyes` and `Closed_Eyes`
- **Expected structure after download:**
  ```
  data/train/Open_Eyes/    → ~1000+ images
  data/train/Closed_Eyes/  → ~1000+ images
  data/test/Open_Eyes/     → ~200+ images
  data/test/Closed_Eyes/   → ~200+ images
  ```
- Always shuffle dataset before splitting.
- Always maintain class balance — log class counts at start of training.

---

## 13. Git & Version Control

- Commit after each completed script with a clear message:
  ```
  git commit -m "feat: add preprocessing pipeline (paper-accurate)"
  git commit -m "feat: train MobileNet model — 5 epochs, MSE loss"
  git commit -m "feat: real-time detection with alarm trigger"
  git commit -m "docs: add README and evaluation plots"
  ```
- Never commit the `model/` folder — add to `.gitignore`, share via Google Drive link.
- Never commit `data/` — too large; document download steps in README.

---

## 14. README Requirements

The generated `README.md` must include:

- Project title + one-line description
- Paper citation (APA format)
- Demo GIF or screenshot placeholder
- Setup instructions (clone → install → download dataset → train → run)
- Results table matching the paper
- Known limitations
- Future work (as stated in paper: attention mechanisms, ADAS integration)

---

## 15. What AI Must NOT Do

- ❌ Do not use MobileNetV2, ResNet, or EfficientNet — paper specifies MobileNet
- ❌ Do not change loss to `categorical_crossentropy` — paper uses MSE
- ❌ Do not hardcode file paths — always use `config.py`
- ❌ Do not skip preprocessing steps or change their order
- ❌ Do not generate code without comments for showcase scripts
- ❌ Do not suggest paid APIs, cloud GPUs, or non-free tools
- ❌ Do not produce monolithic scripts — keep each concern in its own file
- ❌ Do not train for more than 5 epochs without flagging deviation from paper

---

## 16. Showcase-Specific Rules

- Every script must print a **startup banner** when run:
  ```
  ============================================
   DDSH — Driver Drowsiness Shield
   Real-Time Detection System
   Based on: Bhanja et al., ROBOMECH 2025
  ============================================
  ```
- The real-time window must show: eye state label, frame count, drowsiness status.
- All plots must have titles, axis labels, and legends — no bare figures.
- Keep a `demo_mode` flag in `config.py` that, when `True`, uses a pre-recorded
  video instead of webcam for risk-free presentation.

---

*Guidelines version: 1.0 | Project: DDSH Replica | Author: Vivek Ranjan Sahoo, ITER SOA*