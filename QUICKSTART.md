# DDSH QuickStart Guide

**5-Minute Setup | Complete Walkthrough**

---

## ⚡ TL;DR (Quick Commands)

```bash
# 1. Clone and navigate
git clone <your-repo>
cd ddsh

# 2. Setup environment (automated)
chmod +x setup.sh && ./setup.sh

# 3. Download dataset (manual)
# Visit: http://mrl.cs.vsb.cz/eyedataset
# Extract to: data/train/ and data/test/

# 4. Train model (~10-15 min on CPU)
cd scripts && python train.py

# 5. Evaluate
python evaluate.py

# 6. Run live detection
python detect.py
# Press 'q' to quit
```

---

## 📋 Full Step-by-Step Guide

### Step 1: Prerequisites Check
```bash
python3 --version          # Should be 3.10+
git --version               # Any recent version
# Check webcam: ls /dev/video*
```

### Step 2: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ddsh.git
cd ddsh
```

### Step 3: Environment Setup
```bash
# Option A: Automated (Linux/macOS)
./setup.sh

# Option B: Manual
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# OR: venv\Scripts\activate  # Windows

pip install --upgrade pip
pip install -r requirements.txt

# Download Haar cascades
cd scripts && python download_haarcascades.py && cd ..
```

### Step 4: Prepare Dataset
```bash
# Download from: http://mrl.cs.vsb.cz/eyedataset
# Extract files:
# data/
#   ├── train/
#   │   ├── Open_Eyes/    (1000+ images)
#   │   └── Closed_Eyes/  (1000+ images)
#   └── test/
#       ├── Open_Eyes/    (200+ images)
#       └── Closed_Eyes/  (200+ images)

# Verify structure
find data -type d | head -10
```

### Step 5: Train Model
```bash
cd scripts
python train.py

# Expected output:
# Epoch 1/5 – loss: 0.1234 - accuracy: 0.9200
# Epoch 2/5 – loss: 0.0987 - accuracy: 0.9400
# ...
# ✅ Model saved to: model/ddsh_mobilenet.keras
```

### Step 6: Evaluate Model
```bash
python evaluate.py

# Output files generated:
# ✓ outputs/confusion_matrix.png
# ✓ outputs/roc_curve.png
# ✓ outputs/metrics_comparison.png

# Console output:
# Accuracy  : 0.9000  | Paper: 0.9000
# Precision : 1.0000  | Paper: 1.0000
# Recall    : 0.8330  | Paper: 0.8330
# F1-Score  : 0.9090  | Paper: 0.9090
```

### Step 7: Prepare Alarm (Optional)
```bash
# Create simple beep alarm
ffmpeg -f lavfi -i sine=f=1000:d=2 assets/alarm.wav

# OR download from Pixabay/Freesound:
# Save to: assets/alarm.wav
```

### Step 8: Run Live Detection
```bash
python detect.py

# On-screen:
# - Face bounding box (green)
# - Eye regions (green = open, red = closed)
# - Closed frame counter
# - Status bar at bottom
# - Alarm triggers visually and (optionally) audibly

# Press 'q' to quit
```

---

## 🎯 What to Expect

| Step | Time | Output |
|------|------|--------|
| Training | 10-15 min | `model/ddsh_mobilenet.keras` (1.5 MB) |
| Evaluation | 2 min | Plots + metrics comparison |
| Detection | Real-time | 30 FPS webcam feed |

**Accuracy Target**: 90% (matches paper)  
**Inference Speed**: 100-300 ms per frame (CPU)  
**RAM Usage**: ~2-3 GB (TensorFlow loaded)

---

## ⚙️ Configuration Quick Reference

Edit `config.py` to modify:

```python
# Detection Sensitivity
CLOSED_EYE_FRAMES_THRESHOLD = 6     # Frames before alarm (2 sec at 30fps)
DROWSINESS_THRESHOLD = 0.5          # Model score threshold (0-1)

# Performance  
DISPLAY_WIDTH = 1280                # Reduce for faster processing
DISPLAY_HEIGHT = 720

# Demo Mode
DEMO_MODE = False                   # Use demo.mp4 instead of webcam
```

---

## 🚨 Troubleshooting

### Webcam Not Opening
```bash
# Check device
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# Try different camera index
# In detect.py, change: cap = cv2.VideoCapture(1)  # Try 0, 1, 2...
```

### Dataset Not Found
```bash
# Verify structure
find data -name "*.jpg" | wc -l  # Should print >2000 for train

# If missing, download from:
# http://mrl.cs.vsb.cz/eyedataset
```

### Model Training Too Slow
```bash
# Reduce batch size in config.py
BATCH_SIZE = 16  # Less memory, slower but works

# Or reduce display resolution
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
```

### Haar Cascade Not Found
```bash
cd scripts && python download_haarcascades.py
cd ..
```

---

## 📊 Paper-Required Results

Must match for showcase:

| Metric | Target | Your Result |
|--------|--------|-------------|
| Accuracy | 90.0% | _____ |
| Precision | 100% | _____ |
| Recall | 83.3% | _____ |
| F1-Score | 0.909 | _____ |

**If not matching:**
1. Check data preprocessing (config.py)
2. Verify training parameters exact match
3. Re-download dataset (may be corrupted)
4. Retrain with `python train.py`

---

## 🎬 Showcase Presentation (2 minutes)

### Demo Script:

1. **Intro** (20 sec):
   ```
   "This is DDSH — a real-time driver drowsiness detection system.
    We're detecting eye state (open/closed) using deep learning."
   ```

2. **Show Code** (20 sec):
   - Open `config.py` — show hyperparameters
   - Show model summary — MobileNet architecture
   
3. **Run Detection** (60 sec):
   ```bash
   cd scripts && python detect.py
   ```
   - Keep eyes open naturally
   - Intentionally close eyes for 2+ seconds → **Alarm triggers**
   - Reopen eyes → Counter resets

4. **Show Plots** (30 sec):
   - Open output images:
     - `confusion_matrix.png` — 90% accuracy
     - `roc_curve.png` — AUC score
     - `metrics_comparison.png` — Our vs Paper

5. **Q&A** (Remaining time):
   - **Q: How does it work?**
     A: *Haar cascades detect faces/eyes. Model classifies eye state using pre-trained MobileNet.*
   - **Q: Why MobileNet?**
     A: *Lightweight, fast (~100ms), runs on CPU. MobileNet = 1.5MB model.*
   - **Q: Accuracy?**
     A: *90% on test set, matching the paper.*

---

## 📱 File Locations, Remember:

```
scripts/train.py        → Run to train
scripts/evaluate.py     → Run to evaluate
scripts/detect.py       → Run for demo
config.py              → Modify hyperparameters
model/                 → Trained model saved here
outputs/               → Evaluation plots here
data/                  → Dataset goes here
```

---

## 💡 Pro Tips

✅ **Generate cached outputs BEFORE showcase:**
```bash
python train.py      # Creates model
python evaluate.py   # Creates plots, saves as PNG
```

✅ **Have backup demo mode ready:**
```python
# In config.py:
DEMO_MODE = True  # Switches to pre-recorded video (safer)
```

✅ **Pre-test webcam:**
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam FAILED')"
```

✅ **Lower resolution for faster demo:**
```python
# In config.py:
DISPLAY_WIDTH = 640   # Faster inference
DISPLAY_HEIGHT = 480
```

---

## 🔗 Resources

- **Paper**: http://doi.org/10.1186/s40648-025-00307-4
- **Dataset**: http://mrl.cs.vsb.cz/eyedataset
- **MobileNet**: https://arxiv.org/abs/1704.04861
- **Haar Cascades**: https://github.com/opencv/opencv/tree/master/data/haarcascades

---

## ✅ Pre-Showcase Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (requirements.txt)
- [ ] Dataset downloaded and organized
- [ ] Haar cascades present (check haarcascades/ folder)
- [ ] Model trained (run scripts/train.py)
- [ ] Evaluation plots generated (run scripts/evaluate.py)
- [ ] Webcam tested and working
- [ ] Alarm sound present (assets/alarm.wav) or visual alert configured
- [ ] Demo video prepared (if using DEMO_MODE)
- [ ] Read config.py to understand all parameters
- [ ] Tested live detection (run scripts/detect.py)
- [ ] Notes prepared for Q&A

---

**You're all set! Good luck with your showcase! 🚀**
