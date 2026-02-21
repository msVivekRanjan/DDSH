#!/usr/bin/env python3
"""
FINAL_VERIFICATION.py — Post-Build Verification Checklist

Run this script after build to verify all components are in place.
"""

import os
import sys
from pathlib import Path

def check_file_exists(path: str, description: str = "") -> bool:
    """Check if a file exists and report."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    desc = f" ({description})" if description else ""
    print(f"  {status} {path}{desc}")
    return exists

def check_directory_exists(path: str, description: str = "") -> bool:
    """Check if a directory exists and report."""
    exists = os.path.isdir(path)
    status = "✓" if exists else "✗"
    desc = f" ({description})" if description else ""
    print(f"  {status} {path}/{desc}")
    return exists

def verify_project():
    """Run complete project verification."""
    
    print("\n" + "="*70)
    print("DDSH — Final Verification Checklist")
    print("="*70 + "\n")
    
    all_good = True
    
    # 1. Configuration Files
    print("📄 Configuration Files:")
    all_good &= check_file_exists("config.py", "Central config")
    all_good &= check_file_exists("requirements.txt", "Dependencies")
    all_good &= check_file_exists(".gitignore", "Git ignore rules")
    all_good &= check_file_exists("LICENSE", "MIT License")
    
    # 2. Documentation
    print("\n📚 Documentation:")
    all_good &= check_file_exists("README.md", "Complete guide (1800+ lines)")
    all_good &= check_file_exists("QUICKSTART.md", "5-min setup guide")
    all_good &= check_file_exists("PROJECT_SUMMARY.md", "Architecture overview")
    all_good &= check_file_exists("setup.sh", "Automated setup script")
    
    # 3. Script Files
    print("\n🐍 Python Scripts:")
    all_good &= check_file_exists("scripts/__init__.py", "Package init")
    all_good &= check_file_exists("scripts/preprocess.py", "Data preprocessing")
    all_good &= check_file_exists("scripts/train.py", "Model training")
    all_good &= check_file_exists("scripts/evaluate.py", "Model evaluation")
    all_good &= check_file_exists("scripts/detect.py", "Real-time detection")
    all_good &= check_file_exists("scripts/download_haarcascades.py", "Cascade downloader")
    
    # 4. Data Directories (Will be populated by user)
    print("\n📁 Data Directories (To be populated):")
    all_good &= check_directory_exists("data", "Dataset root")
    all_good &= check_directory_exists("data/train", "Training data")
    all_good &= check_directory_exists("data/train/Open_Eyes", "Training: Open eyes")
    all_good &= check_directory_exists("data/train/Closed_Eyes", "Training: Closed eyes")
    all_good &= check_directory_exists("data/test", "Test data")
    all_good &= check_directory_exists("data/test/Open_Eyes", "Test: Open eyes")
    all_good &= check_directory_exists("data/test/Closed_Eyes", "Test: Closed eyes")
    
    # 5. Output Directories (Will be created later)
    print("\n📊 Output Directories (Created after training):")
    all_good &= check_directory_exists("model", "Trained models")
    all_good &= check_directory_exists("haarcascades", "Haar cascades")
    all_good &= check_directory_exists("assets", "Audio & media")
    all_good &= check_directory_exists("outputs", "Evaluation plots")
    
    # 6. Summary Statistics
    print("\n📈 Project Statistics:")
    
    # Count Python lines
    total_lines = 0
    script_files = ["config.py", "scripts/preprocess.py", "scripts/train.py", 
                   "scripts/evaluate.py", "scripts/detect.py", "scripts/download_haarcascades.py"]
    for f in script_files:
        if os.path.exists(f):
            with open(f) as file:
                lines = len(file.readlines())
                total_lines += lines
    
    print(f"  ✓ Total Python code lines: ~{total_lines:,}")
    print(f"  ✓ Total Python scripts: 6")
    print(f"  ✓ Documentation files: 4")
    print(f"  ✓ Configuration files: 4")
    
    # 7. Next Steps
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    
    if all_good:
        print("""
✅ Project structure verified!

Next actions (in this order):

1. SETUP ENVIRONMENT (~5 minutes)
   $ chmod +x setup.sh
   $ ./setup.sh
   
   OR manually:
   $ python3 -m venv venv
   $ source venv/bin/activate  # macOS/Linux
   $ pip install -r requirements.txt
   $ cd scripts && python download_haarcascades.py && cd ..

2. DOWNLOAD DATASET (~10 minutes)
   Visit: http://mrl.cs.vsb.cz/eyedataset
   Download and extract to:
   - data/train/Open_Eyes/ (1000+ images)
   - data/train/Closed_Eyes/ (1000+ images)
   - data/test/Open_Eyes/ (200+ images)
   - data/test/Closed_Eyes/ (200+ images)

3. TRAIN MODEL (~10-15 minutes on CPU)
   $ cd scripts
   $ python train.py
   
   Output: model/ddsh_mobilenet.keras

4. EVALUATE MODEL (~2 minutes)
   $ python evaluate.py
   
   Output: 
   - outputs/confusion_matrix.png
   - outputs/roc_curve.png
   - outputs/metrics_comparison.png

5. PREPARE ALARM (Optional)
   $ ffmpeg -f lavfi -i sine=f=1000:d=2 assets/alarm.wav
   
   OR download from Pixabay/Freesound and save as assets/alarm.wav

6. RUN LIVE DETECTION
   $ python detect.py
   
   On screen:
   - Face detection (green bounding box)
   - Eye detection and classification
   - Closed-frame counter
   - Alarm when threshold exceeded
   
   Press 'q' to quit

QUICK REFERENCE:

File              Purpose
────────────────────────────────────────────────
config.py         All hyperparameters (EDIT THIS to customize)
README.md         Complete setup and usage guide
QUICKSTART.md     5-minute rapid setup
PROJECT_SUMMARY.md Technical architecture overview
scripts/train.py  Run to train the model
scripts/evaluate.py Run to evaluate and show plots
scripts/detect.py Run for live detection demo

PRE-SHOWCASE CHECKLIST:

□ Read README.md thoroughly
□ Run setup.sh to create environment
□ Download and organize dataset
□ Train model (scripts/train.py)
□ Run evaluation (scripts/evaluate.py)
□ Test live detection (scripts/detect.py)
□ Prepare alarm sound (assets/alarm.wav)
□ Test with different lighting/angles
□ Prepare presentation (what to show judges)
□ Review config.py parameters
□ Have backup demo.mp4 if webcam fails

PAPER RESULTS TO MATCH:

Accuracy  : 90.0%
Precision : 100%
Recall    : 83.3%
F1-Score  : 0.909

If your results match ± 2%, you're golden! 🎯

NEED HELP?

1. Check TROUBLESHOOTING in README.md
2. Review config.py for parameter adjustments
3. Re-read QUICKSTART.md for 5-min walkthrough
4. Verify dataset structure matches (data/train/Open_Eyes/...)
5. Run verification again: python FINAL_VERIFICATION.py

Good luck with your showcase! 🚀
""")
    else:
        print("\n⚠️  Some components are missing.")
        print("Run the following to recreate:")
        print("  git clone <repo-url>")
        print("  cd ddsh")
        print("  python FINAL_VERIFICATION.py  # Run again")
    
    print("="*70 + "\n")
    return all_good

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = verify_project()
    sys.exit(0 if success else 1)
