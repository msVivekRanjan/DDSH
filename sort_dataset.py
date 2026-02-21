import os, shutil, random

DATASET_PATH = "/Users/ms.vivekranjan/Downloads/mrlEyes_2018_01"
TRAIN_RATIO  = 0.8

open_imgs   = []
closed_imgs = []

# Walk through all s0001-s0037 folders
for subject_folder in sorted(os.listdir(DATASET_PATH)):
    subject_path = os.path.join(DATASET_PATH, subject_folder)
    
    if not os.path.isdir(subject_path):
        continue
    if not subject_folder.startswith("s"):
        continue

    for img_name in os.listdir(subject_path):
        if not img_name.endswith(".png"):
            continue
        
        parts = img_name.replace(".png", "").split("_")
        
        # parts[4] = eye state (0=closed, 1=open)
        if len(parts) < 5:
            continue
        
        eye_state = parts[4]
        full_path = os.path.join(subject_path, img_name)
        
        if eye_state == "1":
            open_imgs.append(full_path)
        elif eye_state == "0":
            closed_imgs.append(full_path)

print(f"Found: {len(open_imgs)} open, {len(closed_imgs)} closed images")

# Create directories
os.makedirs("/Users/ms.vivekranjan/VIVEK/CODE/PROJECTS/DDSH-VS-CLAUDE/data/train/Open_Eyes",   exist_ok=True)
os.makedirs("/Users/ms.vivekranjan/VIVEK/CODE/PROJECTS/DDSH-VS-CLAUDE/data/train/Closed_Eyes", exist_ok=True)
os.makedirs("/Users/ms.vivekranjan/VIVEK/CODE/PROJECTS/DDSH-VS-CLAUDE/data/test/Open_Eyes",    exist_ok=True)
os.makedirs("/Users/ms.vivekranjan/VIVEK/CODE/PROJECTS/DDSH-VS-CLAUDE/data/test/Closed_Eyes",  exist_ok=True)

def split_and_copy(img_list, label):
    random.shuffle(img_list)
    split       = int(len(img_list) * TRAIN_RATIO)
    train_imgs  = img_list[:split]
    test_imgs   = img_list[split:]
    
    for img in train_imgs:
        shutil.copy(img, f"/Users/ms.vivekranjan/VIVEK/CODE/PROJECTS/DDSH-VS-CLAUDE/data/train/{label}/")
    for img in test_imgs:
        shutil.copy(img, f"/Users/ms.vivekranjan/VIVEK/CODE/PROJECTS/DDSH-VS-CLAUDE/data/test/{label}/")

    
    print(f"{label:12s} → Train: {len(train_imgs):5d} | Test: {len(test_imgs):5d}")

split_and_copy(open_imgs,   "Open_Eyes")
split_and_copy(closed_imgs, "Closed_Eyes")
print("\n✅ Dataset sorted successfully!")