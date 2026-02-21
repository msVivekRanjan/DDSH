"""
preprocess.py — Dataset Loading & Preprocessing Pipeline

Implements the paper-exact preprocessing sequence for MRL Eye Dataset
using tf.data generators to avoid loading all images into RAM.

Preprocessing sequence:
1. Load image from disk
2. Convert to grayscale (single channel)
3. Resize to 224×224 (MobileNet input requirement)
4. Convert back to RGB (3 channels)
5. Normalize to [0, 1] range

Source: MRL Eyes 2018 Dataset (http://mrl.cs.vsb.cz/eyedataset)
Paper: Bhanja et al., ROBOMECH Journal (2025)
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple
import config


# ──────────────────────────────────────────────
# Single-image preprocessing (TensorFlow ops)
# ──────────────────────────────────────────────

def preprocess_image_tf(img_path: str, label: int):
    """
    TensorFlow-native preprocessing pipeline.
    Runs inside tf.data map — no OpenCV, no RAM spike.

    Raw PNG → Grayscale → Resize 224×224 → RGB → Normalize [0,1]
    """
    # Read & decode image
    raw   = tf.io.read_file(img_path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)

    # Convert to grayscale then back to 3-channel RGB
    gray  = tf.image.rgb_to_grayscale(image)          # (H, W, 1)
    rgb   = tf.image.grayscale_to_rgb(gray)            # (H, W, 3)

    # Resize to MobileNet input size
    rgb   = tf.image.resize(rgb, [config.IMG_SIZE, config.IMG_SIZE])

    # Normalize to [0, 1]
    rgb   = tf.cast(rgb, tf.float32) / 255.0

    return rgb, label


# ──────────────────────────────────────────────
# Build file-path + label lists from directory
# ──────────────────────────────────────────────

def _collect_paths_and_labels(root_dir: str):
    """
    Walk root_dir/Open_Eyes and root_dir/Closed_Eyes,
    return (list_of_paths, list_of_labels).
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    class_map   = {"Open_Eyes": 0, "Closed_Eyes": 1}
    all_paths   = []
    all_labels  = []
    extensions  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for class_name, class_label in class_map.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  Warning: {class_dir} not found, skipping.")
            continue

        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if Path(f).suffix.lower() in extensions
        ]
        all_paths.extend(files)
        all_labels.extend([class_label] * len(files))

        print(f"📂 Loading {class_name}...")
        print(f"   ✓ Found {len(files)} images")

    if not all_paths:
        raise ValueError(f"No images found in {root_dir}.")

    return all_paths, all_labels


# ──────────────────────────────────────────────
# Build a tf.data.Dataset (generator-based)
# ──────────────────────────────────────────────

def build_tf_dataset(
    root_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    cache: bool = False,
) -> Tuple[tf.data.Dataset, int, int, int]:
    """
    Build a batched, prefetched tf.data.Dataset from directory.

    Args:
        root_dir   : Path containing Open_Eyes/ and Closed_Eyes/
        batch_size : Images per batch (default 32)
        shuffle    : Shuffle before batching (True for train)
        cache      : Cache dataset in memory (only if RAM allows)

    Returns:
        dataset    : tf.data.Dataset yielding (batch_images, batch_labels)
        n_total    : Total number of images
        n_open     : Count of open-eye images
        n_closed   : Count of closed-eye images
    """
    paths, labels = _collect_paths_and_labels(root_dir)

    n_total  = len(paths)
    n_open   = labels.count(0)
    n_closed = labels.count(1)

    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"  Total images       : {n_total}")
    print(f"  Open eyes (class 0): {n_open}")
    print(f"  Closed eyes (class 1): {n_closed}")
    print(f"{'='*60}\n")

    # Build dataset from slices (paths & labels stay on disk)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(n_total, 10_000), seed=42)

    # Map preprocessing — runs lazily, batch by batch
    ds = ds.map(
        preprocess_image_tf,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)   # prefetch next batch while GPU trains

    return ds, n_total, n_open, n_closed


# ──────────────────────────────────────────────
# Main entry point used by train.py
# ──────────────────────────────────────────────

def prepare_train_test_split(
    train_dir: str,
    test_dir: str,
    batch_size: int = 32,
):
    """
    Build train and test tf.data.Dataset objects.

    Returns:
        train_ds   : Batched, shuffled training dataset
        test_ds    : Batched test dataset
        train_info : (n_total, n_open, n_closed) for train
        test_info  : (n_total, n_open, n_closed) for test
    """
    print("\n🔄 Preparing Dataset...")
    print("=" * 60)

    print("\n[TRAIN]")
    train_ds, tr_total, tr_open, tr_closed = build_tf_dataset(
        train_dir, batch_size=batch_size, shuffle=True
    )

    print("\n[TEST]")
    test_ds, te_total, te_open, te_closed = build_tf_dataset(
        test_dir, batch_size=batch_size, shuffle=False
    )

    print(f"\n📊 Train/Test Split:")
    print(f"  Training samples   : {tr_total}")
    print(f"  Test samples       : {te_total}")
    print(f"  Class balance (train): {tr_open} open, {tr_closed} closed")
    print(f"  Class balance (test) : {te_open} open, {te_closed} closed")
    print("=" * 60 + "\n")

    train_info = (tr_total, tr_open, tr_closed)
    test_info  = (te_total, te_open, te_closed)

    return train_ds, test_ds, train_info, test_info


# ──────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DDSH — Preprocessing Pipeline Test")
    print("=" * 60)

    try:
        train_ds, test_ds, train_info, test_info = prepare_train_test_split(
            config.TRAIN_DIR, config.TEST_DIR
        )

        # Peek at one batch to verify shapes
        for images, labels in train_ds.take(1):
            print(f"\n✅ Batch shape : {images.shape}")
            print(f"   Labels sample: {labels.numpy()[:8]}")
            print(f"   Value range  : [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")

        print("\n✅ Preprocessing pipeline test successful!")

    except FileNotFoundError as e:
        print(f"\n❌ Dataset not found: {str(e)}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise