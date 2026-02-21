"""
train.py — MobileNet Transfer Learning Training Pipeline

Trains a MobileNet model on the preprocessed MRL Eye Dataset for binary
drowsiness detection (Open/Closed eyes).

Paper Implementation Details:
- Base Model: MobileNet (ImageNet pre-trained weights)
- Architecture: Global Average Pooling → Dense(1, activation='linear')
- Loss: Mean Squared Error (MSE) — regression-style, not classification
- Optimizer: Adam (lr=0.001, default parameters)
- Epochs: 5 (exact from paper)
- Batch Size: 32
- Validation Split: 10%

Reference: Bhanja et al., ROBOMECH Journal (2025)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import config
from preprocess import prepare_train_test_split


def create_model() -> keras.Model:
    """
    Build MobileNet-based model for drowsiness detection.

    Architecture:
    1. MobileNet base (ImageNet weights, top excluded)
    2. Global Average Pooling (condenses spatial dimensions)
    3. Fully Connected Dense layer (1 unit, linear activation)

    Output: Drowsiness score ∈ [0, 1] (0=open eyes, 1=closed eyes)

    Returns:
        keras.Model: Compiled model ready for training.
    """

    print("\n🏗️  Building MobileNet Model Architecture...")
    print("=" * 60)

    base_model = tf.keras.applications.MobileNet(
        weights=config.MODEL_WEIGHTS,
        include_top=config.INCLUDE_TOP,
        input_shape=config.INPUT_SHAPE,
    )

    print(f"  ✓ Base Model: MobileNet")
    print(f"  ✓ Pre-trained on: {config.MODEL_WEIGHTS}")
    print(f"  ✓ Input shape: {config.INPUT_SHAPE}")

    base_model.trainable = False
    print(f"  ✓ Base model frozen (weights not updated)")

    x = layers.GlobalAveragePooling2D()(base_model.output)
    print(f"  ✓ Global Average Pooling applied")

    output = layers.Dense(1, activation="linear")(x)
    print(f"  ✓ Output layer: Dense(1, activation='linear')")

    model = keras.Model(inputs=base_model.input, outputs=output)
    print(f"\n  Model created | Total parameters: {model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=config.LOSS_FUNCTION,
        metrics=["accuracy"],
    )

    print(f"  ✓ Optimizer: {config.OPTIMIZER} (lr={config.LEARNING_RATE})")
    print(f"  ✓ Loss function: {config.LOSS_FUNCTION}")
    print(f"  ✓ Metrics: ['accuracy']")
    print("=" * 60 + "\n")

    return model


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    train_total: int,
) -> keras.callbacks.History:
    """
    Train the model using a tf.data.Dataset (generator-based, no RAM spike).

    Args:
        model      : Compiled model from create_model().
        train_ds   : Batched tf.data.Dataset yielding (images, labels).
        train_total: Total number of training samples (for steps_per_epoch).

    Returns:
        keras.callbacks.History: Training history (losses, metrics per epoch).
    """

    print("\n🎓 Training MobileNet Model...")
    print("=" * 60)
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS} (paper-exact)")
    print(f"  Validation split: {config.VALIDATION_SPLIT * 100:.0f}%")
    print(f"  Training samples: {train_total}")
    print("=" * 60)

    # Split off ~10% of batches for validation
    total_batches    = train_total // config.BATCH_SIZE
    val_batches      = max(1, int(total_batches * config.VALIDATION_SPLIT))
    train_batches    = total_batches - val_batches

    val_ds    = train_ds.take(val_batches)
    fit_ds    = train_ds.skip(val_batches)

    print(f"  Train batches : {train_batches} ({train_batches * config.BATCH_SIZE} samples)")
    print(f"  Val batches   : {val_batches}  ({val_batches * config.BATCH_SIZE} samples)")
    print("=" * 60)

    history = model.fit(
        fit_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        verbose=1,
    )

    print("\n✅ Training complete!")
    return history


def save_model(model: keras.Model, save_path: str = None) -> str:
    """
    Save trained model to disk.

    Args:
        model     : Trained model.
        save_path : Path to save model. Defaults to config.MODEL_PATH.

    Returns:
        str: Path where model was saved.
    """

    if save_path is None:
        save_path = config.MODEL_PATH

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\n💾 Model saved to: {save_path}")
    print(f"   File size: {os.path.getsize(save_path) / (1024**2):.2f} MB")

    return save_path


def plot_training_history(history: keras.callbacks.History) -> None:
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: History object from model.fit().
    """

    print("\n📊 Generating Training Plots...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_loss = history.history["loss"]
    val_loss   = history.history["val_loss"]
    train_acc  = history.history["accuracy"]
    val_acc    = history.history["val_accuracy"]
    epochs_range = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, train_loss, "b-o", label="Training Loss",    linewidth=2)
    axes[0].plot(epochs_range, val_loss,   "r-s", label="Validation Loss",  linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (MSE)", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs_range)

    axes[1].plot(epochs_range, train_acc, "b-o", label="Training Accuracy",   linewidth=2)
    axes[1].plot(epochs_range, val_acc,   "r-s", label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs_range)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    plot_path = os.path.join(config.OUTPUT_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Plot saved: {plot_path}")
    plt.show()


def main():
    """
    Main training pipeline: Data → Model → Train → Save → Evaluate
    """

    print("\n" + "=" * 60)
    print("DDSH — MobileNet Training Pipeline")
    print("=" * 60)

    try:
        # Step 1: Load dataset as tf.data generators (no RAM spike)
        print("\n📥 Loading dataset...")
        train_ds, test_ds, train_info, test_info = prepare_train_test_split(
            config.TRAIN_DIR, config.TEST_DIR, batch_size=config.BATCH_SIZE
        )
        tr_total, tr_open, tr_closed = train_info
        te_total, te_open, te_closed = test_info

        # Step 2: Create model architecture
        model = create_model()

        # Step 3: Train model using generator dataset
        history = train_model(model, train_ds, tr_total)

        # Step 4: Save trained model
        save_model(model)

        # Step 5: Plot training history
        plot_training_history(history)

        # Step 6: Evaluate on test set
        print("\n📋 Evaluating on test set...")
        test_loss, test_acc = model.evaluate(test_ds, verbose=1)
        print(f"\n  Test Loss     : {test_loss:.4f}")
        print(f"  Test Accuracy : {test_acc:.4f}")

        print("\n✅ Training pipeline complete!")
        print(f"   Model saved : {config.MODEL_PATH}")
        print(f"   Next step   : Run evaluate.py for detailed metrics")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n📌 Dataset setup required:")
        print("   1. Download from: http://mrl.cs.vsb.cz/eyedataset")
        print("   2. Extract to data/train/ and data/test/")
        print("   3. Re-run this script")

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()