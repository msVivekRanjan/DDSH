"""
pose_evaluate.py — Visualization and evaluation utilities for pose module

Generates:
 - outputs/pose_skeleton.png (sample skeleton visualization)
 - outputs/pose_heatmap_comparison.png (fixed vs adaptive sigma)
 - outputs/keypoint_distribution.png (133 kp scatter colored by region)

Prints a short console summary referencing the paper results and chosen model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config
from pose_model import AdaptiveGaussianGenerator, KeypointRegionSplitter

os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def synthetic_keypoints(img_w: int = 640, img_h: int = 480):
    """Create synthetic keypoints for demonstration (133 keypoints).

    Returns: kp array (K,3)
    """
    K = config.BODY_KP + config.FACE_KP + 2*config.HAND_KP + config.FEET_KP
    kp = np.zeros((K, 3), dtype=np.float32)
    # place body keypoints along center vertical
    for i in range(config.BODY_KP):
        kp[i, 0] = img_w * 0.5 + np.random.randn() * 5
        kp[i, 1] = img_h * (0.2 + 0.6 * i / max(1, config.BODY_KP)) + np.random.randn() * 5
        kp[i, 2] = 0.9
    # face cluster near top center
    face_start = config.BODY_KP
    for i in range(config.FACE_KP):
        kp[face_start + i, 0] = img_w * 0.5 + np.random.randn() * 8
        kp[face_start + i, 1] = img_h * 0.12 + np.random.randn() * 6
        kp[face_start + i, 2] = 0.95
    # hands
    lhand_start = config.BODY_KP + config.FACE_KP
    for i in range(config.HAND_KP):
        kp[lhand_start + i, 0] = img_w * 0.25 + np.random.randn() * 6
        kp[lhand_start + i, 1] = img_h * 0.5 + np.random.randn() * 6
        kp[lhand_start + i, 2] = 0.9
    rhand_start = lhand_start + config.HAND_KP
    for i in range(config.HAND_KP):
        kp[rhand_start + i, 0] = img_w * 0.75 + np.random.randn() * 6
        kp[rhand_start + i, 1] = img_h * 0.5 + np.random.randn() * 6
        kp[rhand_start + i, 2] = 0.9
    # feet
    feet_start = rhand_start + config.HAND_KP
    for i in range(config.FEET_KP):
        kp[feet_start + i, 0] = img_w * (0.3 + 0.4 * (i / max(1, config.FEET_KP)))
        kp[feet_start + i, 1] = img_h * 0.95
        kp[feet_start + i, 2] = 0.8
    return kp


def plot_skeleton_sample(kp: np.ndarray):
    """Plot sample skeleton as outputs/pose_skeleton.png"""
    h, w = 480, 640
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(np.ones((h, w, 3), dtype=np.uint8) * 255)
    splitter = KeypointRegionSplitter()
    body, face, lhand, rhand, feet = splitter.split(kp)
    ax.scatter(body[:,0], body[:,1], c='g', label='Body', s=8)
    ax.scatter(face[:,0], face[:,1], c='c', label='Face', s=6)
    ax.scatter(lhand[:,0], lhand[:,1], c='y', label='Left Hand', s=6)
    ax.scatter(rhand[:,0], rhand[:,1], c='y', label='Right Hand', s=6)
    ax.scatter(feet[:,0], feet[:,1], c='r', label='Feet', s=8)
    ax.set_title('Sample Pose Skeleton')
    ax.axis('off')
    ax.legend(loc='upper right')
    out = config.POSE_SKELETON_OUTPUT
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved sample skeleton: {out}")


def plot_heatmap_comparison(kp: np.ndarray):
    """Compare fixed sigma vs adaptive sigma heatmaps for face and body."""
    h, w = config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH
    gen = AdaptiveGaussianGenerator((h, w))
    splitter = KeypointRegionSplitter()
    body, face, _, _, _ = splitter.split(kp)

    # choose one face keypoint and one body keypoint
    face_k = face[0]
    body_k = body[len(body)//2]

    fixed_sigma = config.SIGMA_BASE
    adaptive_sigma_face = config.SIGMA_BASE * gen.compute_density_factor(config.FACE_KP) * gen.compute_visibility_factor(face_k[2])
    adaptive_sigma_body = config.SIGMA_BASE * gen.compute_density_factor(config.BODY_KP) * gen.compute_visibility_factor(body_k[2])

    face_heat_fixed = gen.generate_adaptive_heatmap((face_k[0], face_k[1]), fixed_sigma)
    face_heat_adapt = gen.generate_adaptive_heatmap((face_k[0], face_k[1]), adaptive_sigma_face)

    body_heat_fixed = gen.generate_adaptive_heatmap((body_k[0], body_k[1]), fixed_sigma)
    body_heat_adapt = gen.generate_adaptive_heatmap((body_k[0], body_k[1]), adaptive_sigma_body)

    # Plot side-by-side
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    axes[0,0].imshow(face_heat_fixed, cmap='hot')
    axes[0,0].set_title(f'Face Fixed σ={fixed_sigma:.2f}')
    axes[0,1].imshow(face_heat_adapt, cmap='hot')
    axes[0,1].set_title(f'Face Adaptive σ={adaptive_sigma_face:.2f}')

    axes[1,0].imshow(body_heat_fixed, cmap='hot')
    axes[1,0].set_title(f'Body Fixed σ={fixed_sigma:.2f}')
    axes[1,1].imshow(body_heat_adapt, cmap='hot')
    axes[1,1].set_title(f'Body Adaptive σ={adaptive_sigma_body:.2f}')

    for ax in axes.flatten():
        ax.axis('off')

    out = config.POSE_HEATMAP_OUTPUT
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved heatmap comparison: {out}")


def plot_keypoint_distribution(kp: np.ndarray):
    h, w = config.DISPLAY_HEIGHT, config.DISPLAY_WIDTH
    splitter = KeypointRegionSplitter()
    body, face, lhand, rhand, feet = splitter.split(kp)
    plt.figure(figsize=(8,6))
    plt.scatter(body[:,0], body[:,1], c='g', s=6, label='Body')
    plt.scatter(face[:,0], face[:,1], c='c', s=4, label='Face')
    plt.scatter(lhand[:,0], lhand[:,1], c='y', s=4, label='Left Hand')
    plt.scatter(rhand[:,0], rhand[:,1], c='y', s=4, label='Right Hand')
    plt.scatter(feet[:,0], feet[:,1], c='r', s=6, label='Feet')
    plt.title('Keypoint Distribution (Synthetic)')
    plt.gca().invert_yaxis()
    plt.legend()
    out = config.POSE_KP_DISTRIBUTION
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Saved keypoint distribution: {out}")


def main():
    print("\n" + "="*60)
    print("DDSH — Pose Module Evaluation & Visualization")
    print("="*60)
    print(f"Paper SOTA: 77.8% AP (COCO-WholeBody)")
    print(f"Our implementation uses: {config.POSE_MODEL_NAME} (backend={config.POSE_MODEL_TYPE})")
    print(f"Adaptive Gaussian: ENABLED | Reference Constraints: ENABLED")

    kp = synthetic_keypoints(config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)
    plot_skeleton_sample(kp)
    plot_heatmap_comparison(kp)
    plot_keypoint_distribution(kp)
    print("\n✓ Pose evaluation artifacts created in outputs/")

if __name__ == '__main__':
    main()
