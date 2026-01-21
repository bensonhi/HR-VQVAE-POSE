"""
Visualize VQ-VAE joint reconstruction using matplotlib with spheres and skeleton lines
Pure Python visualization - no Blender needed
Reads directly from checkpoint runtime samples
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os


def load_runtime_samples(samples_dir):
    """Load all runtime samples (.npz) from a directory."""
    files = [f for f in os.listdir(samples_dir) if f.endswith('.npz')]
    files.sort()
    samples = []
    for f in files:
        p = os.path.join(samples_dir, f)
        try:
            data = np.load(p)
            samples.append({
                'filename': f,
                'path': p,
                'original': data['original'],
                'reconstructed': data['reconstructed'],
            })
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return samples


def vector_to_joints(vec):
    """Convert flat vector to (J, 3) joints"""
    v = np.array(vec).reshape(-1)
    j = v.size // 3
    return v.reshape(j, 3)


def plot_skeleton_3d(ax, joints, color='blue', alpha=0.8, label=''):
    """Plot skeleton with spheres at joints and lines connecting them"""

    # SMPL-X kinematic chain (parent-child bone connections)
    connections = [
        (0, 1), (0, 2), (0, 3),  # Pelvis to hips and spine
        (1, 4), (2, 5),  # Hips to knees
        (4, 7), (5, 8),  # Knees to ankles
        (7, 10), (8, 11),  # Ankles to feet
        (3, 6), (6, 9),  # Spine chain
        (9, 12), (12, 15),  # Spine to neck to head
        (9, 13), (9, 14),  # Spine to collars
        (13, 16), (14, 17),  # Collars to shoulders
        (16, 18), (17, 19),  # Shoulders to elbows
        (18, 20), (19, 21),  # Elbows to wrists
        # Add some hand connections if we have enough joints
        (20, 25), (20, 28), (20, 31), (20, 34), (20, 37),  # Left hand
        (21, 40), (21, 43), (21, 46), (21, 49), (21, 52),  # Right hand
    ]

    # Remap axes: X horizontal, Y vertical, Z depth
    # Original: joints[:, 0]=X, joints[:, 1]=Y, joints[:, 2]=Z
    # We want: X (horizontal) = original X
    #          Y (vertical) = original Y
    #          Z (depth) = original Z
    # Plot as: X, Z, Y so matplotlib treats Y as vertical

    # Plot skeleton lines with axes remapped
    for start_idx, end_idx in connections:
        if start_idx < len(joints) and end_idx < len(joints):
            points = np.array([joints[start_idx], joints[end_idx]])
            # Plot as (X, -Z, Y) - matplotlib will show Y as vertical, Z inverted
            ax.plot3D(points[:, 0], -points[:, 2], points[:, 1],
                     color=color, linewidth=2, alpha=alpha*0.6)

    # Plot joint spheres with axes remapped (X, -Z, Y)
    ax.scatter(joints[:, 0], -joints[:, 2], joints[:, 1],
              c=color, s=8, alpha=alpha, marker='o', label=label)


def main():
    parser = argparse.ArgumentParser(description='Visualize VAE joint reconstruction')
    parser.add_argument('--samples-dir', default='checkpoint/beat2_poses/0/vae/runtime_samples', type=str,
                        help='Directory with runtime sample .npz files')
    parser.add_argument('--sample-idx', default=-1, type=int,
                        help='Which sample file to use (-1 = latest)')
    parser.add_argument('--batch-idx', default=0, type=int,
                        help='Batch index within the sample file')
    parser.add_argument('--frame-idx', default=0, type=int,
                        help='Frame index (time) within the sample')
    parser.add_argument('--save', default=None, type=str, help='Save figure to file')
    parser.add_argument('--no-show', action='store_true', help='Do not display interactive window')

    args = parser.parse_args()

    print("="*60)
    print("VQ-VAE Joint Position Visualization")
    print("="*60)

    # Load runtime samples
    samples = load_runtime_samples(args.samples_dir)
    if not samples:
        print(f"No .npz samples found in {args.samples_dir}")
        return

    idx = args.sample_idx if args.sample_idx >= 0 else len(samples) - 1
    if idx >= len(samples):
        print(f"sample-idx {idx} out of range (num files: {len(samples)})")
        return

    sample = samples[idx]
    print(f"\nUsing sample file: {sample['filename']}")

    orig = sample['original']  # (B, T, J*3)
    recon = sample['reconstructed']

    if orig.ndim != 3 or recon.ndim != 3:
        print(f"Unexpected array shape: original {orig.shape}, reconstructed {recon.shape}")
        return

    b, t, d = orig.shape
    if args.batch_idx >= b:
        print(f"batch-idx {args.batch_idx} out of range (B={b})")
        return
    if args.frame_idx >= t:
        print(f"frame-idx {args.frame_idx} out of range (T={t})")
        return

    # Extract specific frame
    orig_vec = orig[args.batch_idx, args.frame_idx]
    recon_vec = recon[args.batch_idx, args.frame_idx]

    # Convert to joint positions
    original_joints = vector_to_joints(orig_vec)
    reconstructed_joints = vector_to_joints(recon_vec)

    print(f"\nBatch: {args.batch_idx}, Frame: {args.frame_idx}")
    print(f"Number of joints: {original_joints.shape[0]}")

    # Calculate MSE
    mse = np.mean((original_joints - reconstructed_joints) ** 2)
    print(f"Joint position MSE: {mse:.6f}")

    # Create visualization
    print(f"\nCreating visualization...")
    fig = plt.figure(figsize=(16, 8))

    # Left subplot - Original
    ax1 = fig.add_subplot(121, projection='3d')
    plot_skeleton_3d(ax1, original_joints, color='blue', alpha=0.9, label='Original')
    ax1.set_title('Original Pose', fontsize=16, fontweight='bold')
    ax1.legend()

    # Right subplot - Reconstructed
    ax2 = fig.add_subplot(122, projection='3d')
    plot_skeleton_3d(ax2, reconstructed_joints, color='red', alpha=0.9, label='Reconstructed')
    ax2.set_title('VQ-VAE Reconstruction', fontsize=16, fontweight='bold')
    ax2.legend()

    # Make both plots have the same scale
    all_joints = np.vstack([original_joints, reconstructed_joints])
    max_range = np.array([all_joints.max() - all_joints.min()]).max() / 2.0
    mid = all_joints.mean(axis=0)

    for ax in [ax1, ax2]:
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_xlabel('X (Horizontal)')
        ax.set_ylabel('Z (Depth)')
        ax.set_zlabel('Y (Vertical)')
        # View from front angle with slight elevation
        ax.view_init(elev=15, azim=-65)

    plt.suptitle(f'VQ-VAE Joint Reconstruction (MSE: {mse:.6f})', fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Save or show
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"\n[+] Saved to: {args.save}")

    if not args.no_show:
        print(f"\n[+] Displaying visualization...")
        plt.show()
    else:
        plt.close(fig)

    print("="*60)


if __name__ == '__main__':
    main()
