import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_runtime_samples(samples_dir):
    """Load all runtime samples (.npz) from a directory.

    Each file is expected to contain arrays:
      - 'original': (B, T, J*3)
      - 'reconstructed': (B, T, J*3)
    """
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
    """Convert a flat vector (J*3,) into (J, 3) joints array.

    Accepts vec with shape (J*3,) or (1, J*3).
    """
    v = np.array(vec).reshape(-1)
    if v.size % 3 != 0:
        raise ValueError(f"Vector length {v.size} is not divisible by 3")
    j = v.size // 3
    return v.reshape(j, 3)


def center_joints(joints, mode='pelvis'):
    """Center joints in-place.

    - 'pelvis': subtract joint 0 (assumed pelvis/root)
    - 'mean': subtract mean across all joints
    - None: no centering
    """
    if mode is None:
        return joints
    joints = joints.copy()
    if mode == 'pelvis':
        joints -= joints[0:1, :]
    elif mode == 'mean':
        joints -= joints.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown center mode: {mode}")
    return joints


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale for proper aspect ratio."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-6])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_points(ax, joints, title=None, color='C0', alpha=0.9, s=10):
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=s, alpha=alpha)
    if title:
        ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)


def compare_and_plot(orig_vec, recon_vec, center_mode='pelvis', overlay=False, save_path=None, show=True, normalize=False, debug=False):
    """Visualize original vs reconstructed joints in 3D.

    - orig_vec, recon_vec: flat vectors (J*3,)
    - center_mode: 'pelvis' | 'mean' | None
    - overlay: if True draw both in one axis, else side-by-side subplots
    """
    orig = vector_to_joints(orig_vec)
    recon = vector_to_joints(recon_vec)

    # Centering for fair visual comparison
    orig_c = center_joints(orig, center_mode)
    recon_c = center_joints(recon, center_mode)

    # Optional per-axis normalization (helps if magnitudes differ a lot)
    if normalize:
        def norm_xyz(x):
            mu = x.mean(axis=0, keepdims=True)
            sd = x.std(axis=0, keepdims=True) + 1e-8
            return (x - mu) / sd
        orig_c = norm_xyz(orig_c)
        recon_c = norm_xyz(recon_c)

    # Metrics
    diff = recon - orig
    mse = float(np.mean(diff ** 2))
    per_joint_rmse = np.sqrt(np.mean((recon - orig) ** 2, axis=1))

    if debug:
        print(f"Detected joints: {orig.shape[0]}")
        print(f"Original (pre-center) xyz min/max: {orig.min(axis=0)}, {orig.max(axis=0)}")
        print(f"Reconst  (pre-center) xyz min/max: {recon.min(axis=0)}, {recon.max(axis=0)}")
        print(f"Original (post-center) xyz min/max: {orig_c.min(axis=0)}, {orig_c.max(axis=0)}")
        print(f"Reconst  (post-center) xyz min/max: {recon_c.min(axis=0)}, {recon_c.max(axis=0)}")
        # Collapse check
        if np.allclose(orig_c.ptp(axis=0), 0, atol=1e-6):
            print("WARNING: Original joints collapsed to a single location after centering (zero spread).")
        if np.allclose(recon_c.ptp(axis=0), 0, atol=1e-6):
            print("WARNING: Reconstructed joints collapsed to a single location after centering (zero spread).")

    if overlay:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        plot_points(ax, orig_c, title=f'Overlay (MSE={mse:.6f})', color='C0', s=12)
        plot_points(ax, recon_c, title=None, color='C3', s=12, alpha=0.8)
        ax.legend(['Original', 'Reconstructed'])
    else:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        plot_points(ax1, orig_c, title='Original', color='C0', s=12)
        plot_points(ax2, recon_c, title=f'Reconstructed (MSE={mse:.6f})', color='C3', s=12)

    fig.tight_layout()

    # Print brief metrics
    print(f"Overall MSE: {mse:.6f}")
    print("Per-joint RMSE (first 10):")
    for i, v in enumerate(per_joint_rmse[:10]):
        print(f"  Joint {i}: {v:.6f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize joint-position VQ-VAE reconstruction (original vs reconstructed)')
    parser.add_argument('--samples-dir', default='checkpoint/beat2_poses/0/vqvae/runtime_samples', type=str,
                        help='Directory with runtime sample .npz files')
    parser.add_argument('--sample-idx', default=-1, type=int,
                        help='Which sample file to use (-1 = latest)')
    parser.add_argument('--batch-idx', default=0, type=int,
                        help='Batch index within the sample file')
    parser.add_argument('--frame-idx', default=0, type=int,
                        help='Frame index (time) within the sample (default 0)')
    parser.add_argument('--center', default='pelvis', choices=['pelvis', 'mean', 'none'],
                        help='Centering mode for visualization')
    parser.add_argument('--overlay', action='store_true', help='Overlay both poses in one plot')
    parser.add_argument('--save', default=None, type=str, help='Path to save the figure (PNG)')
    parser.add_argument('--no-show', action='store_true', help='Do not display interactive window')
    parser.add_argument('--normalize', action='store_true', help='Per-axis normalize after centering')
    parser.add_argument('--debug', action='store_true', help='Print detailed stats')

    args = parser.parse_args()

    samples = load_runtime_samples(args.samples_dir)
    if not samples:
        print(f"No .npz samples found in {args.samples_dir}")
        return

    idx = args.sample_idx if args.sample_idx >= 0 else len(samples) - 1
    if idx >= len(samples):
        print(f"sample-idx {idx} out of range (num files: {len(samples)})")
        return

    sample = samples[idx]
    print(f"Using sample file: {sample['filename']}")

    orig = sample['original']  # (B, T, J*3) or possibly (B, 1, J*3)
    recon = sample['reconstructed']

    if orig.ndim != 3 or recon.ndim != 3:
        print(f"Unexpected array shape: original {orig.shape}, reconstructed {recon.shape}")
        print("Expected 3D arrays (B, T, J*3). If your tensors are (B, J*3), try expanding with an extra time dim.")
        return

    b, t, d = orig.shape
    if args.batch_idx >= b:
        print(f"batch-idx {args.batch_idx} out of range (B={b})")
        return
    if args.frame_idx >= t:
        print(f"frame-idx {args.frame_idx} out of range (T={t})")
        return

    orig_vec = orig[args.batch_idx, args.frame_idx]
    recon_vec = recon[args.batch_idx, args.frame_idx]

    if args.debug:
        print(f"Selected vector dims: D={orig_vec.shape[-1]}")
        if orig_vec.shape[-1] % 3 != 0:
            print("WARNING: D is not divisible by 3 -> cannot interpret as J*3 (x,y,z).")
        print(f"orig vec stats: min={orig_vec.min():.6f}, max={orig_vec.max():.6f}, mean={orig_vec.mean():.6f}")
        print(f"recon vec stats: min={recon_vec.min():.6f}, max={recon_vec.max():.6f}, mean={recon_vec.mean():.6f}")

    center_mode = None if args.center == 'none' else args.center

    # Default save path next to sample if not provided
    save_path = args.save
    if save_path is None:
        stem = os.path.splitext(sample['filename'])[0]
        save_path = os.path.join(args.samples_dir, f"viz_{stem}_b{args.batch_idx}_t{args.frame_idx}.png")

    compare_and_plot(
        orig_vec,
        recon_vec,
        center_mode=center_mode,
        overlay=args.overlay,
        save_path=save_path,
        show=not args.no_show,
        normalize=args.normalize,
        debug=args.debug,
    )


if __name__ == '__main__':
    main()
