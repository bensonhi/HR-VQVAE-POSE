import os
import argparse
import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_runtime_samples(samples_dir):
    """Load all runtime samples from the directory"""
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.npz')]
    sample_files.sort()

    samples = []
    for filename in sample_files:
        filepath = os.path.join(samples_dir, filename)
        data = np.load(filepath)
        samples.append({
            'filename': filename,
            'original': data['original'],
            'reconstructed': data['reconstructed'],
            'source_paths': data.get('source_paths', None),
            'source_start_idx': data.get('source_start_idx', None)
        })
    return samples


def vector_to_joints(vec):
    """Convert a flat vector (J*3,) into (J, 3) joints array."""
    v = np.array(vec).reshape(-1)
    if v.size % 3 != 0:
        raise ValueError(f"Vector length {v.size} is not divisible by 3")
    j = v.size // 3
    return v.reshape(j, 3)


def load_beat2_smplx_params(beat2_path, frame_idx=0):
    """Load SMPL-X parameters from BEAT2 processed file"""
    try:
        data = np.load(beat2_path)

        if frame_idx >= data['poses'].shape[0]:
            frame_idx = 0

        # Extract SMPL-X parameters for the specific frame
        pose_165d = data['poses'][frame_idx]  # (165,)
        betas = data['betas']  # (300,) - same for all frames
        trans = data['trans'][frame_idx]  # (3,)
        expressions = data['expressions'][frame_idx] if 'expressions' in data else np.zeros(100)

        # Convert 165D pose to SMPL-X parameters (same as original script)
        global_orient = pose_165d[:3]
        body_pose = pose_165d[3:66]
        jaw_pose = pose_165d[66:69]
        leye_pose = pose_165d[69:72]
        reye_pose = pose_165d[72:75]
        left_hand_pose = pose_165d[75:120]
        right_hand_pose = pose_165d[120:165]

        return {
            'global_orient': torch.tensor(global_orient).reshape(1, 3).float(),
            'body_pose': torch.tensor(body_pose).reshape(1, 63).float(),
            'jaw_pose': torch.tensor(jaw_pose).reshape(1, 3).float(),
            'leye_pose': torch.tensor(leye_pose).reshape(1, 3).float(),
            'reye_pose': torch.tensor(reye_pose).reshape(1, 3).float(),
            'left_hand_pose': torch.tensor(left_hand_pose).reshape(1, 45).float(),
            'right_hand_pose': torch.tensor(right_hand_pose).reshape(1, 45).float(),
            'betas': torch.tensor(betas[:10]).reshape(1, 10).float(),  # Use first 10 shape parameters
            'transl': torch.tensor(trans).reshape(1, 3).float(),
            'expression': torch.tensor(expressions[:10]).reshape(1, 10).float()  # Use first 10 expression parameters
        }
    except Exception as e:
        print(f"Error loading BEAT2 parameters: {e}")
        return None


def joints_to_smplx_params_simple(joints_3d, smplx_model, beat2_params=None, device='cpu'):
    """
    Create SMPL-X parameters from joint positions.
    If BEAT2 params available, use those as base. Otherwise, fit from scratch.
    """
    if beat2_params is not None:
        # Use BEAT2 parameters as base and just adjust translation
        params = {}
        for key, value in beat2_params.items():
            params[key] = value.clone().to(device)

        # Adjust translation to match joint positions
        joints_centered = joints_3d - joints_3d.mean(axis=0)
        target_center = torch.tensor(joints_centered.mean(axis=0)).float().to(device)
        params['transl'] = target_center.reshape(1, 3)

        return params
    else:
        # Simple neutral pose with estimated translation
        joints_center = joints_3d.mean(axis=0)

        return {
            'global_orient': torch.zeros(1, 3, device=device).float(),
            'body_pose': torch.zeros(1, 63, device=device).float(),
            'jaw_pose': torch.zeros(1, 3, device=device).float(),
            'leye_pose': torch.zeros(1, 3, device=device).float(),
            'reye_pose': torch.zeros(1, 3, device=device).float(),
            'left_hand_pose': torch.zeros(1, 45, device=device).float(),
            'right_hand_pose': torch.zeros(1, 45, device=device).float(),
            'betas': torch.zeros(1, 10, device=device).float(),
            'transl': torch.tensor(joints_center).reshape(1, 3).float().to(device),
            'expression': torch.zeros(1, 10, device=device).float()
        }


def generate_mesh_data(model, pose_params, title="Pose"):
    """Generate vertices and joints from SMPL-X parameters"""
    try:
        output = model(**pose_params, return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        print(f'{title} - Vertices shape: {vertices.shape}, Joints shape: {joints.shape}')
        return vertices, joints
    except Exception as e:
        print(f"ERROR: Failed to generate mesh for {title}: {e}")
        return None, None


def plot_avatar_mesh(ax, vertices, joints, faces, plot_joints=True, title="", color='lightgray'):
    """Plot a single avatar mesh on given axis"""

    # Sample faces for better performance but still show mesh structure
    face_sample = faces[::10]  # Every 10th face for good mesh density

    # Create triangular mesh patches
    mesh_triangles = []
    for face in face_sample:
        if all(idx < len(vertices) for idx in face):
            triangle = vertices[face]
            mesh_triangles.append(triangle)

    if mesh_triangles:
        # Create mesh collection - this creates the actual 3D mesh surface
        mesh_collection = Poly3DCollection(
            mesh_triangles,
            alpha=0.8,
            facecolor=color,
            edgecolor='gray',
            linewidth=0.05
        )
        ax.add_collection3d(mesh_collection)

    # Add joints as red spheres
    if plot_joints:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                  c='red', s=20, alpha=0.9, marker='o')

    # Set equal aspect ratio
    max_range = np.array([vertices.max() - vertices.min()]).max() / 2.0
    mid = vertices.mean(axis=0)

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Set viewing angle for better pose visibility
    ax.view_init(elev=10, azim=45)

    # Style
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def visualize_side_by_side_comparison(model, original_params, reconstructed_params, args, save_path=None):
    """Create TRUE side-by-side comparison with actual meshes"""

    # Generate mesh data for both poses
    print("Generating original mesh...")
    orig_vertices, orig_joints = generate_mesh_data(model, original_params, "Original")

    print("Generating reconstructed mesh...")
    recon_vertices, recon_joints = generate_mesh_data(model, reconstructed_params, "Reconstructed")

    if orig_vertices is None or recon_vertices is None:
        print("Failed to generate mesh data")
        return

    # Create side-by-side subplot layout
    fig = plt.figure(figsize=(16, 8))

    # LEFT SUBPLOT - Original Pose
    ax1 = fig.add_subplot(121, projection='3d')
    plot_avatar_mesh(ax1, orig_vertices, orig_joints, model.faces,
                    args.plot_joints, "Original Pose", 'lightgray')

    # RIGHT SUBPLOT - Reconstructed Pose
    ax2 = fig.add_subplot(122, projection='3d')
    plot_avatar_mesh(ax2, recon_vertices, recon_joints, model.faces,
                    args.plot_joints, "Reconstructed Pose", 'lightblue')

    # Ensure both subplots have the same scale for fair comparison
    all_vertices = np.vstack([orig_vertices, recon_vertices])
    max_range = np.array([all_vertices.max() - all_vertices.min()]).max() / 2.0
    mid = all_vertices.mean(axis=0)

    for ax in [ax1, ax2]:
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.view_init(elev=10, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved side-by-side comparison to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize HR-VQVAE joint reconstruction with SMPL-X avatar comparison')
    parser.add_argument('--model-folder', default='models_smplx_v1_1/models', type=str,
                        help='Path to SMPL-X model folder')
    parser.add_argument('--samples-dir', default='checkpoint/beat2_poses/0/vqvae/runtime_samples',
                        type=str, help='Path to runtime samples directory')
    parser.add_argument('--beat2-dir', default='BEAT2_joints_vertices/beat_chinese_v2.0.0/smplxflame_30',
                        type=str, help='Path to processed BEAT2 data directory')
    parser.add_argument('--sample-idx', default=-1, type=int,
                        help='Which sample to visualize (-1 for latest)')
    parser.add_argument('--batch-idx', default=0, type=int,
                        help='Batch index within sample')
    parser.add_argument('--frame-idx', default=0, type=int,
                        help='Which frame in the sequence to visualize')
    parser.add_argument('--gender', default='neutral', type=str,
                        help='SMPL-X model gender')
    parser.add_argument('--plot-joints', default=True, type=bool,
                        help='Whether to plot joints')
    parser.add_argument('--save-images', action='store_true',
                        help='Save visualization images instead of displaying')
    parser.add_argument('--output-dir', default='visualization_output', type=str,
                        help='Directory to save visualization images')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Device for SMPL-X computations')

    args = parser.parse_args()

    # Load SMPL-X model
    try:
        model = smplx.create(args.model_folder,
                             model_type='smplx',
                             gender=args.gender,
                             use_face_contour=False,
                             use_pca=False,
                             ext='npz')
        model = model.to(args.device)
        print(f"Successfully loaded SMPL-X model from {args.model_folder}")
    except Exception as e:
        print(f"ERROR: Failed to load SMPL-X model: {e}")
        return

    # Load runtime samples
    try:
        samples = load_runtime_samples(args.samples_dir)
        if not samples:
            print(f"ERROR: No runtime samples found in {args.samples_dir}")
            return
        print(f"Found {len(samples)} runtime samples")
    except Exception as e:
        print(f"ERROR: Failed to load runtime samples: {e}")
        return

    # Select sample
    idx = args.sample_idx if args.sample_idx >= 0 else len(samples) - 1
    if idx >= len(samples):
        print(f"Sample index {idx} out of range (available: {len(samples)})")
        return

    sample = samples[idx]
    print(f"Using sample: {sample['filename']}")

    # Extract joint data
    orig = sample['original']
    recon = sample['reconstructed']

    if args.batch_idx >= orig.shape[0]:
        print(f"Batch index {args.batch_idx} out of range (available: {orig.shape[0]})")
        return
    if args.frame_idx >= orig.shape[1]:
        print(f"Frame index {args.frame_idx} out of range (available: {orig.shape[1]})")
        return

    original_frame = orig[args.batch_idx, args.frame_idx, :]
    reconstructed_frame = recon[args.batch_idx, args.frame_idx, :]

    # Convert to joint positions
    original_joints = vector_to_joints(original_frame)
    reconstructed_joints = vector_to_joints(reconstructed_frame)

    print(f"Joint positions - Original range: [{original_joints.min():.3f}, {original_joints.max():.3f}]")
    print(f"Joint positions - Reconstructed range: [{reconstructed_joints.min():.3f}, {reconstructed_joints.max():.3f}]")

    # Calculate MSE
    mse_joints = np.mean((original_joints - reconstructed_joints) ** 2)
    print(f"Joint position MSE: {mse_joints:.6f}")

    # Try to load corresponding BEAT2 data for better SMPL-X parameters
    beat2_params = None
    if sample.get('source_paths') is not None and len(sample['source_paths']) > args.batch_idx:
        beat2_file = sample['source_paths'][args.batch_idx]
        beat2_frame_idx = sample['source_start_idx'][args.batch_idx] + args.frame_idx if sample.get('source_start_idx') is not None else args.frame_idx

        if isinstance(beat2_file, (str, bytes)) and len(str(beat2_file).strip()) > 0:
            print(f"Loading BEAT2 parameters from: {beat2_file} (frame {beat2_frame_idx})")
            beat2_params = load_beat2_smplx_params(beat2_file, beat2_frame_idx)

    # Create SMPL-X parameters
    print("\n=== Creating SMPL-X parameters ===")
    original_params = joints_to_smplx_params_simple(original_joints, model, beat2_params, args.device)
    reconstructed_params = joints_to_smplx_params_simple(reconstructed_joints, model, beat2_params, args.device)

    # Create output path if saving
    save_path = None
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"side_by_side_comparison_sample{idx}_batch{args.batch_idx}_frame{args.frame_idx}.png")

    # Create TRUE side-by-side visualization with actual meshes
    print("\n=== Creating Side-by-Side Mesh Visualization ===")
    visualize_side_by_side_comparison(model, original_params, reconstructed_params, args, save_path)


if __name__ == '__main__':
    main()