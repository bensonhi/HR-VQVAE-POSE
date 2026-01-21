import os
import argparse
import numpy as np
import torch
import smplx

def load_runtime_samples(samples_dir):
    """Load all runtime samples from the directory"""
    sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.npz')]
    sample_files.sort()

    samples = []
    for filename in sample_files:
        filepath = os.path.join(samples_dir, filename)
        data = np.load(filepath)

        sample_data = {
            'filename': filename,
            'original': data['original'],
            'reconstructed': data['reconstructed']
        }

        # Load per-level reconstructions if available
        for key in data.keys():
            if key.startswith('reconstructed_level_'):
                sample_data[key] = data[key]

        samples.append(sample_data)
    return samples

def pose_to_smplx_params(pose_165d):
    """Convert 165D pose to SMPL-X parameters
    
    SMPL-X pose breakdown (165 dims total):
    - global_orient: 3 (root orientation)
    - body_pose: 63 (21 joints * 3 rot params)
    - jaw_pose: 3 
    - leye_pose: 3
    - reye_pose: 3
    - left_hand_pose: 45 (15 joints * 3)
    - right_hand_pose: 45 (15 joints * 3)
    """
    
    if len(pose_165d) != 165:
        raise ValueError(f"Expected 165D pose, got {len(pose_165d)}D")
    
    # Ensure pose values are reasonable (not NaN or extreme values)
    if np.any(np.isnan(pose_165d)):
        print("WARNING: NaN values detected in pose, replacing with zeros")
        pose_165d = np.nan_to_num(pose_165d)
    
    if np.any(np.abs(pose_165d) > 10):  # Poses should typically be in [-π, π]
        print(f"WARNING: Extreme pose values detected: min={pose_165d.min():.3f}, max={pose_165d.max():.3f}")
    
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
    }

def create_neutral_pose():
    """Create a neutral SMPL-X pose (all zeros)"""
    return {
        'global_orient': torch.zeros(1, 3).float(),
        'body_pose': torch.zeros(1, 63).float(), 
        'jaw_pose': torch.zeros(1, 3).float(),
        'leye_pose': torch.zeros(1, 3).float(),
        'reye_pose': torch.zeros(1, 3).float(),
        'left_hand_pose': torch.zeros(1, 45).float(),
        'right_hand_pose': torch.zeros(1, 45).float(),
    }

def find_corresponding_beat2_file(runtime_sample_path, beat2_dir):
    """
    Try to find the original BEAT2 file that corresponds to a runtime sample.
    This is a best-effort approach since the exact mapping isn't clear.
    """
    sample_files = []
    for root, dirs, files in os.walk(beat2_dir):
        for file in files:
            if file.endswith('.npz'):
                sample_files.append(os.path.join(root, file))
    
    # Just return the first file for now - in a real scenario you'd need proper mapping
    return sample_files[0] if sample_files else None

def normalize_pose_single_frame(poses):
    """
    Normalize poses the same way BEAT2 dataset does for single frames
    Uses global SMPL-X pose range (-π to π) for consistency with training
    """
    pose_min = -3.14159
    pose_max = 3.14159
    normalized = 2.0 * (poses - pose_min) / (pose_max - pose_min) - 1.0
    return normalized.astype(np.float32), pose_min, pose_max

def denormalize_poses(normalized_poses, seq_min, seq_max):
    """
    Denormalize poses using the original sequence min/max
    pose = (normalized + 1.0) * (seq_max - seq_min) / 2.0 + seq_min
    """
    denormalized = (normalized_poses + 1.0) * (seq_max - seq_min) / 2.0 + seq_min
    return denormalized

def visualize_pose(model, pose_params, plotting_module='pyrender', plot_joints=True, title="Pose"):
    """Visualize a single pose using SMPL-X model"""

    try:
        output = model(**pose_params, return_verts=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        print(f'{title} - Vertices shape: {vertices.shape}, Joints shape: {joints.shape}')
    except Exception as e:
        print(f"ERROR: Failed to generate mesh for {title}")
        print(f"Error: {e}")
        print("Check if pose parameters are valid")
        return

    if plotting_module == 'pyrender':
        import pyrender
        import trimesh

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, model.faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)

        if plot_joints:
            sm = trimesh.creation.uv_sphere(radius=0.005)
            sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            tfs = np.tile(np.eye(4), (len(joints), 1, 1))
            tfs[:, :3, 3] = joints
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        print(f"Showing {title}...")
        pyrender.Viewer(scene, use_raymond_lighting=True, window_title=title)


def visualize_multiple_poses(model, pose_params_list, titles, plot_joints=True, spacing=1.5):
    """
    Visualize multiple poses side by side in a single view using pyrender.

    Args:
        model: SMPL-X model
        pose_params_list: List of pose parameter dictionaries
        titles: List of titles for each pose
        plot_joints: Whether to plot joints
        spacing: Horizontal spacing between avatars (in meters)
    """
    import pyrender
    import trimesh

    scene = pyrender.Scene()

    # Define colors for different avatars
    colors = [
        [0.8, 0.2, 0.2, 0.8],  # Red - Original
        [0.2, 0.8, 0.2, 0.8],  # Green - Level 1
        [0.2, 0.2, 0.8, 0.8],  # Blue - Level 2
        [0.8, 0.8, 0.2, 0.8],  # Yellow - Level 3
        [0.8, 0.2, 0.8, 0.8],  # Magenta - Level 4
        [0.2, 0.8, 0.8, 0.8],  # Cyan - Level 5
    ]

    # Joint colors (slightly brighter versions)
    joint_colors = [
        [1.0, 0.3, 0.3, 1.0],  # Bright red
        [0.3, 1.0, 0.3, 1.0],  # Bright green
        [0.3, 0.3, 1.0, 1.0],  # Bright blue
        [1.0, 1.0, 0.3, 1.0],  # Bright yellow
        [1.0, 0.3, 1.0, 1.0],  # Bright magenta
        [0.3, 1.0, 1.0, 1.0],  # Bright cyan
    ]

    num_poses = len(pose_params_list)

    # Calculate starting x position to center the avatars
    total_width = (num_poses - 1) * spacing
    start_x = -total_width / 2

    print(f"\nGenerating {num_poses} avatars...")

    for i, (pose_params, title) in enumerate(zip(pose_params_list, titles)):
        try:
            output = model(**pose_params, return_verts=True)
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            joints = output.joints.detach().cpu().numpy().squeeze()

            print(f'{title} - Vertices: {vertices.shape}, Joints: {joints.shape}')

            # Offset avatar horizontally
            x_offset = start_x + i * spacing
            vertices_offset = vertices.copy()
            vertices_offset[:, 0] += x_offset  # Offset in X direction

            # Create mesh with colored vertices
            vertex_colors = np.ones([vertices_offset.shape[0], 4]) * colors[i % len(colors)]
            tri_mesh = trimesh.Trimesh(vertices_offset, model.faces, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene.add(mesh)

            # Add joints if requested
            if plot_joints:
                joints_offset = joints.copy()
                joints_offset[:, 0] += x_offset

                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = joint_colors[i % len(joint_colors)]
                tfs = np.tile(np.eye(4), (len(joints_offset), 1, 1))
                tfs[:, :3, 3] = joints_offset
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                scene.add(joints_pcl)

            # Add text label above avatar (using a simple sphere as marker)
            # Text rendering is limited in pyrender, so we use a marker instead
            label_sphere = trimesh.creation.uv_sphere(radius=0.02)
            label_sphere.visual.vertex_colors = colors[i % len(colors)]
            label_pos = np.array([x_offset, 0.0, 1.8])  # Above head
            label_mesh = pyrender.Mesh.from_trimesh(label_sphere)
            label_node = scene.add(label_mesh, pose=trimesh.transformations.translation_matrix(label_pos))

        except Exception as e:
            print(f"ERROR: Failed to generate mesh for {title}")
            print(f"Error: {e}")
            continue

    # Create title string
    title_str = " | ".join(titles)
    print(f"\nShowing: {title_str}")
    print(f"Color legend:")
    for i, title in enumerate(titles):
        color = colors[i % len(colors)]
        print(f"  {title}: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")

    # Show the scene
    pyrender.Viewer(scene, use_raymond_lighting=True, window_title="Multi-Level Comparison")

def main():
    parser = argparse.ArgumentParser(description='Visualize HR-VAE pose reconstruction comparison')
    parser.add_argument('--model-folder', default='models_smplx_v1_1/models', type=str,
                        help='Path to SMPL-X model folder')
    parser.add_argument('--samples-dir', default='checkpoint/beat2_poses/0/vae/runtime_samples',
                        type=str, help='Path to runtime samples directory')
    parser.add_argument('--sample-idx', default=-1, type=int,
                        help='Which sample to visualize (-1 for latest, 0-based index)')
    parser.add_argument('--frame-idx', default=0, type=int,
                        help='Which frame in the sequence to visualize (0-based index)')
    parser.add_argument('--gender', default='neutral', type=str,
                        help='SMPL-X model gender')
    parser.add_argument('--plotting-module', default='pyrender', type=str,
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='Plotting module to use')
    parser.add_argument('--plot-joints', default=True, type=bool,
                        help='Whether to plot joints')
    parser.add_argument('--show-neutral', action='store_true',
                        help='Show neutral pose first for reference')
    parser.add_argument('--beat2-dir', default='BEAT2/beat_english_v2.0.0/smplxflame_30', type=str,
                        help='Path to BEAT2 pose files')
    parser.add_argument('--beat2-frame-idx', default=0, type=int,
                        help='Which frame from BEAT2 file to use')
    parser.add_argument('--use-real-beat2', action='store_true',
                        help='Compare with real BEAT2 poses instead of normalized originals')
    parser.add_argument('--multi-view', action='store_true',
                        help='Show all levels side-by-side in a single view')
    parser.add_argument('--spacing', default=1.5, type=float,
                        help='Spacing between avatars in multi-view mode (in meters)')
    parser.add_argument('--batch-idx', default=1, type=int,
                        help='Which batch sample to visualize (0-based index)')

    args = parser.parse_args()
    
    # Load SMPL-X model
    try:
        model = smplx.create(args.model_folder, 
                             model_type='smplx',
                             gender=args.gender, 
                             use_face_contour=False,
                             use_pca=False,  # Use full hand pose (45D) instead of PCA (6D)
                             ext='npz')
        print(f"Successfully loaded SMPL-X model from {args.model_folder}")
    except Exception as e:
        print(f"ERROR: Failed to load SMPL-X model from {args.model_folder}")
        print(f"Error: {e}")
        print("Make sure the model folder contains SMPLX model files (.npz)")
        return
    
    # Load runtime samples
    try:
        samples = load_runtime_samples(args.samples_dir)
        if not samples:
            print(f"ERROR: No runtime samples found in {args.samples_dir}")
            print("Make sure you have trained the model and generated runtime samples")
            return
        print(f"Found {len(samples)} runtime samples")
    except Exception as e:
        print(f"ERROR: Failed to load runtime samples from {args.samples_dir}")
        print(f"Error: {e}")
        return
    
    if args.sample_idx == -1:
        args.sample_idx = len(samples) - 1  # Use latest sample
        print(f"Using latest sample: {samples[args.sample_idx]['filename']}")
    elif args.sample_idx >= len(samples):
        print(f"Sample index {args.sample_idx} out of range. Available samples: {len(samples)}")
        return
    
    sample = samples[args.sample_idx]
    print(f"Visualizing sample: {sample['filename']}")
    print(f"Original shape: {sample['original'].shape}")
    print(f"Reconstructed shape: {sample['reconstructed'].shape}")
    
    # Check batch_idx
    batch_size = sample['original'].shape[0]
    if args.batch_idx >= batch_size:
        print(f"Batch index {args.batch_idx} out of range. Available samples in batch: {batch_size}")
        args.batch_idx = 0
        print(f"Using batch index 0 instead")

    # Extract specific frame
    if args.frame_idx >= sample['original'].shape[1]:
        print(f"Frame index {args.frame_idx} out of range. Available frames: {sample['original'].shape[1]}")
        return

    original_frame = sample['original'][args.batch_idx, args.frame_idx, :]  # (165,)
    reconstructed_frame = sample['reconstructed'][args.batch_idx, args.frame_idx, :]  # (165,)
    
    # Debug: Check pose values
    print(f"Original pose stats - min: {original_frame.min():.3f}, max: {original_frame.max():.3f}, mean: {original_frame.mean():.3f}")
    print(f"Reconstructed pose stats - min: {reconstructed_frame.min():.3f}, max: {reconstructed_frame.max():.3f}, mean: {reconstructed_frame.mean():.3f}")
    print(f"Global orient (first 3): {original_frame[:3]}")
    print(f"Body pose sample (3-9): {original_frame[3:9]}")
    
    # Check for corrupted data (all -1.0 values indicate old corrupted samples)
    if np.all(original_frame == -1.0):
        print("\nWARNING: Original data appears corrupted (all -1.0 values)!")
        print("This suggests using an old runtime sample from before the dataset fix.")
        print("Consider retraining or using --use-real-beat2 flag.")
    
    if np.allclose(reconstructed_frame, -1.0, atol=0.01):
        print("\nWARNING: Reconstructed data appears corrupted (near -1.0 values)!")
        print("This suggests the model was trained on corrupted data or needs retraining.")
    
    # Show neutral pose for reference if requested
    if args.show_neutral:
        print("\n=== NEUTRAL POSE (REFERENCE) ===")
        neutral_params = create_neutral_pose()
        visualize_pose(model, neutral_params, args.plotting_module, args.plot_joints, "Neutral Pose")
    
    if args.use_real_beat2:
        print("\n=== LOADING REAL BEAT2 DATA ===")
        # Load real BEAT2 pose file
        beat2_file = find_corresponding_beat2_file(args.samples_dir, args.beat2_dir)
        if not beat2_file:
            print("No BEAT2 files found! Using normalized original instead.")
            args.use_real_beat2 = False
        else:
            print(f"Using BEAT2 file: {os.path.basename(beat2_file)}")
            beat2_data = np.load(beat2_file)
            real_poses = beat2_data['poses']  # Shape: (T, 165)
            
            if args.beat2_frame_idx >= len(real_poses):
                args.beat2_frame_idx = 0
                print(f"Using frame 0 (max available: {len(real_poses)})")
            
            real_pose = real_poses[args.beat2_frame_idx]  # Shape: (165,)
            print(f"Real BEAT2 pose stats: min={real_pose.min():.3f}, max={real_pose.max():.3f}, mean={real_pose.mean():.3f}")
            
            # Test normalization/denormalization using same method as training
            normalized_real, pose_min, pose_max = normalize_pose_single_frame(real_pose)
            denormalized_real = denormalize_poses(normalized_real, pose_min, pose_max)
            
            print(f"Normalization round-trip MSE: {np.mean((real_pose - denormalized_real) ** 2):.10f}")
            
            # Denormalize reconstructed pose using same parameters
            reconstructed_denorm = denormalize_poses(reconstructed_frame, pose_min, pose_max)
            
            # Use real pose as original
            original_denorm = real_pose
            
            print(f"Using real BEAT2 pose as original comparison")
    
    if not args.use_real_beat2:
        print("Data appears to be in raw radians (not normalized), using directly")
        original_denorm = original_frame
        reconstructed_denorm = reconstructed_frame

    # Convert to SMPL-X parameters
    original_params = pose_to_smplx_params(original_denorm)
    reconstructed_params = pose_to_smplx_params(reconstructed_denorm)

    # Check if multi-view mode is enabled and per-level data is available
    if args.multi_view:
        # Collect all level reconstructions
        level_keys = sorted([k for k in sample.keys() if k.startswith('reconstructed_level_')])

        if not level_keys:
            print("\nWARNING: --multi-view requested but no per-level reconstructions found!")
            print("Falling back to single-view mode...")
            args.multi_view = False
        else:
            # Prepare poses for multi-view
            all_poses = [original_denorm]
            all_titles = ["Original"]

            for key in level_keys:
                level_num = int(key.split('_')[-1])
                level_data = sample[key][args.batch_idx, args.frame_idx, :]  # (165,)
                all_poses.append(level_data)
                all_titles.append(f"Level {level_num}")

            # Add full reconstruction at the end
            all_poses.append(reconstructed_denorm)
            all_titles.append("Full")

            # Convert all to SMPL-X parameters
            all_params = [pose_to_smplx_params(pose) for pose in all_poses]

            # Calculate MSE for each level
            print("\n" + "=" * 60)
            print("MULTI-LEVEL COMPARISON")
            print("=" * 60)
            for i, (pose, title) in enumerate(zip(all_poses, all_titles)):
                if i == 0:
                    continue  # Skip original
                mse = np.mean((original_denorm - pose) ** 2)
                print(f"{title:15s} MSE: {mse:.6f}")
            print("=" * 60)

            # Show all in one view
            visualize_multiple_poses(model, all_params, all_titles,
                                   plot_joints=args.plot_joints,
                                   spacing=args.spacing)

    if not args.multi_view:
        # Original single-view visualization
        if args.use_real_beat2:
            print("\n=== REAL BEAT2 POSE ===")
            visualize_pose(model, original_params, args.plotting_module, args.plot_joints, "Real BEAT2 Pose")
        else:
            print("\n=== ORIGINAL POSE ===")
            visualize_pose(model, original_params, args.plotting_module, args.plot_joints, "Original Pose")

        print("\n=== RECONSTRUCTED POSE ===")
        visualize_pose(model, reconstructed_params, args.plotting_module, args.plot_joints, "Reconstructed Pose")

        # Calculate reconstruction error
        mse_denormalized = np.mean((original_denorm - reconstructed_denorm) ** 2)
        print(f"\nReconstruction MSE: {mse_denormalized:.6f}")

        # Joint-wise comparison (first 10 joints)
        print("Joint angle differences (first 10 dims):")
        diff = np.abs(original_denorm[:10] - reconstructed_denorm[:10])
        for i, d in enumerate(diff):
            print(f"  Dim {i}: {d:.3f} radians ({np.degrees(d):.1f}°)")

if __name__ == '__main__':
    main()