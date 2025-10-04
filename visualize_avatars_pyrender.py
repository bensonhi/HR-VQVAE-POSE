"""
Visualize original vs reconstructed avatars using pyrender
Left: Original avatar
Right: Reconstructed avatar (VQ-VAE + Differential IK)

Reads directly from checkpoint runtime samples
"""
import os
import argparse
import numpy as np
import torch
import smplx
import pyrender
import trimesh
import glob


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
    """Convert a flat vector (J*3,) into (J, 3) joints array."""
    v = np.array(vec).reshape(-1)
    if v.size % 3 != 0:
        raise ValueError(f"Vector length {v.size} is not divisible by 3")
    j = v.size // 3
    return v.reshape(j, 3)


def differential_ik(target_joints, smplx_model, initial_params=None, max_iters=50, device='cuda'):
    """
    Optimize SMPL-X parameters to fit target joint positions
    """
    # Move target to device
    target_joints_tensor = torch.FloatTensor(target_joints).unsqueeze(0).to(device)  # (1, 127, 3)

    # Initialize parameters
    if initial_params is None:
        # T-pose initialization
        global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
        body_pose = torch.zeros(1, 63, device=device, requires_grad=True)
        jaw_pose = torch.zeros(1, 3, device=device, requires_grad=True)
        leye_pose = torch.zeros(1, 3, device=device, requires_grad=True)
        reye_pose = torch.zeros(1, 3, device=device, requires_grad=True)
        left_hand_pose = torch.zeros(1, 45, device=device, requires_grad=True)
        right_hand_pose = torch.zeros(1, 45, device=device, requires_grad=True)
        transl = torch.zeros(1, 3, device=device, requires_grad=True)
    else:
        global_orient = torch.FloatTensor(initial_params['global_orient']).reshape(1, 3).to(device).requires_grad_(True)
        body_pose = torch.FloatTensor(initial_params['body_pose']).reshape(1, 63).to(device).requires_grad_(True)
        jaw_pose = torch.FloatTensor(initial_params['jaw_pose']).reshape(1, 3).to(device).requires_grad_(True)
        leye_pose = torch.FloatTensor(initial_params['leye_pose']).reshape(1, 3).to(device).requires_grad_(True)
        reye_pose = torch.FloatTensor(initial_params['reye_pose']).reshape(1, 3).to(device).requires_grad_(True)
        left_hand_pose = torch.FloatTensor(initial_params['left_hand_pose']).reshape(1, 45).to(device).requires_grad_(True)
        right_hand_pose = torch.FloatTensor(initial_params['right_hand_pose']).reshape(1, 45).to(device).requires_grad_(True)
        transl = torch.FloatTensor(initial_params.get('transl', np.zeros(3))).reshape(1, 3).to(device).requires_grad_(True)

    # Optimizer
    from torch.optim import LBFGS
    opt_params = [global_orient, body_pose, jaw_pose, leye_pose, reye_pose,
                  left_hand_pose, right_hand_pose, transl]
    optimizer = LBFGS(opt_params, lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()

        # Forward pass through SMPL-X
        output = smplx_model(
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            return_verts=True
        )

        predicted_joints = output.joints  # (1, 127, 3)

        # Joint position loss
        joint_loss = torch.mean((predicted_joints - target_joints_tensor) ** 2)

        # Regularization to keep poses natural
        pose_reg = 0.001 * (torch.mean(body_pose ** 2) +
                            torch.mean(left_hand_pose ** 2) +
                            torch.mean(right_hand_pose ** 2))

        total_loss = joint_loss + pose_reg

        total_loss.backward()

        return total_loss

    # Run optimization
    for i in range(max_iters):
        loss = optimizer.step(closure)
        if loss.item() < 1e-6:
            break

    # Extract optimized parameters
    result = {
        'global_orient': global_orient.detach().cpu().numpy().squeeze(),
        'body_pose': body_pose.detach().cpu().numpy().squeeze(),
        'jaw_pose': jaw_pose.detach().cpu().numpy().squeeze(),
        'leye_pose': leye_pose.detach().cpu().numpy().squeeze(),
        'reye_pose': reye_pose.detach().cpu().numpy().squeeze(),
        'left_hand_pose': left_hand_pose.detach().cpu().numpy().squeeze(),
        'right_hand_pose': right_hand_pose.detach().cpu().numpy().squeeze(),
        'transl': transl.detach().cpu().numpy().squeeze(),
    }

    return result


def joints_to_smplx_mesh(joints, smplx_model, device='cuda'):
    """Convert joint positions to SMPL-X mesh using differential IK"""
    print("  Running differential IK...")
    ik_params = differential_ik(joints, smplx_model, initial_params=None, max_iters=50, device=device)

    # Forward pass to get mesh
    with torch.no_grad():
        output = smplx_model(
            global_orient=torch.FloatTensor(ik_params['global_orient']).reshape(1, 3).to(device),
            body_pose=torch.FloatTensor(ik_params['body_pose']).reshape(1, 63).to(device),
            jaw_pose=torch.FloatTensor(ik_params['jaw_pose']).reshape(1, 3).to(device),
            leye_pose=torch.FloatTensor(ik_params['leye_pose']).reshape(1, 3).to(device),
            reye_pose=torch.FloatTensor(ik_params['reye_pose']).reshape(1, 3).to(device),
            left_hand_pose=torch.FloatTensor(ik_params['left_hand_pose']).reshape(1, 45).to(device),
            right_hand_pose=torch.FloatTensor(ik_params['right_hand_pose']).reshape(1, 45).to(device),
            transl=torch.FloatTensor(ik_params['transl']).reshape(1, 3).to(device),
            return_verts=True
        )

    vertices = output.vertices.cpu().numpy().squeeze()
    return vertices


def main():
    parser = argparse.ArgumentParser(description='Visualize SMPL-X avatars (original vs reconstructed)')
    parser.add_argument('--samples-dir', default='checkpoint/beat2_poses/0/vqvae/runtime_samples', type=str,
                        help='Directory with runtime sample .npz files')
    parser.add_argument('--sample-idx', default=-1, type=int,
                        help='Which sample file to use (-1 = latest)')
    parser.add_argument('--batch-idx', default=0, type=int,
                        help='Batch index within the sample file')
    parser.add_argument('--frame-idx', default=0, type=int,
                        help='Frame index (time) within the sample')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("="*60)
    print("Visualizing Avatars with Pyrender")
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

    # Convert to joints
    original_joints = vector_to_joints(orig_vec)
    reconstructed_joints = vector_to_joints(recon_vec)

    print(f"\nBatch: {args.batch_idx}, Frame: {args.frame_idx}")
    print(f"Original joints: {original_joints.shape}")
    print(f"Reconstructed joints: {reconstructed_joints.shape}")

    # Load SMPL-X model
    print("\nLoading SMPL-X model...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    smplx_model = smplx.create(
        'models_smplx_v1_1/models',
        model_type='smplx',
        gender='neutral',
        use_face_contour=False,
        use_pca=False,
        ext='npz'
    ).to(device)

    # Convert joints to meshes using differential IK
    print("\nGenerating original avatar mesh...")
    original_vertices = joints_to_smplx_mesh(original_joints, smplx_model, device=device)

    print("Generating reconstructed avatar mesh...")
    recon_vertices = joints_to_smplx_mesh(reconstructed_joints, smplx_model, device=device)

    # Create meshes
    print("\nCreating visualization...")

    # Original mesh (left side) - shift left by 1.5 units
    original_mesh = trimesh.Trimesh(
        vertices=original_vertices - np.array([1.5, 0, 0]),
        faces=smplx_model.faces,
        process=False
    )

    # Reconstructed mesh (right side) - shift right by 1.5 units
    recon_mesh = trimesh.Trimesh(
        vertices=recon_vertices + np.array([1.5, 0, 0]),
        faces=smplx_model.faces,
        process=False
    )

    # Set colors
    original_mesh.visual.vertex_colors = [200, 200, 250, 255]  # Light blue
    recon_mesh.visual.vertex_colors = [250, 200, 200, 255]  # Light red

    # Create pyrender scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # Add meshes
    scene.add(pyrender.Mesh.from_trimesh(original_mesh, smooth=True))
    scene.add(pyrender.Mesh.from_trimesh(recon_mesh, smooth=True))

    # Add lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # Add camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)

    # Render
    print("\nRendering scene...")
    print("\n" + "="*60)
    print("Left (blue): Original avatar")
    print("Right (red): Reconstructed avatar (VQ-VAE + Differential IK)")
    print("="*60)

    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)


if __name__ == '__main__':
    main()
