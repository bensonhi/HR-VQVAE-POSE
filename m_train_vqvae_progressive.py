import torch
from torch import nn
from tqdm import tqdm


# SMPLX body part definitions (165D pose)
BODY_PARTS = {
    'global_orient': (0, 3),
    'body': (3, 66),           # Main body pose
    'jaw': (66, 69),
    'eyes': (69, 75),          # Left and right eyes
    'hands': (75, 165),        # Both hands
}


def train_progressive(folder_name, epoch_num, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, run_num,
                     use_smplx_loss=False, pose_loss_weight=1.0, vertex_loss_weight=1.0, joint_loss_weight=1.0,
                     level_1_weight=1.0, level_2_weight=1.0, level_3_weight=1.0):
    """
    Progressive training loop with level-specific losses.

    Level-specific loss strategy:
    - Level 1: Focus on body pose (global_orient + body)
    - Level 2: Focus on hands
    - Level 3: Overall reconstruction + SMPLX geometry losses

    Args:
        folder_name: Model folder name
        epoch_num: Current epoch number
        loader: DataLoader
        model: VQ-VAE model (must support return_intermediate=True)
        writer: TensorBoard writer
        do_sample: Whether to sample this epoch
        sampler: Sampling function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (cuda/cpu)
        dataset_name: Dataset name
        run_num: Run number
        use_smplx_loss: Whether to use SMPLX geometry losses (only for level 3)
        pose_loss_weight: Weight for pose reconstruction loss
        vertex_loss_weight: Weight for vertex reconstruction loss
        joint_loss_weight: Weight for joint reconstruction loss
        level_1_weight: Weight for level 1 loss (default: 1.0)
        level_2_weight: Weight for level 2 loss (default: 1.0)
        level_3_weight: Weight for level 3 loss (default: 1.0)
    """
    loader = tqdm(loader)

    criterion = nn.MSELoss()
    latent_loss_weight = 0.25

    # Track losses
    level_1_loss_sum = 0
    level_2_loss_sum = 0
    level_3_pose_sum = 0
    level_3_vertex_sum = 0
    level_3_joint_sum = 0
    total_loss_sum = 0
    mse_n = 0

    for i, (data, label) in enumerate(loader):
        model.zero_grad()

        # Handle dict-based data loader (with ground truth geometry)
        if isinstance(data, dict):
            poses = data['poses'].to(device)  # (batch, seq_len, 165)
            gt_joints = data.get('gt_joints', None)
            gt_vertices = data.get('gt_vertices', None)

            if gt_joints is not None:
                gt_joints = gt_joints.to(device)
            if gt_vertices is not None:
                gt_vertices = gt_vertices.to(device)
        else:
            # Legacy format (backward compatibility)
            poses = data.to(device)
            gt_joints = None
            gt_vertices = None

        # Forward pass with intermediate outputs
        result = model(poses, compute_geometry=use_smplx_loss, return_intermediate=True)

        if use_smplx_loss:
            intermediate_outputs, latent_loss, pred_vertices, pred_joints = result
        else:
            intermediate_outputs, latent_loss = result
            pred_vertices, pred_joints = None, None

        # Ensure we have enough levels
        if len(intermediate_outputs) < 3:
            raise ValueError(f"Expected at least 3 levels, got {len(intermediate_outputs)}")

        level_1_output = intermediate_outputs[0]  # After level 1
        level_2_output = intermediate_outputs[1]  # After level 2
        level_3_output = intermediate_outputs[2]  # After level 3 (final)

        # ==================== Level 1 Loss: Body Pose ====================
        # Focus on global orientation and body pose (dims 0-66)
        body_parts_level_1 = slice(BODY_PARTS['global_orient'][0], BODY_PARTS['body'][1])
        level_1_loss = criterion(
            level_1_output[:, :, body_parts_level_1],
            poses[:, :, body_parts_level_1]
        )

        # ==================== Level 2 Loss: Hands ====================
        # Focus on hand pose (dims 75-165)
        hands_parts = slice(BODY_PARTS['hands'][0], BODY_PARTS['hands'][1])
        level_2_loss = criterion(
            level_2_output[:, :, hands_parts],
            poses[:, :, hands_parts]
        )

        # ==================== Level 3 Loss: Overall + Geometry ====================
        # Full pose reconstruction
        level_3_pose_loss = criterion(level_3_output, poses)

        # SMPLX geometry losses (only for level 3)
        vertex_recon_loss = torch.tensor(0.0, device=device)
        joint_recon_loss = torch.tensor(0.0, device=device)

        if use_smplx_loss and pred_vertices is not None and gt_vertices is not None:
            vertex_recon_loss = criterion(pred_vertices, gt_vertices)

        if use_smplx_loss and pred_joints is not None and gt_joints is not None:
            joint_recon_loss = criterion(pred_joints, gt_joints)

        # ==================== Combine All Losses ====================
        latent_loss = latent_loss.mean()

        total_loss = (
            level_1_weight * level_1_loss +
            level_2_weight * level_2_loss +
            level_3_weight * (
                pose_loss_weight * level_3_pose_loss +
                vertex_loss_weight * vertex_recon_loss +
                joint_loss_weight * joint_recon_loss
            ) +
            latent_loss_weight * latent_loss
        )

        # Backward pass
        total_loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # Track metrics
        batch_size = poses.shape[0]
        level_1_loss_sum += level_1_loss.item() * batch_size
        level_2_loss_sum += level_2_loss.item() * batch_size
        level_3_pose_sum += level_3_pose_loss.item() * batch_size
        level_3_vertex_sum += vertex_recon_loss.item() * batch_size
        level_3_joint_sum += joint_recon_loss.item() * batch_size
        total_loss_sum += total_loss.item() * batch_size
        mse_n += batch_size

        lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        desc = (
            f'epoch: {epoch_num + 1}; '
            f'L1(body): {level_1_loss.item():.5f}; '
            f'L2(hands): {level_2_loss.item():.5f}; '
            f'L3(pose): {level_3_pose_loss.item():.5f}; '
        )
        if use_smplx_loss:
            desc += (
                f'L3(vert): {vertex_recon_loss.item():.5f}; '
                f'L3(joint): {joint_recon_loss.item():.5f}; '
            )
        desc += (
            f'latent: {latent_loss.item():.3f}; '
            f'total: {total_loss.item():.5f}; '
            f'lr: {lr:.5f}'
        )
        loader.set_description(desc)

    # Log to TensorBoard
    writer.add_scalar('Loss/total', total_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/level_1_body', level_1_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/level_2_hands', level_2_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/level_3_pose', level_3_pose_sum / mse_n, epoch_num)

    if use_smplx_loss:
        writer.add_scalar('Loss/level_3_vertex', level_3_vertex_sum / mse_n, epoch_num)
        writer.add_scalar('Loss/level_3_joint', level_3_joint_sum / mse_n, epoch_num)

    # Sample if needed (use final output)
    if do_sample:
        sampler(folder_name, model, poses, dataset_name, run_num, epoch_num, poses.shape[0])
