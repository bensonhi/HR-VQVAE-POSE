import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def train(folder_name, epoch_num, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, run_num,
          use_smplx_loss=False, pose_loss_weight=1.0, vertex_loss_weight=1.0, joint_loss_weight=1.0,
          kl_anneal_epochs=100, max_kl_weight=0.05):
    """
    Training loop with optional SMPLX-based multi-loss supervision.
    Supports variable-length batches with audio and gesture type conditioning.
    """
    loader = tqdm(loader)

    # latent_loss_weight: KL annealing
    if kl_anneal_epochs > 0:
        latent_loss_weight = min(max_kl_weight, (epoch_num / kl_anneal_epochs) * max_kl_weight)
    else:
        latent_loss_weight = max_kl_weight

    # Track losses
    pose_mse_sum = 0
    vertex_mse_sum = 0
    joint_mse_sum = 0
    total_loss_sum = 0
    mse_n = 0

    for i, (data, label) in enumerate(loader):
        model.zero_grad()

        # Handle dict-based data loader (new variable-length format or legacy)
        if isinstance(data, dict):
            poses = data['poses'].to(device)  # (batch, seq_len, 165)

            # New conditioning signals
            gesture_type = data.get('gesture_type', None)
            if gesture_type is not None:
                gesture_type = gesture_type.to(device)

            lengths = data.get('lengths', None)
            if lengths is not None:
                lengths = lengths.to(device)

            padding_mask = data.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            audio = data.get('audio', None)
            if audio is not None:
                audio = audio.to(device)

            gt_joints = data.get('gt_joints', None)
            gt_vertices = data.get('gt_vertices', None)
            if gt_joints is not None:
                gt_joints = gt_joints.to(device)
            if gt_vertices is not None:
                gt_vertices = gt_vertices.to(device)
        else:
            # Legacy format
            poses = data.to(device)
            gesture_type = None
            lengths = None
            padding_mask = None
            audio = None
            gt_joints = None
            gt_vertices = None

        # Forward pass through model
        if use_smplx_loss:
            model_has_smplx = False
            if hasattr(model, 'module'):
                model_has_smplx = hasattr(model.module, 'use_smplx') and model.module.use_smplx
            elif hasattr(model, 'use_smplx'):
                model_has_smplx = model.use_smplx

            if model_has_smplx:
                result = model(poses, padding_mask=padding_mask, gesture_type=gesture_type,
                              lengths=lengths, audio_features=audio, compute_geometry=True)
                reconstructed_poses, latent_loss, pred_vertices, pred_joints = result
            else:
                reconstructed_poses, latent_loss = model(poses, padding_mask=padding_mask,
                                                         gesture_type=gesture_type,
                                                         lengths=lengths, audio_features=audio)
                pred_vertices, pred_joints = None, None
        else:
            result = model(poses, padding_mask=padding_mask, gesture_type=gesture_type,
                          lengths=lengths, audio_features=audio)
            if len(result) == 4:
                reconstructed_poses, latent_loss, pred_vertices, pred_joints = result
                pred_vertices, pred_joints = None, None
            else:
                reconstructed_poses, latent_loss = result
                pred_vertices, pred_joints = None, None

        # Compute masked reconstruction loss
        if padding_mask is not None:
            valid_mask = ~padding_mask  # (B, T) True where valid
            valid_mask_expanded = valid_mask.unsqueeze(-1)  # (B, T, 1)
            num_valid = valid_mask.sum().clamp(min=1)
            pose_recon_loss = (((reconstructed_poses - poses) ** 2) * valid_mask_expanded).sum() / (num_valid * poses.shape[-1])
        else:
            pose_recon_loss = F.mse_loss(reconstructed_poses, poses)

        # Compute geometry losses if available
        vertex_recon_loss = torch.tensor(0.0, device=device)
        joint_recon_loss = torch.tensor(0.0, device=device)

        if use_smplx_loss and pred_vertices is not None and gt_vertices is not None:
            vertex_recon_loss = F.mse_loss(pred_vertices, gt_vertices)

        if use_smplx_loss and pred_joints is not None and gt_joints is not None:
            joint_recon_loss = F.mse_loss(pred_joints, gt_joints)

        # Combine losses
        latent_loss = latent_loss.mean()
        total_loss = (pose_loss_weight * pose_recon_loss +
                     vertex_loss_weight * vertex_recon_loss +
                     joint_loss_weight * joint_recon_loss +
                     latent_loss_weight * latent_loss)

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # Track metrics
        batch_size = poses.shape[0]
        pose_mse_sum += pose_recon_loss.item() * batch_size
        vertex_mse_sum += vertex_recon_loss.item() * batch_size
        joint_mse_sum += joint_recon_loss.item() * batch_size
        total_loss_sum += total_loss.item() * batch_size
        mse_n += batch_size

        lr = optimizer.param_groups[0]['lr']

        # Compute running averages for display
        avg_pose = pose_mse_sum / mse_n
        avg_vertex = vertex_mse_sum / mse_n
        avg_joint = joint_mse_sum / mse_n
        avg_total = total_loss_sum / mse_n

        desc = (
            f'epoch: {epoch_num + 1}; '
            f'pose: {avg_pose:.5f}; '
        )
        if use_smplx_loss:
            desc += (
                f'vert: {avg_vertex:.5f}; '
                f'joint: {avg_joint:.5f}; '
            )
        desc += (
            f'total: {avg_total:.5f}; '
            f'\u03b2: {latent_loss_weight:.4f}; '
            f'lr: {lr:.5f}'
        )
        loader.set_description(desc)

    # Log to TensorBoard
    writer.add_scalar('Loss/train', pose_mse_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/pose_mse', pose_mse_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/total', total_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/kl_weight', latent_loss_weight, epoch_num)

    if use_smplx_loss:
        writer.add_scalar('Loss/vertex_mse', vertex_mse_sum / mse_n, epoch_num)
        writer.add_scalar('Loss/joint_mse', joint_mse_sum / mse_n, epoch_num)

    # Sample if needed
    if do_sample:
        sampler(folder_name, model, poses, dataset_name, run_num, epoch_num, poses.shape[0],
                padding_mask=padding_mask, gesture_type=gesture_type,
                lengths=lengths, audio_features=audio)

    # Return average total loss for early stopping
    return total_loss_sum / mse_n
