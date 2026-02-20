import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from m_smplx_layer import SMPLX_JOINT_INDICES, SMPLX_VERTEX_INDICES


# ============================================================================
# SMPLX body part definitions (165D axis-angle pose)
# ============================================================================
BODY_PARTS_POSE = {
    'face': (66, 75),          # Jaw (66-69) + eyes (69-75)
    'body': (0, 66),           # Global orient (0-3) + body pose (3-66)
    'hands': (75, 165),        # Left hand (75-120) + right hand (120-165)
}


# ============================================================================
# Helper Functions for Body Part Geometry Extraction
# ============================================================================

def extract_body_part_geometry(vertices, joints, part_name):
    """
    Extract vertices and joints for a specific body part.

    Args:
        vertices: (batch, seq_len, num_vertices, 3) or None
        joints: (batch, seq_len, num_joints, 3) or None
        part_name: 'face', 'body', or 'hands'

    Returns:
        part_vertices: Vertices for this body part
        part_joints: Joints for this body part
    """
    if vertices is not None:
        vertex_indices = SMPLX_VERTEX_INDICES[part_name]
        part_vertices = vertices[:, :, vertex_indices, :]
    else:
        part_vertices = None

    if joints is not None:
        joint_indices = SMPLX_JOINT_INDICES[part_name]
        part_joints = joints[:, :, joint_indices, :]
    else:
        part_joints = None

    return part_vertices, part_joints


# ============================================================================
# Loss Computation Functions
# ============================================================================

def _masked_mse(pred, target, valid_mask_expanded, num_valid, feat_dim):
    """Compute MSE only on valid (non-padded) frames."""
    return (((pred - target) ** 2) * valid_mask_expanded).sum() / (num_valid * feat_dim).clamp(min=1)


def _velocity_loss(pred, target, padding_mask=None):
    """Compute MSE on frame-to-frame differences (velocity).

    For masked sequences, a velocity frame is valid only if both adjacent frames are valid.
    """
    vel_pred = pred[:, 1:] - pred[:, :-1]   # (B, T-1, D)
    vel_gt = target[:, 1:] - target[:, :-1]  # (B, T-1, D)

    if padding_mask is not None:
        valid_mask = ~padding_mask  # (B, T)
        # Both frame t and frame t+1 must be valid for velocity at t
        vel_valid = valid_mask[:, 1:] & valid_mask[:, :-1]  # (B, T-1)
        vel_valid_expanded = vel_valid.unsqueeze(-1)  # (B, T-1, 1)
        num_valid = vel_valid.sum().clamp(min=1).float()
        feat_dim = pred.shape[-1]
        return (((vel_pred - vel_gt) ** 2) * vel_valid_expanded).sum() / (num_valid * feat_dim).clamp(min=1)
    else:
        return F.mse_loss(vel_pred, vel_gt)


def compute_level_local_loss(level_output, gt_poses, pred_vertices, pred_joints,
                             gt_vertices, gt_joints, part_name, criterion, device,
                             padding_mask=None, vel_weight=1.0):
    """
    Compute local loss for a specific level on a specific body part.
    Supports masked loss for variable-length sequences.
    """
    loss_dict = {}
    total_local_loss = torch.tensor(0.0, device=device)

    # Prepare masking
    if padding_mask is not None:
        valid_mask = ~padding_mask  # (B, T)
        num_valid = valid_mask.sum().clamp(min=1).float()
        valid_mask_2d = valid_mask.unsqueeze(-1)  # (B, T, 1)
        valid_mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
    else:
        valid_mask_2d = None

    # 1. Axis-angle loss on specific body part
    start_idx, end_idx = BODY_PARTS_POSE[part_name]
    pred_part = level_output[:, :, start_idx:end_idx]
    gt_part = gt_poses[:, :, start_idx:end_idx]

    if valid_mask_2d is not None:
        feat_dim = end_idx - start_idx
        axis_angle_loss = _masked_mse(pred_part, gt_part, valid_mask_2d, num_valid, feat_dim)
    else:
        axis_angle_loss = criterion(pred_part, gt_part)

    total_local_loss += axis_angle_loss
    loss_dict[f'axis_angle_{part_name}'] = axis_angle_loss.item()

    # 2. Velocity loss on specific body part
    if vel_weight > 0 and level_output.shape[1] > 1:
        vel_loss = _velocity_loss(pred_part, gt_part, padding_mask=padding_mask)
        total_local_loss += vel_weight * vel_loss
        loss_dict[f'vel_{part_name}'] = vel_loss.item()
    else:
        loss_dict[f'vel_{part_name}'] = 0.0

    # 3. Mesh (vertices) loss on specific body part
    if pred_vertices is not None and gt_vertices is not None:
        pred_part_verts, _ = extract_body_part_geometry(pred_vertices, None, part_name)
        gt_part_verts, _ = extract_body_part_geometry(gt_vertices, None, part_name)

        if valid_mask_4d is not None:
            n_verts = pred_part_verts.shape[2]
            mesh_loss = _masked_mse(pred_part_verts, gt_part_verts, valid_mask_4d, num_valid, n_verts * 3)
        else:
            mesh_loss = criterion(pred_part_verts, gt_part_verts)

        total_local_loss += mesh_loss
        loss_dict[f'mesh_{part_name}'] = mesh_loss.item()
    else:
        loss_dict[f'mesh_{part_name}'] = 0.0

    # 3. Joint position loss on specific body part
    if pred_joints is not None and gt_joints is not None:
        _, pred_part_joints = extract_body_part_geometry(None, pred_joints, part_name)
        _, gt_part_joints = extract_body_part_geometry(None, gt_joints, part_name)

        if valid_mask_4d is not None:
            n_joints = pred_part_joints.shape[2]
            joint_loss = _masked_mse(pred_part_joints, gt_part_joints, valid_mask_4d, num_valid, n_joints * 3)
        else:
            joint_loss = criterion(pred_part_joints, gt_part_joints)

        total_local_loss += joint_loss
        loss_dict[f'joint_{part_name}'] = joint_loss.item()
    else:
        loss_dict[f'joint_{part_name}'] = 0.0

    return total_local_loss, loss_dict


def compute_global_loss(final_output, gt_poses, pred_vertices, pred_joints,
                       gt_vertices, gt_joints, criterion, device,
                       pose_weight=1.0, mesh_weight=1.0, joint_weight=1.0,
                       padding_mask=None, vel_weight=1.0):
    """
    Compute global losses that backpropagate through all levels.
    Supports masked loss for variable-length sequences.
    """
    loss_dict = {}
    total_global_loss = torch.tensor(0.0, device=device)

    # Prepare masking
    if padding_mask is not None:
        valid_mask = ~padding_mask
        num_valid = valid_mask.sum().clamp(min=1).float()
        valid_mask_2d = valid_mask.unsqueeze(-1)
        valid_mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1)
    else:
        valid_mask_2d = None

    # 1. Full axis-angle reconstruction loss
    if valid_mask_2d is not None:
        feat_dim = final_output.shape[-1]
        pose_loss = _masked_mse(final_output, gt_poses, valid_mask_2d, num_valid, feat_dim)
    else:
        pose_loss = criterion(final_output, gt_poses)

    total_global_loss += pose_weight * pose_loss
    loss_dict['global_axis_angle'] = pose_loss.item()

    # 2. Full velocity loss
    if vel_weight > 0 and final_output.shape[1] > 1:
        vel_loss = _velocity_loss(final_output, gt_poses, padding_mask=padding_mask)
        total_global_loss += vel_weight * vel_loss
        loss_dict['global_vel'] = vel_loss.item()
    else:
        loss_dict['global_vel'] = 0.0

    # 3. Full mesh reconstruction loss
    if pred_vertices is not None and gt_vertices is not None:
        if valid_mask_4d is not None:
            n_verts = pred_vertices.shape[2]
            mesh_loss = _masked_mse(pred_vertices, gt_vertices, valid_mask_4d, num_valid, n_verts * 3)
        else:
            mesh_loss = criterion(pred_vertices, gt_vertices)

        total_global_loss += mesh_weight * mesh_loss
        loss_dict['global_mesh'] = mesh_loss.item()
    else:
        loss_dict['global_mesh'] = 0.0

    # 3. Full joint reconstruction loss
    if pred_joints is not None and gt_joints is not None:
        if valid_mask_4d is not None:
            n_joints = pred_joints.shape[2]
            joint_loss = _masked_mse(pred_joints, gt_joints, valid_mask_4d, num_valid, n_joints * 3)
        else:
            joint_loss = criterion(pred_joints, gt_joints)

        total_global_loss += joint_weight * joint_loss
        loss_dict['global_joint'] = joint_loss.item()
    else:
        loss_dict['global_joint'] = 0.0

    return total_global_loss, loss_dict


# ============================================================================
# Main Training Function
# ============================================================================

def train_progressive(folder_name, epoch_num, loader, model, writer, do_sample, sampler,
                     optimizer, scheduler, device, dataset_name, run_num,
                     use_smplx_loss=False, pose_loss_weight=1.0, vertex_loss_weight=1.0, joint_loss_weight=1.0,
                     level_1_weight=1.0, level_2_weight=1.0, level_3_weight=1.0,
                     level_loss_configs=None, kl_anneal_epochs=100, max_kl_weight=0.05,
                     vel_weight=1.0):
    """
    Progressive training with stop gradients between levels and global losses.
    Supports variable-length batches with audio and gesture type conditioning.
    """
    loader = tqdm(loader)

    criterion = nn.MSELoss()
    if kl_anneal_epochs > 0:
        latent_loss_weight = min(max_kl_weight, (epoch_num / kl_anneal_epochs) * max_kl_weight)
    else:
        latent_loss_weight = max_kl_weight

    # Track losses
    level_1_loss_sum = 0.0
    level_2_loss_sum = 0.0
    level_3_loss_sum = 0.0
    global_loss_sum = 0.0
    kl_loss_sum = 0.0
    kl_raw_sum = 0.0
    total_loss_sum = 0.0
    grad_norm_sum = 0.0
    kl_level_sums = [0.0, 0.0, 0.0]
    level_1_components = {}
    level_2_components = {}
    level_3_components = {}
    global_components = {}
    mse_n = 0

    # Get latent_dim for KL/dim logging
    actual_model = model.module if hasattr(model, 'module') else model
    latent_dim = actual_model.latent_dim

    for i, (data, label) in enumerate(loader):
        model.zero_grad()

        # Handle dict-based data loader
        if isinstance(data, dict):
            poses = data['poses'].to(device)

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
            poses = data.to(device)
            gesture_type = None
            lengths = None
            padding_mask = None
            audio = None
            gt_joints = None
            gt_vertices = None

        # ==================== Forward Pass ====================
        result = model(poses, padding_mask=padding_mask, gesture_type=gesture_type,
                      lengths=lengths, audio_features=audio,
                      compute_geometry=use_smplx_loss, return_intermediate=True)

        if use_smplx_loss:
            intermediate_outputs, latent_loss, raw_kl, pred_vertices, pred_joints, kl_per_level = result
        else:
            intermediate_outputs, latent_loss, raw_kl, kl_per_level = result
            pred_vertices, pred_joints = None, None

        # Compute GT joints/vertices from GT poses if not provided by dataloader
        if use_smplx_loss and gt_joints is None:
            smplx_layer = model.module.smplx_layer if hasattr(model, 'module') else model.smplx_layer
            if smplx_layer is not None:
                with torch.no_grad():
                    gt_vertices, gt_joints = smplx_layer(poses)

        level_1_output = intermediate_outputs[0]
        level_2_output = intermediate_outputs[1]
        level_3_output = intermediate_outputs[2]

        # ==================== Level 1 Loss ====================
        level_1_detached = level_1_output
        pred_verts_l1, pred_joints_l1 = None, None
        if use_smplx_loss:
            smplx_layer = model.module.smplx_layer if hasattr(model, 'module') else model.smplx_layer
            if smplx_layer is not None:
                pred_verts_l1, pred_joints_l1 = smplx_layer(level_1_detached)

        level_1_local_loss, level_1_dict = compute_level_local_loss(
            level_1_detached, poses, pred_verts_l1, pred_joints_l1,
            gt_vertices, gt_joints, 'face', criterion, device,
            padding_mask=padding_mask, vel_weight=vel_weight
        )

        # ==================== Level 2 Loss ====================
        level_2_detached = level_2_output
        pred_verts_l2, pred_joints_l2 = None, None
        if use_smplx_loss:
            smplx_layer = model.module.smplx_layer if hasattr(model, 'module') else model.smplx_layer
            if smplx_layer is not None:
                pred_verts_l2, pred_joints_l2 = smplx_layer(level_2_detached)

        level_2_local_loss, level_2_dict = compute_level_local_loss(
            level_2_detached, poses, pred_verts_l2, pred_joints_l2,
            gt_vertices, gt_joints, 'body', criterion, device,
            padding_mask=padding_mask, vel_weight=vel_weight
        )

        # ==================== Level 3 Loss ====================
        level_3_detached = level_3_output
        pred_verts_l3, pred_joints_l3 = None, None
        if use_smplx_loss:
            smplx_layer = model.module.smplx_layer if hasattr(model, 'module') else model.smplx_layer
            if smplx_layer is not None:
                pred_verts_l3, pred_joints_l3 = smplx_layer(level_3_detached)

        level_3_local_loss, level_3_dict = compute_level_local_loss(
            level_3_detached, poses, pred_verts_l3, pred_joints_l3,
            gt_vertices, gt_joints, 'hands', criterion, device,
            padding_mask=padding_mask, vel_weight=vel_weight
        )

        # ==================== Global Loss (BACKPROP THROUGH ALL) ====================
        global_loss, global_dict = compute_global_loss(
            level_3_output, poses, pred_vertices, pred_joints,
            gt_vertices, gt_joints, criterion, device,
            pose_weight=pose_loss_weight,
            mesh_weight=vertex_loss_weight,
            joint_weight=joint_loss_weight,
            padding_mask=padding_mask,
            vel_weight=vel_weight
        )

        # ==================== Combine All Losses ====================
        latent_loss = latent_loss.mean()

        total_loss = (
            level_1_weight * level_1_local_loss +
            level_2_weight * level_2_local_loss +
            level_3_weight * level_3_local_loss +
            global_loss +
            latent_loss_weight * latent_loss
        )

        # ==================== Backward Pass ====================
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # ==================== Track Metrics ====================
        batch_size = poses.shape[0]
        level_1_loss_sum += level_1_local_loss.item() * batch_size
        level_2_loss_sum += level_2_local_loss.item() * batch_size
        level_3_loss_sum += level_3_local_loss.item() * batch_size
        global_loss_sum += global_loss.item() * batch_size
        kl_loss_sum += latent_loss.item() * batch_size
        kl_raw_sum += raw_kl.item() * batch_size
        total_loss_sum += total_loss.item() * batch_size
        grad_norm_sum += grad_norm.item() * batch_size
        if kl_per_level is not None:
            for lvl_idx, kl_lvl in enumerate(kl_per_level):
                kl_level_sums[lvl_idx] += kl_lvl.item() * batch_size
        mse_n += batch_size

        for k, v in level_1_dict.items():
            level_1_components[k] = level_1_components.get(k, 0.0) + v * batch_size
        for k, v in level_2_dict.items():
            level_2_components[k] = level_2_components.get(k, 0.0) + v * batch_size
        for k, v in level_3_dict.items():
            level_3_components[k] = level_3_components.get(k, 0.0) + v * batch_size
        for k, v in global_dict.items():
            global_components[k] = global_components.get(k, 0.0) + v * batch_size

        # ==================== Progress Bar ====================
        lr = optimizer.param_groups[0]['lr']
        avg_l1 = level_1_loss_sum / mse_n
        avg_l2 = level_2_loss_sum / mse_n
        avg_l3 = level_3_loss_sum / mse_n
        avg_global = global_loss_sum / mse_n
        avg_kl_loss = kl_loss_sum / mse_n
        avg_kl_raw = kl_raw_sum / mse_n
        avg_total = total_loss_sum / mse_n
        avg_grad_norm = grad_norm_sum / mse_n
        avg_kl_per_dim = avg_kl_raw / latent_dim
        avg_global_vel = global_components.get('global_vel', 0.0) / max(mse_n, 1)

        desc = (
            f'epoch: {epoch_num + 1}; '
            f'L1: {avg_l1:.4f}; '
            f'L2: {avg_l2:.4f}; '
            f'L3: {avg_l3:.4f}; '
            f'G: {avg_global:.4f}; '
            f'Vel: {avg_global_vel:.4f}; '
            f'KL: {avg_kl_raw:.1f}({avg_kl_per_dim:.2f}/d); '
            f'KL[{kl_level_sums[0]/max(mse_n,1):.0f},{kl_level_sums[1]/max(mse_n,1):.0f},{kl_level_sums[2]/max(mse_n,1):.0f}]; '
            f'|g|: {avg_grad_norm:.2f}; '
            f'lr: {lr:.2e}'
        )
        loader.set_description(desc)

    # ==================== TensorBoard Logging ====================
    writer.add_scalar('Loss/total', total_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/level_1_total', level_1_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/level_2_total', level_2_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/level_3_total', level_3_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/global_total', global_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/kl_raw', kl_raw_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/kl_per_dim', kl_raw_sum / mse_n / latent_dim, epoch_num)
    writer.add_scalar('Loss/kl_loss', kl_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/kl_weighted', latent_loss_weight * kl_loss_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/kl_weight', latent_loss_weight, epoch_num)
    writer.add_scalar('Loss/kl_level_1', kl_level_sums[0] / mse_n, epoch_num)
    writer.add_scalar('Loss/kl_level_2', kl_level_sums[1] / mse_n, epoch_num)
    writer.add_scalar('Loss/kl_level_3', kl_level_sums[2] / mse_n, epoch_num)
    writer.add_scalar('Train/grad_norm', grad_norm_sum / mse_n, epoch_num)

    for k, v in level_1_components.items():
        writer.add_scalar(f'Loss/level_1_{k}', v / mse_n, epoch_num)
    for k, v in level_2_components.items():
        writer.add_scalar(f'Loss/level_2_{k}', v / mse_n, epoch_num)
    for k, v in level_3_components.items():
        writer.add_scalar(f'Loss/level_3_{k}', v / mse_n, epoch_num)
    for k, v in global_components.items():
        writer.add_scalar(f'Loss/{k}', v / mse_n, epoch_num)

    # Sample if needed
    if do_sample:
        sampler(folder_name, model, poses, dataset_name, run_num, epoch_num, poses.shape[0],
                padding_mask=padding_mask, gesture_type=gesture_type,
                lengths=lengths, audio_features=audio)

    # Build metrics dict for CSV logging
    metrics = {
        'total': total_loss_sum / mse_n,
        'level_1': level_1_loss_sum / mse_n,
        'level_2': level_2_loss_sum / mse_n,
        'level_3': level_3_loss_sum / mse_n,
        'global': global_loss_sum / mse_n,
        'kl_raw': kl_raw_sum / mse_n,
        'kl_per_dim': kl_raw_sum / mse_n / latent_dim,
        'kl_loss': kl_loss_sum / mse_n,
        'kl_weight': latent_loss_weight,
        'kl_level_1': kl_level_sums[0] / mse_n,
        'kl_level_2': kl_level_sums[1] / mse_n,
        'kl_level_3': kl_level_sums[2] / mse_n,
        'grad_norm': grad_norm_sum / mse_n,
        'lr': optimizer.param_groups[0]['lr'],
    }
    for k, v in level_1_components.items():
        metrics[f'level_1_{k}'] = v / mse_n
    for k, v in level_2_components.items():
        metrics[f'level_2_{k}'] = v / mse_n
    for k, v in level_3_components.items():
        metrics[f'level_3_{k}'] = v / mse_n
    for k, v in global_components.items():
        metrics[k] = v / mse_n

    return metrics
