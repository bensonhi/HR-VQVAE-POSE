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

# Level index → body part name. Levels beyond this list use whole-body loss.
LEVEL_TO_PART = ['face', 'body', 'hands']


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


def _continuation_velocity_loss(recon, gt, anchor_frames, anchor_pool, n_frames=30):
    """Velocity loss near the anchor→generated boundary with decaying weight.

    Frame 0: boundary velocity (recon[0] - anchor[-1]) vs (gt[0] - anchor_pool[-1])
    Frame i>0: intra-sequence velocity (recon[i] - recon[i-1]) vs (gt[i] - gt[i-1])
    Weight decays linearly: w_i = 1 - i/n_frames (strongest at boundary).

    This gives extra velocity penalty near the start of each generated chunk,
    on top of the uniform velocity loss that covers all frames equally.
    """
    T = recon.shape[1]
    n = min(n_frames, T)
    total_loss = 0.0
    weight_sum = 0.0

    for i in range(n):
        w = 1.0 - i / n_frames  # linear decay
        if i == 0:
            pred_vel = recon[:, 0] - anchor_frames[:, -1]
            gt_vel = gt[:, 0] - anchor_pool[:, -1]
        else:
            pred_vel = recon[:, i] - recon[:, i - 1]
            gt_vel = gt[:, i] - gt[:, i - 1]
        total_loss = total_loss + w * F.mse_loss(pred_vel, gt_vel)
        weight_sum += w

    return total_loss / weight_sum


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

    # 4. Joint position loss on specific body part
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

    # 4. Full joint reconstruction loss
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
                     level_weights=None,
                     level_loss_configs=None, kl_anneal_epochs=100, max_kl_weight=0.05,
                     vel_weight=1.0, cont_vel_weight=50.0,
                     anchor_recon_weight=0.1):
    """
    Progressive training with stop gradients between levels and global losses.
    Supports variable-length batches with audio and gesture type conditioning.

    level_weights: list of per-level loss weights. Length should match n_level.
                   Defaults to [1.0] * n_level. Levels 0-2 use body-part losses
                   (face/body/hands); levels 3+ use whole-body losses.
    """
    loader = tqdm(loader)

    criterion = nn.MSELoss()
    if kl_anneal_epochs > 0:
        latent_loss_weight = min(max_kl_weight, (epoch_num / kl_anneal_epochs) * max_kl_weight)
    else:
        latent_loss_weight = max_kl_weight

    # Get model config
    actual_model = model.module if hasattr(model, 'module') else model
    n_levels = actual_model.n_level
    latent_dim = actual_model.latent_dim
    temporal_downsample = getattr(actual_model, 'temporal_downsample', 0)
    smplx_layer = actual_model.smplx_layer if use_smplx_loss else None

    import math
    if temporal_downsample > 1:
        kl_total_dims = math.ceil(150 / temporal_downsample) * latent_dim
    else:
        kl_total_dims = latent_dim

    # Default level weights
    if level_weights is None:
        level_weights_used = [1.0] * n_levels
    else:
        level_weights_used = list(level_weights)
        while len(level_weights_used) < n_levels:
            level_weights_used.append(1.0)

    # Track losses
    level_loss_sums = [0.0] * n_levels
    level_components_list = [{} for _ in range(n_levels)]
    global_loss_sum = 0.0
    kl_loss_sum = 0.0
    kl_raw_sum = 0.0
    total_loss_sum = 0.0
    grad_norm_sum = 0.0
    kl_level_sums = [0.0] * n_levels
    global_components = {}
    mse_n = 0

    for i, (data, label) in enumerate(loader):
        model.zero_grad()

        # Handle dict-based data loader
        if isinstance(data, dict):
            poses = data['poses'].to(device)

            gesture_type = data.get('gesture_type', None)
            if gesture_type is not None:
                gesture_type = gesture_type.to(device)

            speaker_id = data.get('speaker_id', None)
            if speaker_id is not None:
                speaker_id = speaker_id.to(device)

            lengths = data.get('lengths', None)
            if lengths is not None:
                lengths = lengths.to(device)

            padding_mask = data.get('padding_mask', None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            audio = data.get('audio', None)
            if audio is not None:
                audio = audio.to(device)

            anchor_pool = data.get('anchor_pool', None)
            if anchor_pool is not None:
                anchor_pool = anchor_pool.to(device)

            anchor_audio = data.get('anchor_audio', None)
            if anchor_audio is not None:
                anchor_audio = anchor_audio.to(device)

            text_features = data.get('text', None)
            if text_features is not None:
                text_features = text_features.to(device)

            gt_joints = data.get('gt_joints', None)
            gt_vertices = data.get('gt_vertices', None)
            if gt_joints is not None:
                gt_joints = gt_joints.to(device)
            if gt_vertices is not None:
                gt_vertices = gt_vertices.to(device)
        else:
            poses = data.to(device)
            gesture_type = None
            speaker_id = None
            lengths = None
            padding_mask = None
            audio = None
            anchor_pool = None
            anchor_audio = None
            text_features = None
            gt_joints = None
            gt_vertices = None

        # ==================== Forward Pass ====================
        result = model(poses, padding_mask=padding_mask, gesture_type=gesture_type,
                      lengths=lengths, audio_features=audio,
                      compute_geometry=use_smplx_loss, return_intermediate=True,
                      anchor_pool=anchor_pool, speaker_id=speaker_id,
                      anchor_audio=anchor_audio, text_features=text_features)

        if use_smplx_loss:
            intermediate_outputs, latent_loss, raw_kl, pred_vertices, pred_joints, kl_per_level = result
        else:
            intermediate_outputs, latent_loss, raw_kl, kl_per_level = result
            pred_vertices, pred_joints = None, None

        # Compute GT joints/vertices from GT poses if not provided by dataloader
        if use_smplx_loss and gt_joints is None:
            _smplx = actual_model.smplx_layer
            if _smplx is not None:
                with torch.no_grad():
                    gt_vertices, gt_joints = _smplx(poses)

        # ==================== Per-Level Losses ====================
        # Levels 0-2: body-part-specific (face, body, hands)
        # Levels 3+: whole-body loss
        level_local_losses = []

        for lvl_idx in range(n_levels):
            lvl_output = intermediate_outputs[lvl_idx]
            is_final = (lvl_idx == n_levels - 1)

            if lvl_idx < len(LEVEL_TO_PART):
                part_name = LEVEL_TO_PART[lvl_idx]
                pred_verts_lvl, pred_joints_lvl = None, None
                if use_smplx_loss and smplx_layer is not None:
                    pred_verts_lvl, pred_joints_lvl = smplx_layer(lvl_output)
                lvl_loss, lvl_dict = compute_level_local_loss(
                    lvl_output, poses, pred_verts_lvl, pred_joints_lvl,
                    gt_vertices, gt_joints, part_name, criterion, device,
                    padding_mask=padding_mask, vel_weight=vel_weight
                )
            else:
                # Whole-body loss for levels beyond face/body/hands
                # For the final level, reuse pre-computed vertices/joints from forward pass
                if is_final:
                    pred_verts_lvl, pred_joints_lvl = pred_vertices, pred_joints
                else:
                    pred_verts_lvl, pred_joints_lvl = None, None
                    if use_smplx_loss and smplx_layer is not None:
                        pred_verts_lvl, pred_joints_lvl = smplx_layer(lvl_output)
                lvl_loss, lvl_dict = compute_global_loss(
                    lvl_output, poses, pred_verts_lvl, pred_joints_lvl,
                    gt_vertices, gt_joints, criterion, device,
                    pose_weight=pose_loss_weight,
                    mesh_weight=vertex_loss_weight,
                    joint_weight=joint_loss_weight,
                    padding_mask=padding_mask,
                    vel_weight=vel_weight
                )

            level_local_losses.append(lvl_loss)
            # Accumulate components
            for k, v in lvl_dict.items():
                level_components_list[lvl_idx][k] = level_components_list[lvl_idx].get(k, 0.0) + v * poses.shape[0]

        # ==================== Global Loss (BACKPROP THROUGH ALL) ====================
        global_loss, global_dict = compute_global_loss(
            intermediate_outputs[-1], poses, pred_vertices, pred_joints,
            gt_vertices, gt_joints, criterion, device,
            pose_weight=pose_loss_weight,
            mesh_weight=vertex_loss_weight,
            joint_weight=joint_loss_weight,
            padding_mask=padding_mask,
            vel_weight=vel_weight
        )

        # ==================== Continuation Velocity Loss ====================
        used_anchor_frames = actual_model.get_last_anchor_frames()
        cont_vel_loss = torch.tensor(0.0, device=device)
        if used_anchor_frames is not None and anchor_pool is not None and cont_vel_weight > 0:
            cont_vel_loss = _continuation_velocity_loss(
                intermediate_outputs[-1], poses, used_anchor_frames, anchor_pool)
        global_dict['cont_vel'] = cont_vel_loss.item()

        # ==================== Anchor Reconstruction Loss ====================
        anchor_recon = actual_model.get_last_anchor_recon()
        anchor_recon_loss = torch.tensor(0.0, device=device)
        if anchor_recon is not None and anchor_pool is not None and anchor_recon_weight > 0:
            K_recon = anchor_recon.shape[1]
            anchor_gt = anchor_pool[:, -K_recon:, :]
            anchor_recon_loss = F.mse_loss(anchor_recon, anchor_gt)
        global_dict['anchor_recon'] = anchor_recon_loss.item()

        # ==================== Combine All Losses ====================
        latent_loss = latent_loss.mean()
        raw_kl = raw_kl.mean()
        if kl_per_level is not None:
            kl_per_level = [k.mean() for k in kl_per_level]

        weighted_level_loss = sum(level_weights_used[i] * level_local_losses[i] for i in range(n_levels))
        total_loss = (
            weighted_level_loss +
            global_loss +
            cont_vel_weight * cont_vel_loss +
            anchor_recon_weight * anchor_recon_loss +
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
        for lvl_idx in range(n_levels):
            level_loss_sums[lvl_idx] += level_local_losses[lvl_idx].item() * batch_size
        global_loss_sum += global_loss.item() * batch_size
        kl_loss_sum += latent_loss.item() * batch_size
        kl_raw_sum += raw_kl.item() * batch_size
        total_loss_sum += total_loss.item() * batch_size
        grad_norm_sum += grad_norm.item() * batch_size
        if kl_per_level is not None:
            for lvl_idx, kl_lvl in enumerate(kl_per_level):
                if lvl_idx < len(kl_level_sums):
                    kl_level_sums[lvl_idx] += kl_lvl.item() * batch_size
        mse_n += batch_size

        for k, v in global_dict.items():
            global_components[k] = global_components.get(k, 0.0) + v * batch_size

        # ==================== Progress Bar ====================
        lr = optimizer.param_groups[0]['lr']
        avg_levels = [level_loss_sums[i] / mse_n for i in range(n_levels)]
        avg_global = global_loss_sum / mse_n
        avg_kl_raw = kl_raw_sum / mse_n
        avg_total = total_loss_sum / mse_n
        avg_grad_norm = grad_norm_sum / mse_n
        avg_kl_per_dim = avg_kl_raw / kl_total_dims
        avg_global_vel = global_components.get('global_vel', 0.0) / max(mse_n, 1)
        avg_cont_vel = global_components.get('cont_vel', 0.0) / max(mse_n, 1)

        level_str = '; '.join(f'L{i+1}: {avg_levels[i]:.4f}' for i in range(n_levels))
        kl_levels_str = ','.join(f'{kl_level_sums[i]/max(mse_n,1):.0f}' for i in range(n_levels))
        desc = (
            f'epoch: {epoch_num + 1}; '
            f'{level_str}; '
            f'G: {avg_global:.4f}; '
            f'Vel: {avg_global_vel:.4f}; '
            f'CVel: {avg_cont_vel:.4f}; '
            f'KL: {avg_kl_raw:.1f}({avg_kl_per_dim:.2f}/d); '
            f'KL[{kl_levels_str}]; '
            f'|g|: {avg_grad_norm:.2f}; '
            f'lr: {lr:.2e}'
        )
        loader.set_description(desc)

    # ==================== TensorBoard Logging (rank 0 only) ====================
    if writer is not None:
        writer.add_scalar('Loss/total', total_loss_sum / mse_n, epoch_num)
        for i in range(n_levels):
            writer.add_scalar(f'Loss/level_{i+1}_total', level_loss_sums[i] / mse_n, epoch_num)
        writer.add_scalar('Loss/global_total', global_loss_sum / mse_n, epoch_num)
        writer.add_scalar('Loss/kl_raw', kl_raw_sum / mse_n, epoch_num)
        writer.add_scalar('Loss/kl_per_dim', kl_raw_sum / mse_n / kl_total_dims, epoch_num)
        writer.add_scalar('Loss/kl_loss', kl_loss_sum / mse_n, epoch_num)
        writer.add_scalar('Loss/kl_weighted', latent_loss_weight * kl_loss_sum / mse_n, epoch_num)
        writer.add_scalar('Loss/kl_weight', latent_loss_weight, epoch_num)
        for i in range(n_levels):
            writer.add_scalar(f'Loss/kl_level_{i+1}', kl_level_sums[i] / mse_n, epoch_num)
        writer.add_scalar('Train/grad_norm', grad_norm_sum / mse_n, epoch_num)

        for i in range(n_levels):
            for k, v in level_components_list[i].items():
                writer.add_scalar(f'Loss/level_{i+1}_{k}', v / mse_n, epoch_num)
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
        'global': global_loss_sum / mse_n,
        'kl_raw': kl_raw_sum / mse_n,
        'kl_per_dim': kl_raw_sum / mse_n / kl_total_dims,
        'kl_loss': kl_loss_sum / mse_n,
        'kl_weight': latent_loss_weight,
        'grad_norm': grad_norm_sum / mse_n,
        'lr': optimizer.param_groups[0]['lr'],
    }
    for i in range(n_levels):
        metrics[f'level_{i+1}'] = level_loss_sums[i] / mse_n
        metrics[f'kl_level_{i+1}'] = kl_level_sums[i] / mse_n
        for k, v in level_components_list[i].items():
            metrics[f'level_{i+1}_{k}'] = v / mse_n
    for k, v in global_components.items():
        metrics[k] = v / mse_n

    return metrics
