import torch
from torch import nn
from tqdm import tqdm


# ============================================================================
# SMPLX body part definitions (165D pose)
# ============================================================================
BODY_PARTS = {
    'global_orient': (0, 3),
    'body': (3, 66),           # Main body pose (21 joints × 3)
    'jaw': (66, 69),
    'eyes': (69, 75),          # Left and right eyes
    'hands': (75, 165),        # Both hands (left: 75-120, right: 120-165)
    'face': (66, 75),          # Jaw + eyes combined
}


# ============================================================================
# MODULAR LOSS CONFIGURATION
# ============================================================================
# Define which losses to apply at each level
# Available loss types:
#   - 'axis_angle': MSE loss on axis-angle pose parameters (specify body parts)
#   - 'mesh': MSE loss on SMPLX mesh vertices
#   - 'joint': MSE loss on SMPLX joint positions
#
# Each loss is a dict with:
#   - 'type': Loss type ('axis_angle', 'mesh', 'joint')
#   - 'weight': Relative weight for this loss component
#   - 'parts': (for axis_angle only) List of body part names from BODY_PARTS
#   - 'name': Descriptive name for logging
# ============================================================================

LEVEL_1_LOSSES = [
    {
        'type': 'axis_angle',
        'parts': ['face'],  # jaw + eyes
        'weight': 1.0,
        'name': 'face'
    }
]

LEVEL_2_LOSSES = [
    {
        'type': 'axis_angle',
        'parts': ['body'],  # Main body pose (21 joints)
        'weight': 1.0,
        'name': 'body'
    }
]

LEVEL_3_LOSSES = [
    {
        'type': 'axis_angle',
        'parts': ['hands'],  # Both hands
        'weight': 1.0,
        'name': 'hands'
    },
    {
        'type': 'mesh',
        'weight': 1.0,
        'name': 'mesh'
    },
    {
        'type': 'joint',
        'weight': 1.0,
        'name': 'joint'
    }
]

# Aggregate all level loss configurations
LEVEL_LOSS_CONFIGS = [LEVEL_1_LOSSES, LEVEL_2_LOSSES, LEVEL_3_LOSSES]


# ============================================================================
# LOSS COMPUTATION HELPER
# ============================================================================
def compute_level_loss(loss_config, level_output, gt_poses, pred_vertices, pred_joints,
                       gt_vertices, gt_joints, criterion, device):
    """
    Compute loss for a given level based on its loss configuration.

    Args:
        loss_config: List of loss dicts for this level (from LEVEL_LOSS_CONFIGS)
        level_output: Reconstructed poses at this level (batch, seq_len, 165)
        gt_poses: Ground truth poses (batch, seq_len, 165)
        pred_vertices: Predicted SMPLX vertices (batch, seq_len, 10475, 3) or None
        pred_joints: Predicted SMPLX joints (batch, seq_len, 127, 3) or None
        gt_vertices: Ground truth vertices or None
        gt_joints: Ground truth joints or None
        criterion: Loss function (e.g., nn.MSELoss())
        device: Device

    Returns:
        total_loss: Combined loss for this level
        loss_components: Dict of individual loss values for logging
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_components = {}

    for loss_spec in loss_config:
        loss_type = loss_spec['type']
        weight = loss_spec['weight']
        name = loss_spec['name']

        if loss_type == 'axis_angle':
            # Compute MSE on specific body parts
            parts = loss_spec['parts']

            # Collect indices for all specified parts
            indices = []
            for part_name in parts:
                if part_name not in BODY_PARTS:
                    raise ValueError(f"Unknown body part: {part_name}")
                start, end = BODY_PARTS[part_name]
                indices.extend(range(start, end))

            # Convert to tensor for advanced indexing
            indices = torch.tensor(indices, device=device)

            # Compute loss on selected dimensions
            loss = criterion(
                level_output[:, :, indices],
                gt_poses[:, :, indices]
            )

            total_loss += weight * loss
            loss_components[name] = loss.item()

        elif loss_type == 'mesh':
            # Compute MSE on mesh vertices
            if pred_vertices is None or gt_vertices is None:
                # Skip if geometry not available
                loss = torch.tensor(0.0, device=device)
            else:
                loss = criterion(pred_vertices, gt_vertices)
                total_loss += weight * loss

            loss_components[name] = loss.item()

        elif loss_type == 'joint':
            # Compute MSE on joint positions
            if pred_joints is None or gt_joints is None:
                # Skip if geometry not available
                loss = torch.tensor(0.0, device=device)
            else:
                loss = criterion(pred_joints, gt_joints)
                total_loss += weight * loss

            loss_components[name] = loss.item()

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    return total_loss, loss_components


def train_progressive(folder_name, epoch_num, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, run_num,
                     use_smplx_loss=False, pose_loss_weight=1.0, vertex_loss_weight=1.0, joint_loss_weight=1.0,
                     level_1_weight=1.0, level_2_weight=1.0, level_3_weight=1.0,
                     level_loss_configs=None):
    """
    Progressive training loop with modular level-specific losses.

    Loss configuration is defined by LEVEL_LOSS_CONFIGS at the top of this file.
    You can customize which losses apply to each level by modifying:
    - LEVEL_1_LOSSES
    - LEVEL_2_LOSSES
    - LEVEL_3_LOSSES

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
        use_smplx_loss: Whether to use SMPLX geometry losses
        pose_loss_weight: DEPRECATED - use loss configs instead
        vertex_loss_weight: DEPRECATED - use loss configs instead
        joint_loss_weight: DEPRECATED - use loss configs instead
        level_1_weight: Overall weight multiplier for level 1 loss (default: 1.0)
        level_2_weight: Overall weight multiplier for level 2 loss (default: 1.0)
        level_3_weight: Overall weight multiplier for level 3 loss (default: 1.0)
        level_loss_configs: Custom loss configs (defaults to LEVEL_LOSS_CONFIGS)
    """
    # Use default configs if none provided
    if level_loss_configs is None:
        level_loss_configs = LEVEL_LOSS_CONFIGS

    # Note: pose_loss_weight, vertex_loss_weight, joint_loss_weight are deprecated.
    # Adjust weights in the LEVEL_LOSS_CONFIGS at the top of this file instead.
    loader = tqdm(loader)

    criterion = nn.MSELoss()
    latent_loss_weight = 0.25

    # Initialize loss tracking for all levels and their components
    n_levels = len(level_loss_configs)
    level_loss_sums = [0.0 for _ in range(n_levels)]
    # Track individual loss components for detailed logging
    level_component_sums = [{} for _ in range(n_levels)]
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
        if len(intermediate_outputs) < n_levels:
            raise ValueError(f"Expected at least {n_levels} levels, got {len(intermediate_outputs)}")

        # ==================== Compute Losses for Each Level ====================
        level_weights = [level_1_weight, level_2_weight, level_3_weight]
        level_losses = []
        level_components_list = []

        for level_idx in range(n_levels):
            level_output = intermediate_outputs[level_idx]
            loss_config = level_loss_configs[level_idx]

            # Compute loss for this level using the modular loss configuration
            level_loss, loss_components = compute_level_loss(
                loss_config=loss_config,
                level_output=level_output,
                gt_poses=poses,
                pred_vertices=pred_vertices,
                pred_joints=pred_joints,
                gt_vertices=gt_vertices,
                gt_joints=gt_joints,
                criterion=criterion,
                device=device
            )

            level_losses.append(level_loss)
            level_components_list.append(loss_components)

        # ==================== Combine All Losses ====================
        latent_loss = latent_loss.mean()

        # Weighted sum of all level losses
        total_loss = sum(
            level_weights[i] * level_losses[i]
            for i in range(n_levels)
        ) + latent_loss_weight * latent_loss

        # Backward pass
        total_loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        # Track metrics
        batch_size = poses.shape[0]

        # Track level losses
        for level_idx in range(n_levels):
            level_loss_sums[level_idx] += level_losses[level_idx].item() * batch_size

            # Track individual components
            for component_name, component_value in level_components_list[level_idx].items():
                if component_name not in level_component_sums[level_idx]:
                    level_component_sums[level_idx][component_name] = 0.0
                level_component_sums[level_idx][component_name] += component_value * batch_size

        total_loss_sum += total_loss.item() * batch_size
        mse_n += batch_size

        lr = optimizer.param_groups[0]['lr']

        # Compute running averages for display
        avg_level_losses = [level_loss_sums[i] / mse_n for i in range(n_levels)]
        avg_total = total_loss_sum / mse_n

        # Build progress bar description with dynamic loss names
        desc = f'epoch: {epoch_num + 1}; '

        for level_idx in range(n_levels):
            # Get component names for this level
            component_names = [loss_spec['name'] for loss_spec in level_loss_configs[level_idx]]
            components_str = '+'.join(component_names)
            desc += f'L{level_idx + 1}({components_str}): {avg_level_losses[level_idx]:.5f}; '

        desc += f'total: {avg_total:.5f}; lr: {lr:.5f}'
        loader.set_description(desc)

    # Log to TensorBoard
    writer.add_scalar('Loss/total', total_loss_sum / mse_n, epoch_num)

    # Log level-specific losses
    for level_idx in range(n_levels):
        avg_level_loss = level_loss_sums[level_idx] / mse_n
        writer.add_scalar(f'Loss/level_{level_idx + 1}_total', avg_level_loss, epoch_num)

        # Log individual components
        for component_name, component_sum in level_component_sums[level_idx].items():
            avg_component = component_sum / mse_n
            writer.add_scalar(f'Loss/level_{level_idx + 1}_{component_name}', avg_component, epoch_num)

    # Sample if needed (use final output)
    if do_sample:
        sampler(folder_name, model, poses, dataset_name, run_num, epoch_num, poses.shape[0])

    # Return average total loss for early stopping
    return total_loss_sum / mse_n
