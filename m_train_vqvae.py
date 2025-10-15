import torch
from torch import nn
from tqdm import tqdm


def train(folder_name, epoch_num, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, run_num,
          use_smplx_loss=False, pose_loss_weight=1.0, vertex_loss_weight=1.0, joint_loss_weight=1.0):
    """
    Training loop with optional SMPLX-based multi-loss supervision.

    Args:
        folder_name: Model folder name
        epoch_num: Current epoch number
        loader: DataLoader
        model: VQ-VAE model (optionally with SMPLX layer)
        writer: TensorBoard writer
        do_sample: Whether to sample this epoch
        sampler: Sampling function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (cuda/cpu)
        dataset_name: Dataset name
        run_num: Run number
        use_smplx_loss: Whether to use SMPLX geometry losses
        pose_loss_weight: Weight for pose reconstruction loss
        vertex_loss_weight: Weight for vertex reconstruction loss
        joint_loss_weight: Weight for joint reconstruction loss
    """
    loader = tqdm(loader)

    criterion = nn.MSELoss()
    latent_loss_weight = 0.25

    # Track losses
    pose_mse_sum = 0
    vertex_mse_sum = 0
    joint_mse_sum = 0
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

        # Forward pass through model
        if use_smplx_loss:
            # Check if model has SMPLX capability
            model_has_smplx = False
            if hasattr(model, 'module'):  # DataParallel wrapper
                model_has_smplx = hasattr(model.module, 'use_smplx') and model.module.use_smplx
            elif hasattr(model, 'use_smplx'):
                model_has_smplx = model.use_smplx

            if model_has_smplx:
                result = model(poses, compute_geometry=True)
                reconstructed_poses, latent_loss, pred_vertices, pred_joints = result
            else:
                reconstructed_poses, latent_loss = model(poses)
                pred_vertices, pred_joints = None, None
        else:
            result = model(poses)
            if len(result) == 4:  # Model returned geometry even though not requested
                reconstructed_poses, latent_loss, pred_vertices, pred_joints = result
                pred_vertices, pred_joints = None, None  # Ignore geometry
            else:
                reconstructed_poses, latent_loss = result
                pred_vertices, pred_joints = None, None

        # Compute pose reconstruction loss
        pose_recon_loss = criterion(reconstructed_poses, poses)

        # Compute geometry losses if available
        vertex_recon_loss = torch.tensor(0.0, device=device)
        joint_recon_loss = torch.tensor(0.0, device=device)

        if use_smplx_loss and pred_vertices is not None and gt_vertices is not None:
            vertex_recon_loss = criterion(pred_vertices, gt_vertices)

        if use_smplx_loss and pred_joints is not None and gt_joints is not None:
            joint_recon_loss = criterion(pred_joints, gt_joints)

        # Combine losses
        latent_loss = latent_loss.mean()
        total_loss = (pose_loss_weight * pose_recon_loss +
                     vertex_loss_weight * vertex_recon_loss +
                     joint_loss_weight * joint_recon_loss +
                     latent_loss_weight * latent_loss)

        # Backward pass
        total_loss.backward()

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

        # Update progress bar
        desc = (
            f'epoch: {epoch_num + 1}; '
            f'pose_mse: {pose_recon_loss.item():.5f}; '
        )
        if use_smplx_loss:
            desc += (
                f'vert_mse: {vertex_recon_loss.item():.5f}; '
                f'joint_mse: {joint_recon_loss.item():.5f}; '
            )
        desc += (
            f'latent: {latent_loss.item():.3f}; '
            f'avg_pose: {pose_mse_sum / mse_n:.5f}; '
            f'lr: {lr:.5f}'
        )
        loader.set_description(desc)

    # Log to TensorBoard
    writer.add_scalar('Loss/train', pose_mse_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/pose_mse', pose_mse_sum / mse_n, epoch_num)
    writer.add_scalar('Loss/total', total_loss_sum / mse_n, epoch_num)

    if use_smplx_loss:
        writer.add_scalar('Loss/vertex_mse', vertex_mse_sum / mse_n, epoch_num)
        writer.add_scalar('Loss/joint_mse', joint_mse_sum / mse_n, epoch_num)

    # Sample if needed
    if do_sample:
        sampler(folder_name, model, poses, dataset_name, run_num, epoch_num, poses.shape[0])
