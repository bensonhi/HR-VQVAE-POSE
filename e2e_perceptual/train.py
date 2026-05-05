"""
E2E perceptual training: flow matching prior with FGD feature matching + beat alignment.

Uses BEAT2 dataloader directly (online) with frozen VAE encoder for z targets.
Supports anchor conditioning (matching autoregressive generation at test time).

Combined loss:
  L = flow_matching_loss + lambda_fgd * fgd_feature_loss + lambda_beat * beat_alignment_loss

The FGD and beat losses use multi-step differentiable ODE sampling:
  Run N Euler steps (with gradients) from noise to clean z, then decode through frozen
  VAE decoder to get motion, and compute perceptual losses on the decoded motion.

Usage:
    conda run -n hr-vqvae-poses python e2e_perceptual/train.py --checkpoint-dir e2e_perceptual/checkpoints_e2e

    # Resume from prior checkpoint (warm start)
    conda run -n hr-vqvae-poses python e2e_perceptual/train.py --checkpoint-dir e2e_perceptual/checkpoints_e2e --resume prior_net/checkpoints_diff_online/best.pt
"""
import argparse
import copy
import csv
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m_beat_dataset import BEAT2PoseDataset, variable_length_collate_fn
from m_util import conf_parser, create_model_object, get_model_type, load_checkpoint
from prior_net.diffusion_model import TemporalFlowMatchingPrior
from emage_evaltools.mertic import Arg
from emage_evaltools.motion_encoder import VAESKConv


# =========================================================================
# Differentiable axis-angle to rot6d (Rodrigues)
# =========================================================================

def axis_angle_to_rot6d(axis_angle):
    """Convert axis-angle (..., 3) to rotation 6D (..., 6). Fully differentiable."""
    batch_shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)

    angle = torch.norm(aa, dim=1, keepdim=True)
    axis = aa / angle.clamp(min=1e-8)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    t = 1 - cos_a

    x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]

    R = torch.cat([
        t*x*x + cos_a,   t*x*y - sin_a*z, t*x*z + sin_a*y,
        t*x*y + sin_a*z, t*y*y + cos_a,   t*y*z - sin_a*x,
    ], dim=1)  # (N, 6) — first two rows

    return R.reshape(*batch_shape, 6)


# =========================================================================
# FGD feature extractor (frozen)
# =========================================================================

def load_fgd_feature_extractor(device, download_path='./emage_evaltools/'):
    """Load frozen VAESKConv for FGD feature extraction."""
    args = Arg()
    model = VAESKConv(args, model_save_path=download_path)
    ckpt_path = os.path.join(download_path, 'AESKConv_240_100.bin')
    old_stat = torch.load(ckpt_path, map_location=device)['model_state']
    new_stat = {k.replace('module.', ''): v for k, v in old_stat.items()}
    model.load_state_dict(new_stat)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def fgd_feature_loss(motion, fgd_model, device):
    """Extract FGD features from motion (B, T, 165) -> (B, 240).

    Converts axis-angle to rot6d, truncates T to multiple of 32.
    """
    B, T, _ = motion.shape
    T_trunc = (T // 32) * 32
    if T_trunc < 32:
        return None
    motion = motion[:, :T_trunc]

    r6d = axis_angle_to_rot6d(motion.reshape(B, T_trunc, 55, 3))
    r6d = r6d.reshape(B, T_trunc, 330)

    features = fgd_model.map2latent(r6d.to(device))
    return features


def compute_fgd_moment_matching_loss(pred_motion, gt_motion, fgd_model, device, lambda_cov=1.0):
    """Batch-level moment matching in FGD feature space — directly approximates FGD.

    FGD = ||mu_pred - mu_gt||^2 + Tr(Sigma_pred + Sigma_gt - 2*sqrtm(Sigma_pred @ Sigma_gt))
    We approximate this with a differentiable surrogate:
      L = ||mu_pred - mu_gt||^2 + lambda_cov * ||Sigma_pred - Sigma_gt||_F^2

    Features are pooled across batch AND time (flattened to (N, 240)) to match
    how FGD is computed at evaluation time.
    """
    pred_feat = fgd_feature_loss(pred_motion, fgd_model, device)
    if pred_feat is None:
        return torch.tensor(0.0, device=device)
    with torch.no_grad():
        gt_feat = fgd_feature_loss(gt_motion, fgd_model, device)
    if gt_feat is None:
        return torch.tensor(0.0, device=device)

    # Flatten (B, T_down, 240) -> (N, 240)
    pred_flat = pred_feat.reshape(-1, pred_feat.shape[-1])
    gt_flat = gt_feat.reshape(-1, gt_feat.shape[-1])

    # Mean matching: ||mu_pred - mu_gt||^2
    mu_pred = pred_flat.mean(dim=0)
    with torch.no_grad():
        mu_gt = gt_flat.mean(dim=0)
    mean_loss = ((mu_pred - mu_gt) ** 2).sum()

    # Covariance matching: ||Sigma_pred - Sigma_gt||_F^2
    pred_centered = pred_flat - mu_pred.detach()  # detach mu to avoid double-counting
    cov_pred = (pred_centered.T @ pred_centered) / max(pred_flat.shape[0] - 1, 1)

    with torch.no_grad():
        gt_centered = gt_flat - mu_gt
        cov_gt = (gt_centered.T @ gt_centered) / max(gt_flat.shape[0] - 1, 1)

    cov_loss = ((cov_pred - cov_gt) ** 2).sum()

    return mean_loss + lambda_cov * cov_loss


# =========================================================================
# Beat alignment loss
# =========================================================================

def beat_alignment_loss(motion, audio_features):
    """Negative correlation between motion velocity and audio change.

    motion: (B, T, 165)
    audio_features: (B, T, 768) — WavLM features
    """
    B, T, _ = motion.shape
    if T < 3:
        return torch.tensor(0.0, device=motion.device)

    motion_vel = torch.norm(motion[:, 1:] - motion[:, :-1], dim=-1)  # (B, T-1)
    audio_vel = torch.norm(audio_features[:, 1:] - audio_features[:, :-1], dim=-1)  # (B, T-1)

    motion_vel = motion_vel - motion_vel.mean(dim=1, keepdim=True)
    motion_std = motion_vel.std(dim=1, keepdim=True).clamp(min=1e-6)
    motion_vel = motion_vel / motion_std

    audio_vel = audio_vel - audio_vel.mean(dim=1, keepdim=True)
    audio_std = audio_vel.std(dim=1, keepdim=True).clamp(min=1e-6)
    audio_vel = audio_vel / audio_std

    corr = (motion_vel * audio_vel).mean(dim=1)  # (B,)
    return -corr.mean()


# =========================================================================
# VAE encoding helper
# =========================================================================

@torch.no_grad()
def encode_batch(vae_model, batch, device):
    """Encode a batch through the frozen VAE encoder to get z targets."""
    poses = batch['poses'].to(device)
    audio = batch['audio'].to(device)
    gesture_type = batch['gesture_type'].to(device)
    speaker_id = batch['speaker_id'].to(device)
    padding_mask = batch['padding_mask'].to(device)

    z, mu, logvar, kl_loss, raw_kl, kl_per_level = vae_model.encode(
        poses, padding_mask,
        gesture_type=gesture_type,
        audio_features=audio,
        speaker_id=speaker_id,
    )
    z_padding_mask = vae_model._z_padding_mask
    return mu, z_padding_mask


# =========================================================================
# EMA
# =========================================================================

def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# =========================================================================
# Differentiable ODE sampling (for E2E perceptual loss)
# =========================================================================

def differentiable_ode_sample(model, audio_ctx, z_seq_len, num_steps=8,
                               z_padding_mask=None):
    """Run N Euler ODE steps WITH gradients to get clean z predictions.

    Unlike the predicted-x0 trick (single noisy estimate), this runs the full
    ODE integration so the output z is clean — matching what we get at inference.
    Gradients flow through all steps back to model parameters.

    Args:
        model: TemporalFlowMatchingPrior (in training mode)
        audio_ctx: (B, T', d_model) pre-encoded audio context
        z_seq_len: int, number of z tokens
        num_steps: number of Euler steps (5-10 is good for training)
        z_padding_mask: optional (B, T') bool mask

    Returns:
        z_clean: (B, T', latent_dim) denormalized z (same space as VAE mu)
    """
    B = audio_ctx.shape[0]
    device = audio_ctx.device

    # Pad/trim audio_ctx to match z_seq_len
    if audio_ctx.shape[1] != z_seq_len:
        if audio_ctx.shape[1] < z_seq_len:
            pad = audio_ctx[:, -1:, :].expand(B, z_seq_len - audio_ctx.shape[1], -1)
            audio_ctx = torch.cat([audio_ctx, pad], dim=1)
        else:
            audio_ctx = audio_ctx[:, :z_seq_len, :]

    # Start from noise (detached — we only want grads through the velocity network)
    z = torch.randn(B, z_seq_len, model.latent_dim, device=device)
    if z_padding_mask is not None:
        z = z * (~z_padding_mask).unsqueeze(-1).float()

    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)
        v = model.predict_velocity(z, t, audio_ctx, z_padding_mask=z_padding_mask)
        z = z + v * dt

    # Denormalize
    z_clean = z * model.z_std + model.z_mean
    return z_clean


# =========================================================================
# Training
# =========================================================================

def train_epoch(model, vae_model, fgd_model, loader, optimizer, device,
                grad_clip=1.0, cond_drop_prob=0.1, anchor_drop_prob=0.3,
                lambda_fgd=1.0, lambda_beat=0.1, lambda_cov=1.0,
                e2e_t_threshold=0.5, lambda_recon=0.0, use_text=False):
    """Train one epoch with combined flow matching + perceptual losses.

    Uses BEAT2 dataloader with online VAE encoding and anchor conditioning.
    Perceptual losses use multi-step differentiable ODE sampling for clean z.
    """
    model.train()
    vae_model.eval()

    total_loss = 0
    total_fm_loss = 0
    total_fgd_loss = 0
    total_beat_loss = 0
    total_samples = 0

    for batch_data, _ in loader:
        # Encode through frozen VAE
        mu, z_padding_mask = encode_batch(vae_model, batch_data, device)

        audio = batch_data['audio'].to(device)
        speaker_id = batch_data['speaker_id'].to(device)
        gesture_type = batch_data['gesture_type'].to(device)
        padding_mask = batch_data['padding_mask'].to(device)
        anchor_pool = batch_data['anchor_pool'].to(device)
        anchor_audio = batch_data['anchor_audio'].to(device)
        text_features = batch_data.get('text')
        if text_features is not None:
            text_features = text_features.to(device)
        B = mu.shape[0]

        # Anchor dropout (so model works without anchors too)
        if anchor_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < anchor_drop_prob
            if drop_mask.all():
                anchor_pool_input = None
                anchor_audio_input = None
            elif drop_mask.any():
                anchor_pool_input = anchor_pool.clone()
                anchor_audio_input = anchor_audio.clone()
                anchor_pool_input[drop_mask] = 0.0
                anchor_audio_input[drop_mask] = 0.0
            else:
                anchor_pool_input = anchor_pool
                anchor_audio_input = anchor_audio
        else:
            anchor_pool_input = anchor_pool
            anchor_audio_input = anchor_audio

        # === Flow matching forward pass (manual, for predicted-x0 trick) ===
        t = torch.sigmoid(torch.randn(B, device=device) * 0.5).clamp(1e-5, 1 - 1e-5)

        z1 = (mu - model.z_mean) / model.z_std
        if z_padding_mask is not None:
            z1 = z1 * (~z_padding_mask).unsqueeze(-1).float()

        z0 = torch.randn_like(z1)
        if z_padding_mask is not None:
            z0 = z0 * (~z_padding_mask).unsqueeze(-1).float()

        t_expand = t[:, None, None]  # always temporal: (B, 1, 1)
        z_t = (1 - t_expand) * z0 + t_expand * z1
        v_target = z1 - z0

        # Encode audio conditioning (with anchors and optional text)
        audio_ctx = model.encode_audio(
            audio, speaker_id, gesture_type, padding_mask,
            anchor_frames=anchor_pool_input, anchor_audio=anchor_audio_input,
            text_features=text_features)

        # CFG dropout
        if cond_drop_prob > 0 and model.training:
            cfg_drop = torch.rand(B, device=device) < cond_drop_prob
            if cfg_drop.any():
                null = model.null_token.expand(B, audio_ctx.shape[1], -1)
                audio_ctx = torch.where(cfg_drop[:, None, None], null, audio_ctx)

        # Predict velocity
        v_pred = model.predict_velocity(z_t, t, audio_ctx, z_padding_mask=z_padding_mask)

        # Flow matching loss
        if z_padding_mask is not None:
            valid = (~z_padding_mask).unsqueeze(-1).float()
            fm_loss = ((v_pred - v_target) ** 2 * valid).sum() / valid.sum() / mu.shape[-1]
        else:
            fm_loss = F.mse_loss(v_pred, v_target)

        # === E2E perceptual losses (predicted-x0 trick) ===
        fgd_loss = torch.tensor(0.0, device=device)
        b_loss = torch.tensor(0.0, device=device)
        z_clean = None

        if lambda_fgd > 0 or lambda_beat > 0:
            # Predicted-x0 trick: z1_hat = z_t + (1-t)*v_pred
            # Only use samples where t > e2e_t_threshold for cleaner estimates
            t_mask = t > e2e_t_threshold  # (B,)
            if t_mask.any():
                z1_hat = z_t + (1 - t_expand) * v_pred
                z_clean = z1_hat * model.z_std + model.z_mean
                # Mask to only use high-t samples
                z_clean = z_clean[t_mask]
            else:
                z_clean = None

        if z_clean is not None and (lambda_fgd > 0 or lambda_beat > 0):

            gen_len = audio.shape[1]
            actual_vae = vae_model.module if hasattr(vae_model, 'module') else vae_model

            # Subset conditioning for masked samples
            gesture_type_sub = gesture_type[t_mask]
            audio_sub = audio[t_mask]
            speaker_id_sub = speaker_id[t_mask]
            z_pmask_sub = z_padding_mask[t_mask] if z_padding_mask is not None else None

            # Decode predicted z through frozen VAE (grads flow through z → prior)
            pred_motion, _ = actual_vae.decode(
                z_clean, gen_len,
                gesture_type=gesture_type_sub,
                audio_features=audio_sub,
                speaker_id=speaker_id_sub,
                z_padding_mask=z_pmask_sub,
            )

            # GT motion: raw poses (no VAE decoding needed)
            gt_motion = batch_data['poses'].to(device)[t_mask]

            T_motion = pred_motion.shape[1]
            T_gt = gt_motion.shape[1]
            T_common = min(T_motion, T_gt)

            if lambda_fgd > 0 and T_common >= 64:
                fgd_loss = compute_fgd_moment_matching_loss(
                    pred_motion[:, :T_common], gt_motion[:, :T_common],
                    fgd_model, device, lambda_cov=lambda_cov)

            if lambda_beat > 0 and T_common >= 3:
                audio_for_beat = audio_sub[:, :T_common]
                b_loss = beat_alignment_loss(pred_motion[:, :T_common], audio_for_beat)

        # Motion reconstruction loss (direct MSE on poses)
        recon_loss = torch.tensor(0.0, device=device)
        if lambda_recon > 0 and z_clean is not None:
            try:
                T_common_r = min(pred_motion.shape[1], gt_motion.shape[1])
                if T_common_r >= 1:
                    recon_loss = F.mse_loss(pred_motion[:, :T_common_r], gt_motion[:, :T_common_r])
            except NameError:
                pass

        loss = fm_loss + lambda_fgd * fgd_loss + lambda_beat * b_loss + lambda_recon * recon_loss

        optimizer.zero_grad()
        loss.backward()
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 1:
            # Clip each param group separately
            for pg in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(pg['params'], grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * B
        total_fm_loss += fm_loss.item() * B
        total_fgd_loss += fgd_loss.item() * B
        total_beat_loss += b_loss.item() * B
        total_samples += B

    return {
        'total': total_loss / total_samples,
        'fm': total_fm_loss / total_samples,
        'fgd': total_fgd_loss / total_samples,
        'beat': total_beat_loss / total_samples,
    }


@torch.no_grad()
def validate_fgd(model, vae_model, fgd_model, loader, device, num_steps=50, use_text=False):
    """Validate by computing actual FGD: sample z, decode to motion, compute Fréchet distance.

    This is the real metric we care about — distributional distance in FGD feature space.
    """
    from scipy import linalg

    model.eval()
    pred_features_all = []
    gt_features_all = []
    total_fm_loss = 0
    total_samples = 0

    for batch_data, _ in loader:
        mu, z_padding_mask = encode_batch(vae_model, batch_data, device)

        audio = batch_data['audio'].to(device)
        speaker_id = batch_data['speaker_id'].to(device)
        gesture_type = batch_data['gesture_type'].to(device)
        padding_mask = batch_data['padding_mask'].to(device)
        anchor_pool = batch_data['anchor_pool'].to(device)
        anchor_audio = batch_data['anchor_audio'].to(device)
        text_features = batch_data.get('text')
        if text_features is not None:
            text_features = text_features.to(device)
        B = mu.shape[0]

        # FM loss (cheap)
        fm_loss = model.compute_loss(
            mu, audio, speaker_id, gesture_type, padding_mask,
            z_padding_mask=z_padding_mask, cond_drop_prob=0.0,
            anchor_frames=anchor_pool, anchor_audio=anchor_audio,
            text_features=text_features)
        total_fm_loss += fm_loss.item() * B
        total_samples += B

        # Sample z via ODE
        z_sampled = model.sample(
            audio, speaker_id, gesture_type, padding_mask,
            num_steps=num_steps, guidance_scale=0.5,
            anchor_frames=anchor_pool, anchor_audio=anchor_audio,
            z_seq_len=mu.shape[1], text_features=text_features)

        gen_len = audio.shape[1]
        actual_vae = vae_model.module if hasattr(vae_model, 'module') else vae_model

        # Decode sampled z -> motion
        pred_motion, _ = actual_vae.decode(
            z_sampled, gen_len,
            gesture_type=gesture_type,
            audio_features=audio,
            speaker_id=speaker_id,
            z_padding_mask=z_padding_mask,
        )

        # Raw GT poses (not VAE-reconstructed) for real FGD
        gt_motion = batch_data['poses'].to(device)

        T_motion = pred_motion.shape[1]
        T_gt = gt_motion.shape[1]
        T_common = min(T_motion, T_gt)
        T_trunc = (T_common // 32) * 32
        if T_trunc < 32:
            continue

        # Convert to rot6d and extract FGD features
        pred_r6d = axis_angle_to_rot6d(pred_motion[:, :T_trunc].reshape(B, T_trunc, 55, 3))
        pred_r6d = pred_r6d.reshape(B, T_trunc, 330)
        pred_feat = fgd_model.map2latent(pred_r6d).cpu().numpy()
        # Flatten (B, T_down, 240) -> (B*T_down, 240) to match FGD convention
        pred_feat = pred_feat.reshape(-1, pred_feat.shape[-1])

        gt_r6d = axis_angle_to_rot6d(gt_motion[:, :T_trunc].reshape(B, T_trunc, 55, 3))
        gt_r6d = gt_r6d.reshape(B, T_trunc, 330)
        gt_feat = fgd_model.map2latent(gt_r6d).cpu().numpy()
        gt_feat = gt_feat.reshape(-1, gt_feat.shape[-1])

        pred_features_all.append(pred_feat)
        gt_features_all.append(gt_feat)

    # Compute Fréchet distance
    pred_features = np.concatenate(pred_features_all, axis=0)
    gt_features = np.concatenate(gt_features_all, axis=0)

    mu_pred = np.mean(pred_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False)

    diff = mu_pred - mu_gt
    eps = 1e-6
    offset = np.eye(sigma_pred.shape[0]) * eps
    covmean = linalg.sqrtm((sigma_pred + offset).dot(sigma_gt + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fgd = float(diff.dot(diff) + np.trace(sigma_pred) + np.trace(sigma_gt) - 2 * np.trace(covmean))

    return {
        'fm_loss': total_fm_loss / total_samples,
        'fgd': fgd,
    }


@torch.no_grad()
def validate_fgd_autoregressive(model, vae_model, fgd_model, device,
                                 data_root='BEAT2', language='english',
                                 speaker=None, test_fraction=1.0, seed=42,
                                 chunk_length=150, anchor_frames=90,
                                 num_steps=50, guidance_scale=0.5,
                                 velocity_scale=1.12, text_dim=0, text_dir=None):
    """Validate using the same autoregressive pipeline as test evaluation.

    Runs full-recording generation with anchor chaining + velocity scaling,
    matching the exact test-time procedure. Slower but gives accurate FGD.
    """
    import contextlib, io, random
    from scipy import linalg
    from generate_autoregressive_v2 import load_recording, build_chunks
    from prior_net.generate_diffusion import generate_autoregressive_with_diffusion
    from evaluate_generation import get_test_files, axis_angle_to_rot6d

    model.eval()

    # Clear stale z_padding_mask from training forward passes
    actual_vae = vae_model.module if hasattr(vae_model, 'module') else vae_model
    if hasattr(actual_vae, '_z_padding_mask'):
        actual_vae._z_padding_mask = None

    # Use val split files (not test — no leakage)
    test_files = get_test_files(data_root, language, test_fraction, seed, speaker=speaker, split='val')

    pred_features_all = []
    gt_features_all = []

    for pose_path in test_files:
        basename = os.path.splitext(os.path.basename(pose_path))[0]
        try:
            recording = load_recording(pose_path, language, data_root)
        except Exception:
            continue

        # Load text
        text_features_np = None
        if text_dim > 0 and text_dir is not None:
            text_path = os.path.join(text_dir, f'{basename}.npy')
            if os.path.exists(text_path):
                text_features_np = np.load(text_path)
                text_features_np = text_features_np[:recording['audio'].shape[0]]

        chunks = build_chunks(recording, max_chunk_length=chunk_length)
        if not chunks:
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            gen_poses, _ = generate_autoregressive_with_diffusion(
                vae_model, model, recording, chunks, device,
                anchor_k=anchor_frames, overlap=0,
                num_steps=num_steps, guidance_scale=guidance_scale,
                velocity_scale=velocity_scale,
                text_features_np=text_features_np,
            )

        T_gen = gen_poses.shape[0]
        start_frame = chunks[0]['start']
        gt_poses = torch.FloatTensor(recording['poses'][start_frame:start_frame + T_gen])
        T = min(T_gen, gt_poses.shape[0])
        T_trunc = (T // 32) * 32
        if T_trunc < 64:
            continue

        gen_r6d = axis_angle_to_rot6d(gen_poses[:T_trunc].reshape(T_trunc, 55, 3)).reshape(1, T_trunc, 330).to(device)
        gt_r6d = axis_angle_to_rot6d(gt_poses[:T_trunc].reshape(T_trunc, 55, 3)).reshape(1, T_trunc, 330).to(device)

        pred_feat = fgd_model.map2latent(gen_r6d.float()).cpu().numpy().reshape(-1, 240)
        gt_feat = fgd_model.map2latent(gt_r6d.float()).cpu().numpy().reshape(-1, 240)
        pred_features_all.append(pred_feat)
        gt_features_all.append(gt_feat)

    if not pred_features_all:
        return {'fgd_autoreg': float('inf')}

    pred_features = np.concatenate(pred_features_all, axis=0)
    gt_features = np.concatenate(gt_features_all, axis=0)

    mu_pred = np.mean(pred_features, axis=0)
    sigma_pred = np.cov(pred_features, rowvar=False)
    mu_gt = np.mean(gt_features, axis=0)
    sigma_gt = np.cov(gt_features, rowvar=False)

    diff = mu_pred - mu_gt
    eps = 1e-6
    offset = np.eye(sigma_pred.shape[0]) * eps
    covmean = linalg.sqrtm((sigma_pred + offset).dot(sigma_gt + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fgd = float(diff.dot(diff) + np.trace(sigma_pred) + np.trace(sigma_gt) - 2 * np.trace(covmean))

    return {'fgd_autoreg': fgd}


def main():
    parser = argparse.ArgumentParser(description='E2E perceptual flow matching prior (online)')
    parser.add_argument('--checkpoint-dir', type=str, default='e2e_perceptual/checkpoints_e2e')
    parser.add_argument('--resume', type=str, default=None)
    # VAE (frozen)
    parser.add_argument('--vae-checkpoint', type=str, default='checkpoint/beat2_poses/0/vae_temporal_lean/best.pt')
    parser.add_argument('--dataset-name', type=str, default='beat2_poses')
    parser.add_argument('--run-num', type=int, default=0)
    parser.add_argument('--folder-name', type=str, default='vae_temporal_lean')
    # Diffusion model
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--temporal-downsample', type=int, default=8)
    parser.add_argument('--audio-dim', type=int, default=768)
    parser.add_argument('--d-model', type=int, default=192)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-audio-layers', type=int, default=3)
    parser.add_argument('--num-denoiser-layers', type=int, default=4)
    parser.add_argument('--dim-feedforward', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-speakers', type=int, default=31)
    parser.add_argument('--text-dim', type=int, default=0, help='Text embedding dim (768 for BERT, 0 to disable)')
    # Loss weights
    parser.add_argument('--lambda-fgd', type=float, default=0.1, help='Weight for FGD moment matching loss')
    parser.add_argument('--lambda-beat', type=float, default=0.1)
    parser.add_argument('--lambda-cov', type=float, default=1.0, help='Weight for covariance matching in moment loss')
    parser.add_argument('--lambda-recon', type=float, default=0.0, help='Weight for motion reconstruction MSE loss')
    parser.add_argument('--e2e-t-threshold', type=float, default=0.5, help='Min t for predicted-x0 trick (higher = cleaner but fewer samples)')
    parser.add_argument('--unfreeze-decoder', action='store_true', help='Unfreeze VAE decoder for joint training')
    parser.add_argument('--decoder-lr', type=float, default=1e-6, help='Learning rate for unfrozen decoder')
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--cond-drop-prob', type=float, default=0.1)
    parser.add_argument('--anchor-drop-prob', type=float, default=0.3)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-every', type=int, default=10)
    # Dataset
    parser.add_argument('--data-root', type=str, default='BEAT2')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--min-length', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=300)
    parser.add_argument('--anchor-max-frames', type=int, default=30)
    parser.add_argument('--speaker', type=int, default=None, help='Train/val on single speaker only (e.g. 2 for scott)')
    parser.add_argument('--text-dir', type=str, default=None, help='Directory with pre-computed text embeddings (.npy)')
    parser.add_argument('--val-autoreg-every', type=int, default=5, help='Run autoregressive FGD validation every N epochs (0=off)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_csv = os.path.join(args.checkpoint_dir, 'loss_log.csv')

    # Load VAE config
    options, _ = conf_parser(args.dataset_name, args.run_num, args.folder_name)

    # Data — BEAT2 dataset
    use_text = args.text_dim > 0
    print(f"Loading BEAT2 dataset...{' (with text)' if use_text else ''}")
    train_dataset = BEAT2PoseDataset(
        data_path=args.data_root, language=args.language,
        min_length=args.min_length, max_length=args.max_length,
        use_axis_angle=True, split='train',
        anchor_max_frames=args.anchor_max_frames,
        speaker=args.speaker,
        load_text=use_text, text_dir=args.text_dir,
    )
    val_dataset = BEAT2PoseDataset(
        data_path=args.data_root, language=args.language,
        min_length=args.min_length, max_length=args.max_length,
        use_axis_angle=True, split='val',
        anchor_max_frames=args.anchor_max_frames,
        speaker=args.speaker,
        load_text=use_text, text_dir=args.text_dir,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=variable_length_collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=variable_length_collate_fn)

    # Load frozen VAE
    print("Loading frozen VAE...")
    model_type = get_model_type(args.folder_name)
    vae_model = create_model_object(model_type, options)
    vae_state = load_checkpoint(args.vae_checkpoint, device)
    model_sd = vae_model.state_dict()
    filtered = {k: v for k, v in vae_state.items() if k in model_sd and v.shape == model_sd[k].shape}
    vae_model.load_state_dict(filtered, strict=False)
    vae_model.to(device).eval()
    if args.unfreeze_decoder:
        # Freeze encoder, unfreeze decoder
        actual_vae = vae_model.module if hasattr(vae_model, 'module') else vae_model
        for p in vae_model.parameters():
            p.requires_grad = False
        # Unfreeze decoder parameters
        for name, p in actual_vae.named_parameters():
            if 'decoder' in name or 'output_proj' in name or 'level_output' in name:
                p.requires_grad = True
        n_dec = sum(p.numel() for p in vae_model.parameters() if p.requires_grad)
        print(f"  VAE loaded — encoder frozen, decoder unfrozen ({n_dec:,} trainable params)")
    else:
        for p in vae_model.parameters():
            p.requires_grad = False
        print("  VAE loaded and frozen")

    # Compute z normalization stats from a pass over training data
    print("Computing z normalization stats...")
    z_stats_tokens = []
    with torch.no_grad():
        for i, (batch_data, _) in enumerate(train_loader):
            mu, z_pmask = encode_batch(vae_model, batch_data, device)
            if z_pmask is not None:
                for j in range(mu.shape[0]):
                    valid_len = (~z_pmask[j]).sum().item()
                    z_stats_tokens.append(mu[j, :valid_len].cpu())
            else:
                z_stats_tokens.append(mu.reshape(-1, mu.shape[-1]).cpu())
            if i >= 50:  # sample ~50 batches for stats
                break
    all_tokens = torch.cat(z_stats_tokens, dim=0)
    z_mean = all_tokens.mean(dim=0)
    z_std = all_tokens.std(dim=0).clamp(min=1e-6)
    print(f"  z stats from {all_tokens.shape[0]} tokens: mean_norm={z_mean.norm():.2f}, mean_std={z_std.mean():.4f}")

    # Diffusion model
    model = TemporalFlowMatchingPrior(
        latent_dim=args.latent_dim,
        audio_dim=args.audio_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_audio_layers=args.num_audio_layers,
        num_denoiser_layers=args.num_denoiser_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_speakers=args.num_speakers,
        temporal_downsample=args.temporal_downsample,
        cond_drop_prob=args.cond_drop_prob,
        text_dim=args.text_dim,
    ).to(device)

    model.z_mean.copy_(z_mean.to(device))
    model.z_std.copy_(z_std.to(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Diffusion model: {n_params:,} parameters")

    # EMA
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    # Load FGD feature extractor
    print("Loading FGD feature extractor...")
    fgd_model = load_fgd_feature_extractor(device)
    print("  FGD extractor loaded")

    # Optimizer
    if args.unfreeze_decoder:
        decoder_params = [p for p in vae_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': args.lr},
            {'params': decoder_params, 'lr': args.decoder_lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_fgd = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        if missing:
            print(f"  Missing keys (randomly initialized): {missing}")
        if 'ema_model' in ckpt:
            ema_model.load_state_dict(ckpt['ema_model'], strict=False)
        if 'z_mean' in ckpt:
            model.z_mean.copy_(ckpt['z_mean'].to(device))
            model.z_std.copy_(ckpt['z_std'].to(device))
        print(f"  Loaded weights from epoch {ckpt.get('epoch', '?')}")
        # Don't restore optimizer/scheduler — fresh training with E2E losses

    # CSV log
    csv_fields = ['epoch', 'train_total', 'train_fm', 'train_fgd', 'train_beat',
                  'val_fm_loss', 'val_fgd', 'lr', 'val_fgd_autoreg']
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(csv_fields)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  lambda_fgd={args.lambda_fgd}, lambda_cov={args.lambda_cov}, lambda_beat={args.lambda_beat}")
    print(f"  e2e_t_threshold={args.e2e_t_threshold}")
    print(f"  anchor_drop_prob={args.anchor_drop_prob}")
    print(f"  batch_size={args.batch_size}, lr={args.lr}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model, vae_model, fgd_model, train_loader, optimizer, device,
            grad_clip=args.grad_clip, cond_drop_prob=args.cond_drop_prob,
            anchor_drop_prob=args.anchor_drop_prob,
            lambda_fgd=args.lambda_fgd, lambda_beat=args.lambda_beat,
            lambda_cov=args.lambda_cov,
            e2e_t_threshold=args.e2e_t_threshold,
            lambda_recon=args.lambda_recon, use_text=use_text)

        update_ema(ema_model, model, args.ema_decay)

        # Full FGD validation every epoch
        val_metrics = validate_fgd(model, vae_model, fgd_model, val_loader, device, num_steps=5, use_text=use_text)
        val_fm_loss = val_metrics['fm_loss']
        val_fgd = val_metrics['fgd']

        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Autoregressive FGD (matches test-time pipeline)
        val_fgd_ar = None
        if args.val_autoreg_every > 0 and (epoch + 1) % args.val_autoreg_every == 0:
            ar_metrics = validate_fgd_autoregressive(
                model, vae_model, fgd_model, device,
                data_root=args.data_root, language=args.language,
                speaker=2, test_fraction=1.0, seed=42,
                chunk_length=150, anchor_frames=90,
                num_steps=50, guidance_scale=1.0,
                velocity_scale=1.0,
                text_dim=args.text_dim, text_dir=args.text_dir,
            )
            val_fgd_ar = ar_metrics['fgd_autoreg']

        with open(log_csv, 'a', newline='') as f:
            row = [
                epoch,
                f"{train_metrics['total']:.6f}",
                f"{train_metrics['fm']:.6f}",
                f"{train_metrics['fgd']:.6f}",
                f"{train_metrics['beat']:.6f}",
                f"{val_fm_loss:.6f}",
                f"{val_fgd:.4f}",
                f"{lr:.6f}",
            ]
            if val_fgd_ar is not None:
                row.append(f"{val_fgd_ar:.4f}")
            csv.writer(f).writerow(row)

        # Best checkpoint selected by AR FGD only (most faithful to eval pipeline).
        # Batch val_fgd is logged for monitoring but not used for checkpoint selection.
        if val_fgd_ar is not None:
            is_best = val_fgd_ar < best_fgd
            if is_best:
                best_fgd = val_fgd_ar
        else:
            is_best = False

        ar_str = f" ar_FGD={val_fgd_ar:.4f}" if val_fgd_ar is not None else ""
        tqdm.write(
            f"Ep {epoch:3d} | total={train_metrics['total']:.4f} "
            f"fm={train_metrics['fm']:.4f} fgd_loss={train_metrics['fgd']:.4f} "
            f"beat={train_metrics['beat']:.4f} | val_fm={val_fm_loss:.4f} "
            f"val_FGD={val_fgd:.4f}{ar_str} lr={lr:.6f}"
            f"{' *BEST*' if is_best else ''}")

        save_dict = {
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_fgd': best_fgd,
            'z_mean': model.z_mean,
            'z_std': model.z_std,
            'args': vars(args),
            'is_temporal': True,
        }
        if args.unfreeze_decoder:
            save_dict['vae_state'] = vae_model.state_dict()

        if is_best:
            torch.save(save_dict, os.path.join(args.checkpoint_dir, 'best.pt'))

        if (epoch + 1) % args.save_every == 0:
            torch.save(save_dict, os.path.join(args.checkpoint_dir, f'{epoch:03d}.pt'))

    print("Done.")


if __name__ == '__main__':
    main()
