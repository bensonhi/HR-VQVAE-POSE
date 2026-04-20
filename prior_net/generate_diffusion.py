"""
Generate motion using the flow-matching diffusion prior.

Drop-in replacement for generate.py but uses FlowMatchingPrior for z sampling.
Supports both single-z and temporal z sequences.
"""
import argparse
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_autoregressive_v2 import load_recording, build_chunks
from prior_net.diffusion_model import FlowMatchingPrior, TemporalFlowMatchingPrior
from m_util import conf_parser, create_model_object, get_model_type, load_checkpoint


def generate_autoregressive_with_diffusion(
    vae_model, diff_model, recording, chunks, device,
    anchor_k=90, overlap=0, num_steps=50, guidance_scale=1.0,
    smooth_sigma=0.0, temperature=1.0, num_samples=1, truncation=1.0,
    refine_strength=0.0, use_euler=False, velocity_scale=1.0,
    text_features_np=None,
):
    """Generate motion using diffusion prior for z prediction.

    Works with both single-z and temporal diffusion priors.

    Args:
        vae_model: Frozen VAE decoder.
        diff_model: Trained FlowMatchingPrior or TemporalFlowMatchingPrior.
        recording: Dict from load_recording().
        chunks: List of chunk dicts from build_chunks().
        device: torch device.
        anchor_k: Number of anchor frames.
        overlap: Crossfade overlap frames between chunks.
        num_steps: ODE integration steps for diffusion sampling.
        guidance_scale: CFG guidance scale (1.0 = no guidance).

    Returns:
        all_poses: (T_total, 165) tensor.
        blend_mask: (T_total,) bool array.
    """
    actual_vae = vae_model.module if hasattr(vae_model, 'module') else vae_model
    actual_vae.eval()
    diff_model.eval()

    is_temporal = isinstance(diff_model, TemporalFlowMatchingPrior) or hasattr(diff_model, 'predict_z')
    temporal_downsample = getattr(actual_vae, 'temporal_downsample', 1)

    poses_np = recording['poses']
    audio_np = recording['audio']
    speaker_id = recording['speaker_id']
    speaker_id_tensor = torch.tensor([speaker_id], dtype=torch.long, device=device)

    output_parts = []
    blend_parts = []
    prev_anchor = None
    prev_anchor_audio = None
    output_end_frame = chunks[0]['start']

    for ci, chunk in enumerate(chunks):
        start, end = chunk['start'], chunk['end']
        chunk_len = chunk['length']
        gtype = chunk['gesture_type']
        gesture_type_tensor = torch.tensor([gtype], dtype=torch.long, device=device)

        # Overlap
        if ci == 0:
            ov = 0
        else:
            prev_len = output_parts[-1].shape[0] if output_parts else 0
            ov = min(overlap, prev_len, start, chunk_len)

        gen_len = chunk_len + ov
        audio_start = start - ov

        audio_slice = torch.FloatTensor(
            audio_np[audio_start:end]
        ).unsqueeze(0).to(device)

        # Text features (optional)
        text_slice = None
        if text_features_np is not None:
            text_slice = torch.FloatTensor(
                text_features_np[audio_start:end]
            ).unsqueeze(0).to(device)

        # Anchor frames
        anchor_frames = None
        anchor_audio_slice = None
        if anchor_k > 0 and prev_anchor is not None:
            anchor_frames = prev_anchor
            anchor_audio_slice = prev_anchor_audio
        elif anchor_k > 0 and ci == 0 and start >= anchor_k:
            anchor_frames = torch.FloatTensor(
                poses_np[start - anchor_k:start]
            ).unsqueeze(0).to(device)
            anchor_audio_slice = torch.FloatTensor(
                audio_np[start - anchor_k:start]
            ).unsqueeze(0).to(device)

        # --- Sample z from diffusion prior ---
        with torch.no_grad():
            if is_temporal:
                z_seq_len = math.ceil(gen_len / temporal_downsample)
                z = diff_model.sample(
                    audio_slice, speaker_id_tensor, gesture_type_tensor,
                    padding_mask=None,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    anchor_frames=anchor_frames,
                    anchor_audio=anchor_audio_slice,
                    z_seq_len=z_seq_len,
                    temperature=temperature,
                    num_samples=num_samples,
                    truncation=truncation,
                    refine_strength=refine_strength,
                    use_euler=use_euler,
                    text_features=text_slice,
                )
                gen, _anchor_recon = actual_vae.decode(
                    z, gen_len,
                    gesture_type=gesture_type_tensor,
                    audio_features=audio_slice,
                    anchor_frames=anchor_frames,
                    speaker_id=speaker_id_tensor,
                    anchor_audio=anchor_audio_slice,
                    z_padding_mask=None,
                )
            else:
                z = diff_model.sample(
                    audio_slice, speaker_id_tensor, gesture_type_tensor,
                    padding_mask=None,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
                gen, _anchor_recon = actual_vae.decode(
                    z, gen_len,
                    gesture_type=gesture_type_tensor,
                    audio_features=audio_slice,
                    anchor_frames=anchor_frames,
                    speaker_id=speaker_id_tensor,
                    anchor_audio=anchor_audio_slice,
                )
        chunk_poses = gen[0].cpu()

        # Crossfade
        if ov > 0 and len(output_parts) > 0:
            t = torch.linspace(0, 1, ov)
            alpha = (0.5 * (1 - torch.cos(torch.pi * t))).unsqueeze(-1)
            prev_tail = output_parts[-1][-ov:]
            new_head = chunk_poses[:ov]
            blended = (1 - alpha) * prev_tail + alpha * new_head
            output_parts[-1] = output_parts[-1][:-ov]
            blend_parts[-1] = blend_parts[-1][:-ov]
            output_parts.append(blended)
            blend_parts.append(np.ones(ov, dtype=bool))
            output_parts.append(chunk_poses[ov:])
            blend_parts.append(np.zeros(chunk_poses.shape[0] - ov, dtype=bool))
        else:
            output_parts.append(chunk_poses)
            blend_parts.append(np.zeros(chunk_poses.shape[0], dtype=bool))

        output_end_frame = end

        # Update anchor
        if anchor_k > 0:
            all_so_far = torch.cat(output_parts, dim=0)
            n_out = all_so_far.shape[0]
            tail_len = min(anchor_k, n_out)
            tail = all_so_far[-tail_len:]
            prev_anchor = tail.unsqueeze(0).to(device)

            anchor_audio_start = max(0, output_end_frame - tail_len)
            anchor_audio_end = output_end_frame
            prev_anchor_audio = torch.FloatTensor(
                audio_np[anchor_audio_start:anchor_audio_end]
            ).unsqueeze(0).to(device)

        gtype_str = 'semantic' if gtype else 'beat'
        ov_str = f', blend={ov}fr' if ov > 0 else ''
        print(f"  Chunk {ci+1}/{len(chunks)}: frames {start}-{end} "
              f"({chunk_len}fr, {gtype_str}{ov_str})")

    all_poses = torch.cat(output_parts, dim=0)
    blend_mask = np.concatenate(blend_parts)

    # Post-hoc smoothing on blend regions
    if blend_mask.any():
        poses_arr = all_poses.numpy().copy()
        smoothed = gaussian_filter1d(poses_arr, sigma=3.0, axis=0)
        weight = np.zeros(len(blend_mask), dtype=np.float32)
        weight[blend_mask] = 1.0
        weight = gaussian_filter1d(weight, sigma=2.0)[:, None]
        poses_arr = (1 - weight) * poses_arr + weight * smoothed
        all_poses = torch.FloatTensor(poses_arr)

    # Global temporal smoothing (reduces high-frequency jitter)
    if smooth_sigma > 0:
        poses_arr = all_poses.numpy().copy()
        all_poses = torch.FloatTensor(gaussian_filter1d(poses_arr, sigma=smooth_sigma, axis=0))

    # Velocity scaling: amplify/reduce motion dynamics
    should_scale = not (isinstance(velocity_scale, (int, float)) and velocity_scale == 1.0)
    if should_scale:
        poses_arr = all_poses.numpy().copy()
        # Compute per-frame mean as "rest pose" trajectory
        mean_pose = gaussian_filter1d(poses_arr, sigma=15.0, axis=0)
        # Scale deviations from smoothed trajectory
        deviation = poses_arr - mean_pose
        if isinstance(velocity_scale, (list, tuple, np.ndarray)):
            # Per-joint velocity scaling (55 joints x 3 = 165 dims)
            scale = np.array(velocity_scale, dtype=np.float32)
            poses_arr = mean_pose + deviation * scale[None, :]
        else:
            poses_arr = mean_pose + deviation * velocity_scale
        all_poses = torch.FloatTensor(poses_arr)

    return all_poses, blend_mask
