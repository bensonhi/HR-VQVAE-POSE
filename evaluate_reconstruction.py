"""
Evaluate reconstruction upper bound on BEAT2 test set.

For each test recording, does chunk-by-chunk autoregressive generation
BUT instead of sampling z from N(0,1), ENCODES the GT poses for each chunk
to get mu from the encoder (reconstruction mode).

This measures the ceiling: how good generation can be with a perfect prior.

Uses the same metrics as evaluate_generation.py:
  - FGD   (Frechet Gesture Distance) — distributional realism (lower=better)
  - BC    (Beat Consistency) — audio-motion synchronization (higher=better)
  - L1Div (Diversity) — variety of generated motions (higher=better)
  - LVD   (Lip Vertex Distance) — face metric (lower=better)
  - MSEFace — face vertex MSE (lower=better)
  - MPJPE — mean per-joint position error in mm (lower=better)
  - Velocity — motion velocity comparison

Key difference from evaluate_generation.py:
  - z comes from encode(GT_chunk_poses) -> mu, NOT from N(0,1)
  - Anchor frames still come from the *decoded* output (not GT) to
    faithfully simulate the autoregressive pipeline

Usage:
  conda run -n hr-vqvae-poses python evaluate_reconstruction.py

  conda run -n hr-vqvae-poses python evaluate_reconstruction.py \
      --test-fraction 0.2 --overlap 10
"""
import argparse
import contextlib
import csv
import io
import json
import os
import random
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _this_dir)
sys.path.append(_parent_dir)

from generate_autoregressive_v2 import (
    load_recording, build_chunks,
)
from m_util import conf_parser, create_model_object, get_model_type, load_checkpoint
from m_smplx_layer import SMPLXLayer
from emage_evaltools.mertic import FGD, BC, L1div, LVDFace, MSEFace
from evaluate_generation import get_test_files as _get_test_files_with_split


# ============================================================================
# Rotation conversions (same as evaluate_generation.py)
# ============================================================================

def axis_angle_to_rot6d(axis_angle):
    """Convert axis-angle (..., 3) to rotation 6D (..., 6)."""
    batch_shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)

    angle = torch.norm(aa, dim=1, keepdim=True)  # (N, 1)
    axis = aa / angle.clamp(min=1e-8)  # (N, 3)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    t = 1 - cos_a

    x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]

    R = torch.cat([
        t*x*x + cos_a,   t*x*y - sin_a*z, t*x*z + sin_a*y,
        t*x*y + sin_a*z, t*y*y + cos_a,   t*y*z - sin_a*x,
        t*x*z - sin_a*y, t*y*z + sin_a*x, t*z*z + cos_a,
    ], dim=1).reshape(-1, 3, 3)

    # Identity for near-zero angles
    mask = (angle.squeeze(-1) < 1e-8)
    if mask.any():
        R[mask] = torch.eye(3, device=aa.device, dtype=aa.dtype)

    rot6d = R[:, :2, :].reshape(-1, 6)
    return rot6d.reshape(*batch_shape, 6)


# ============================================================================
# Test file listing (same as evaluate_generation.py)
# ============================================================================

def get_test_files(data_root='BEAT2', language='english', fraction=0.1, seed=42, speaker=None, split='test'):
    return _get_test_files_with_split(data_root, language, fraction, seed, speaker=speaker, split=split)


# ============================================================================
# SMPLX joint position extraction (same as evaluate_generation.py)
# ============================================================================

def poses_to_joints_and_vertices(poses_tensor, smplx_layer, device, betas=None, chunk_size=128):
    """Convert axis-angle poses (T, 165) to joints (T, 55, 3) and vertices (T, V, 3)."""
    T = poses_tensor.shape[0]
    all_joints = []
    all_verts = []

    betas_tensor = None
    if betas is not None:
        if not isinstance(betas, torch.Tensor):
            betas = torch.FloatTensor(betas)
        betas_tensor = betas.unsqueeze(0).to(device)

    with torch.no_grad():
        for i in range(0, T, chunk_size):
            batch = poses_tensor[i:i+chunk_size].unsqueeze(0).to(device)
            verts, joints = smplx_layer(batch, betas=betas_tensor)
            all_joints.append(joints[0].cpu())
            all_verts.append(verts[0].cpu())

    return torch.cat(all_joints, dim=0), torch.cat(all_verts, dim=0)


FACE_VERTEX_INDICES = list(range(0, 2000))


# ============================================================================
# Reconstruction-mode autoregressive generation
# ============================================================================

def generate_autoregressive_reconstruction(model, recording, chunks, device,
                                           anchor_k=30, overlap=0):
    """Generate motion autoregressively using ENCODED GT poses instead of random z.

    For each chunk:
      1. Encode the GT poses for that chunk -> get mu (deterministic)
      2. Decode using that mu, with anchor frames from previous *decoded* output
      3. Crossfade overlap region (same logic as generate_autoregressive)

    This gives the reconstruction upper bound: how good the pipeline can be
    if the prior perfectly matches the encoder posterior.

    Args:
        model: the VAE model (in eval mode)
        recording: dict from load_recording()
        chunks: list of chunk dicts from build_chunks()
        device: torch device
        anchor_k: number of anchor frames between chunks
        overlap: number of crossfade frames between adjacent chunks

    Returns:
        all_poses: (T_total, 165) tensor of reconstructed poses
        blend_mask: (T_total,) bool array — True for frames in a crossfade region
    """
    actual_model = model.module if hasattr(model, 'module') else model
    actual_model.eval()

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

        # --- Determine overlap for this chunk ---
        if ci == 0:
            ov = 0
        else:
            prev_len = output_parts[-1].shape[0] if output_parts else 0
            audio_room = start
            ov = min(overlap, prev_len, audio_room, chunk_len)

        gen_len = chunk_len + ov
        audio_start = start - ov

        # Audio slice covering [audio_start, end)
        audio_slice = torch.FloatTensor(
            audio_np[audio_start:end]
        ).unsqueeze(0).to(device)

        # GT poses for this chunk (including overlap prefix) — used for ENCODING
        gt_chunk_poses = torch.FloatTensor(
            poses_np[audio_start:end]
        ).unsqueeze(0).to(device)  # (1, gen_len, 165)

        # --- Anchor frames ---
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

        # --- Encode GT poses to get latent ---
        with torch.no_grad():
            is_vqvae = hasattr(actual_model, 'quantizes')
            if is_vqvae:
                quant_sum, commitment_loss, quant_codes, embed_ids, z_padding_mask, _ = actual_model.encode(
                    gt_chunk_poses,
                    padding_mask=None,
                    gesture_type=gesture_type_tensor,
                    audio_features=audio_slice,
                    speaker_id=speaker_id_tensor,
                )
                latent = quant_sum
            else:
                z, mu, logvar, kl_for_loss, total_raw_kl, kl_losses = actual_model.encode(
                    gt_chunk_poses,
                    padding_mask=None,
                    gesture_type=gesture_type_tensor,
                    audio_features=audio_slice,
                    speaker_id=speaker_id_tensor,
                )
                latent = mu  # deterministic reconstruction
                z_padding_mask = getattr(actual_model, '_z_padding_mask', None)

            # --- Decode ---
            gen, _anchor_recon = actual_model.decode(
                latent, gen_len,
                gesture_type=gesture_type_tensor,
                audio_features=audio_slice,
                anchor_frames=anchor_frames,
                speaker_id=speaker_id_tensor,
                anchor_audio=anchor_audio_slice,
                z_padding_mask=z_padding_mask,
            )
        chunk_poses = gen[0].cpu()  # (gen_len, 165)

        # --- Crossfade overlap region ---
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

        # --- Update anchor from fused output (always anchor_k frames) ---
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
        if is_vqvae:
            latent_info = f"commit={commitment_loss.item():.4f}"
        else:
            latent_info = f"kl={total_raw_kl.item():.1f}"
        print(f"  Chunk {ci+1}/{len(chunks)}: frames {start}-{end} "
              f"({chunk_len}fr, {gtype_str}{ov_str}, {latent_info})")

    all_poses = torch.cat(output_parts, dim=0)
    blend_mask = np.concatenate(blend_parts)

    # --- Post-hoc temporal smoothing on blend regions ---
    if blend_mask.any():
        poses_np_out = all_poses.numpy().copy()
        smoothed = gaussian_filter1d(poses_np_out, sigma=3.0, axis=0)

        weight = np.zeros(len(blend_mask), dtype=np.float32)
        weight[blend_mask] = 1.0
        weight = gaussian_filter1d(weight, sigma=2.0)
        weight = weight[:, None]

        poses_np_out = (1 - weight) * poses_np_out + weight * smoothed
        all_poses = torch.FloatTensor(poses_np_out)

    return all_poses, blend_mask


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate reconstruction upper bound on BEAT2 test set')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoint/beat2_poses/0/vae_kalin/167.pt')
    parser.add_argument('--test-fraction', type=float, default=0.1,
                        help='Fraction of test set to evaluate (default: 0.1)')
    parser.add_argument('--chunk-length', type=int, default=150)
    parser.add_argument('--anchor-frames', type=int, default=90)
    parser.add_argument('--overlap', type=int, default=0,
                        help='Crossfade overlap frames between chunks (default: 0)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-name', type=str, default='beat2_poses')
    parser.add_argument('--run-num', type=int, default=0)
    parser.add_argument('--folder-name', type=str, default='vae_kalin')
    parser.add_argument('--data-root', type=str, default='BEAT2')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--speaker', type=int, default=None,
                        help='Evaluate only this speaker (default: all). Standard BEAT2 protocol uses --speaker 2')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'],
                        help='Which split to evaluate on (default: test)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: next to checkpoint)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Load model ----
    print("Loading model...")
    options, _ = conf_parser(args.dataset_name, args.run_num, args.folder_name)
    model_type = get_model_type(args.folder_name)
    model = create_model_object(model_type, options)
    state_dict = load_checkpoint(args.checkpoint, device)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.to(device).eval()
    print(f"  Checkpoint: {args.checkpoint}")

    # ---- Load SMPLX ----
    print("Loading SMPLX layer...")
    smplx_layer = SMPLXLayer(model_path='models_smplx_v1_1/models')
    smplx_layer.to(device)

    # ---- Get test files ----
    test_files = get_test_files(
        args.data_root, args.language, args.test_fraction, args.seed,
        speaker=args.speaker, split=args.split)
    print(f"\nTest files: {len(test_files)} "
          f"({args.test_fraction*100:.0f}% of {args.split} set)")

    # ---- Initialize metrics ----
    fgd_evaluator = FGD(download_path='./emage_evaltools/', device=str(device))
    bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    l1div_evaluator = L1div()

    lvd_evaluator = LVDFace()
    mse_face_evaluator = MSEFace()

    gt_bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    gt_l1div_evaluator = L1div()

    mse_sum = 0.0
    mse_count = 0
    mpjpe_sum = 0.0
    mpjpe_frames = 0

    # Per-level MPJPE: SMPLX joint layout (first 55 joints)
    # body: 0:22, face: 22:25, left_hand: 25:40, right_hand: 40:55
    LEVEL_JOINTS = {
        'body':       (0, 22),
        'face':       (22, 25),
        'left_hand':  (25, 40),
        'right_hand': (40, 55),
    }
    mpjpe_level_sum = {k: 0.0 for k in LEVEL_JOINTS}
    mpjpe_level_frames = {k: 0 for k in LEVEL_JOINTS}

    per_recording = []

    # ---- Evaluate ----
    skipped = 0
    for fi, pose_path in enumerate(tqdm(test_files, desc='Evaluating (recon)')):
        basename = os.path.splitext(os.path.basename(pose_path))[0]

        try:
            recording = load_recording(pose_path, args.language, args.data_root)
        except Exception as e:
            tqdm.write(f"  Skip {basename}: {e}")
            skipped += 1
            continue

        chunks = build_chunks(recording, max_chunk_length=args.chunk_length)
        if not chunks:
            skipped += 1
            continue

        # Generate via reconstruction (encode GT -> mu -> decode)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            gen_poses, blend_mask = generate_autoregressive_reconstruction(
                model, recording, chunks, device,
                anchor_k=args.anchor_frames,
                overlap=args.overlap,
            )

        T_gen = gen_poses.shape[0]

        # Align GT
        start_frame = chunks[0]['start']
        gt_poses = torch.FloatTensor(
            recording['poses'][start_frame:start_frame + T_gen])

        T = min(T_gen, gt_poses.shape[0])
        gen_poses = gen_poses[:T]
        gt_poses = gt_poses[:T]

        if T < 64:
            skipped += 1
            continue

        # --- FGD: axis-angle -> rot6d ---
        gen_r6d = axis_angle_to_rot6d(
            gen_poses.reshape(T, 55, 3)).reshape(1, T, 330).to(device)
        gt_r6d = axis_angle_to_rot6d(
            gt_poses.reshape(T, 55, 3)).reshape(1, T, 330).to(device)
        fgd_evaluator.update(gen_r6d.float(), gt_r6d.float())

        # --- Joint positions + vertices ---
        speaker_betas = recording.get('betas')
        gen_joints_full, gen_verts = poses_to_joints_and_vertices(
            gen_poses, smplx_layer, device, betas=speaker_betas)
        gt_joints_full, gt_verts = poses_to_joints_and_vertices(
            gt_poses, smplx_layer, device, betas=speaker_betas)

        gen_joints = gen_joints_full[:, :55, :]
        gt_joints = gt_joints_full[:, :55, :]

        gen_joints_np = gen_joints.reshape(T, -1).numpy()
        gt_joints_np = gt_joints.reshape(T, -1).numpy()

        # --- BC ---
        trim = 60
        wav_path = recording.get('wav_path')
        if wav_path and os.path.exists(wav_path) and T > trim * 2 + 30:
            audio_start_sec = (start_frame + trim) / 30.0
            audio_end_sec = (start_frame + T - trim) / 30.0
            t_start_audio = int(audio_start_sec * 16000)
            t_end_audio = int(audio_end_sec * 16000)

            try:
                audio_beats = bc_evaluator.load_audio(
                    wav_path, t_start=t_start_audio, t_end=t_end_audio)

                if len(audio_beats) == 0:
                    tqdm.write(f"  BC skip {basename}: no audio onsets detected")
                else:
                    gen_motion_beats = bc_evaluator.load_motion(
                        gen_joints_np, t_start=trim, t_end=T - trim,
                        pose_fps=30, without_file=True)
                    bc_evaluator.compute(
                        audio_beats, gen_motion_beats,
                        length=T - 2*trim, pose_fps=30)

                    gt_audio_beats = gt_bc_evaluator.load_audio(
                        wav_path, t_start=t_start_audio, t_end=t_end_audio)
                    gt_motion_beats = gt_bc_evaluator.load_motion(
                        gt_joints_np, t_start=trim, t_end=T - trim,
                        pose_fps=30, without_file=True)
                    gt_bc_evaluator.compute(
                        gt_audio_beats, gt_motion_beats,
                        length=T - 2*trim, pose_fps=30)
            except Exception as e:
                tqdm.write(f"  BC error {basename}: {e}")

        # --- L1Div (on joint positions, matches GestureLSM) ---
        l1div_evaluator.compute(gen_joints_np)
        gt_l1div_evaluator.compute(gt_joints_np)

        # --- LVD & MSEFace ---
        gen_face_verts = gen_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        gt_face_verts = gt_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        lvd_evaluator.compute(gen_face_verts, gt_face_verts)
        mse_face_evaluator.compute(gen_face_verts, gt_face_verts)

        # --- MSE (body joint positions) ---
        joint_diff = gen_joints - gt_joints
        n_joints = gen_joints.shape[1]
        mse_sum += (joint_diff ** 2).sum().item()
        mse_count += T * n_joints * 3

        # --- MPJPE ---
        per_joint_err = torch.norm(joint_diff, dim=-1)
        rec_mpjpe = per_joint_err.mean().item() * 1000
        mpjpe_sum += per_joint_err.sum().item()
        mpjpe_frames += T * n_joints

        # --- Per-level MPJPE ---
        for lvl, (j0, j1) in LEVEL_JOINTS.items():
            lvl_err = torch.norm(gen_joints[:, j0:j1, :] - gt_joints[:, j0:j1, :], dim=-1)
            mpjpe_level_sum[lvl] += lvl_err.sum().item()
            mpjpe_level_frames[lvl] += T * (j1 - j0)

        # --- Per-recording velocity ---
        gen_vel = torch.norm(
            gen_joints[1:] - gen_joints[:-1], dim=-1).mean().item()
        gt_vel = torch.norm(
            gt_joints[1:] - gt_joints[:-1], dim=-1).mean().item()

        per_recording.append({
            'basename': basename,
            'frames': T,
            'mpjpe_mm': round(rec_mpjpe, 2),
            'gen_vel': round(gen_vel, 6),
            'gt_vel': round(gt_vel, 6),
        })

        tqdm.write(
            f"  {basename}: {T}fr, MPJPE={rec_mpjpe:.1f}mm, "
            f"vel gen={gen_vel:.4f} gt={gt_vel:.4f}")

    # ---- Compute final metrics ----
    n_eval = len(test_files) - skipped
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION UPPER BOUND ({n_eval} recordings, "
          f"{args.test_fraction*100:.0f}% of test set)")
    print(f"{'='*60}")

    metrics = {}

    # FGD
    fgd = fgd_evaluator.compute()
    metrics['FGD'] = round(fgd, 4)

    # BC
    bc = bc_evaluator.avg() if bc_evaluator.counter > 0 else float('nan')
    gt_bc = gt_bc_evaluator.avg() if gt_bc_evaluator.counter > 0 else float('nan')
    metrics['BC'] = round(bc, 4)
    metrics['BC_GT'] = round(gt_bc, 4)

    # L1Div
    l1div = l1div_evaluator.avg()
    gt_l1div = gt_l1div_evaluator.avg()
    metrics['L1Div'] = round(l1div, 4)
    metrics['L1Div_GT'] = round(gt_l1div, 4)

    # LVD & MSEFace
    lvd = lvd_evaluator.avg()
    mse_face = mse_face_evaluator.avg()
    metrics['LVD'] = round(lvd, 8)
    metrics['MSEFace'] = round(mse_face, 8)

    # MSE (full-body joints)
    mse_body = mse_sum / mse_count if mse_count > 0 else 0
    metrics['MSE'] = round(mse_body, 8)

    # MPJPE
    mpjpe = (mpjpe_sum / mpjpe_frames) * 1000 if mpjpe_frames > 0 else 0
    metrics['MPJPE_mm'] = round(mpjpe, 2)

    # Per-level MPJPE
    for lvl in LEVEL_JOINTS:
        lvl_mpjpe = (mpjpe_level_sum[lvl] / mpjpe_level_frames[lvl]) * 1000 if mpjpe_level_frames[lvl] > 0 else 0
        metrics[f'MPJPE_{lvl}_mm'] = round(lvl_mpjpe, 2)

    # Mean velocity
    if per_recording:
        mean_gen_vel = np.mean([r['gen_vel'] for r in per_recording])
        mean_gt_vel = np.mean([r['gt_vel'] for r in per_recording])
        metrics['mean_vel_gen'] = round(mean_gen_vel, 6)
        metrics['mean_vel_gt'] = round(mean_gt_vel, 6)

    # Paper convention
    fgd_paper = metrics['FGD'] * 10
    bc_paper = metrics['BC'] * 10
    bc_gt_paper = metrics['BC_GT'] * 10
    print(f"  FGD:      {metrics['FGD']:.4f}  (x10 paper scale: {fgd_paper:.3f}, lower=better)")
    print(f"  BC:       {metrics['BC']:.4f}  (x10 paper scale: {bc_paper:.3f}, GT={bc_gt_paper:.3f}, higher=better)")
    print(f"  L1Div:    {metrics['L1Div']:.4f}  (GT={metrics['L1Div_GT']:.4f}, higher=better)")
    print(f"  LVD:      {metrics['LVD']:.2e}  (lower=better)")
    print(f"  MSEFace:  {metrics['MSEFace']:.2e}  (lower=better)")
    print(f"  MSE:      {metrics['MSE']:.2e}  (lower=better)")
    print(f"  MPJPE:    {metrics['MPJPE_mm']:.2f} mm  (lower=better)")
    for lvl in LEVEL_JOINTS:
        print(f"    MPJPE_{lvl}: {metrics[f'MPJPE_{lvl}_mm']:.2f} mm")
    print(f"  Velocity: gen={metrics.get('mean_vel_gen', 0):.4f}  "
          f"gt={metrics.get('mean_vel_gt', 0):.4f}")
    print(f"{'='*60}")

    # ---- Save ----
    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        ov_str = f'_ov{args.overlap}' if args.overlap > 0 else ''
        args.output = os.path.join(
            ckpt_dir,
            f'eval_recon_{ckpt_name}_f{args.test_fraction}{ov_str}.json')

    results = {
        'metrics': metrics,
        'config': {
            'mode': 'reconstruction_upper_bound',
            'checkpoint': args.checkpoint,
            'test_fraction': args.test_fraction,
            'num_recordings': n_eval,
            'chunk_length': args.chunk_length,
            'anchor_frames': args.anchor_frames,
            'overlap': args.overlap,
            'seed': args.seed,
            'note': 'z = encoder mu (deterministic), anchor from decoded output',
        },
        'per_recording': per_recording,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
