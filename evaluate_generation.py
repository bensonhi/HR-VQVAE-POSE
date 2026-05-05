"""
Evaluate motion generation on BEAT2 test set.

Uses autoregressive generation (v2 method) and EMAGE benchmark metrics:
  - FGD   (Frechet Gesture Distance) — distributional realism (lower=better)
  - BC    (Beat Consistency) — audio-motion synchronization (higher=better)
  - L1Div (Diversity) — variety of generated motions (higher=better)
  - MPJPE — mean per-joint position error in mm (lower=better)

Usage:
  conda run -n hr-vqvae-poses python evaluate_generation.py \
      --checkpoint checkpoint/beat2_poses/0/vae/best.pt

  # Evaluate 20% of test set
  conda run -n hr-vqvae-poses python evaluate_generation.py \
      --checkpoint checkpoint/beat2_poses/0/vae/best.pt --test-fraction 0.2
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
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_autoregressive_v2 import (
    load_recording, build_chunks, generate_autoregressive,
)
from m_util import conf_parser, create_model_object, get_model_type, load_checkpoint
from m_smplx_layer import SMPLXLayer
from emage_evaltools.mertic import FGD, BC, L1div, LVDFace, MSEFace


# ============================================================================
# Rotation conversions
# ============================================================================

def axis_angle_to_rot6d(axis_angle):
    """Convert axis-angle (..., 3) to rotation 6D (..., 6).

    Rodrigues formula then first two rows of rotation matrix
    (pytorch3d / EMAGE convention).
    """
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
# Test file listing
# ============================================================================

def get_test_files(data_root='BEAT2', language='english', fraction=0.1, seed=42, speaker=None, split='test'):
    """Get pose file paths for a given split, subsampled by fraction.

    Args:
        speaker: If set (e.g. 2), only include files from that speaker.
                 Standard BEAT2 protocol uses speaker=2.
        split: Which split to use: 'test' (default), 'val', or 'train'.
    """
    lang_folders = {'english': 'beat_english_v2.0.0'}
    lang_folder = lang_folders.get(language, f'beat_{language}_v2.0.0')
    lang_path = os.path.join(data_root, lang_folder)

    csv_path = os.path.join(lang_path, 'train_test_split.csv')
    test_ids = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == split:
                if speaker is not None and int(row['id'].split('_')[0]) != speaker:
                    continue
                test_ids.append(row['id'])

    rng = random.Random(seed)
    rng.shuffle(test_ids)
    n = max(1, int(len(test_ids) * fraction))
    selected = sorted(test_ids[:n])

    paths = []
    for tid in selected:
        p = os.path.join(lang_path, 'smplxflame_30', f'{tid}.npz')
        if os.path.exists(p):
            paths.append(p)

    return paths


# ============================================================================
# SMPLX joint position extraction
# ============================================================================

def poses_to_joints_and_vertices(poses_tensor, smplx_layer, device, betas=None, chunk_size=128):
    """Convert axis-angle poses (T, 165) to joints (T, 55, 3) and vertices (T, V, 3).

    Args:
        betas: Optional per-speaker shape params (300,) tensor or numpy array.
    """
    T = poses_tensor.shape[0]
    all_joints = []
    all_verts = []

    # Prepare betas for SMPLX layer
    betas_tensor = None
    if betas is not None:
        if not isinstance(betas, torch.Tensor):
            betas = torch.FloatTensor(betas)
        betas_tensor = betas.unsqueeze(0).to(device)  # (1, 300)

    with torch.no_grad():
        for i in range(0, T, chunk_size):
            batch = poses_tensor[i:i+chunk_size].unsqueeze(0).to(device)
            verts, joints = smplx_layer(batch, betas=betas_tensor)
            all_joints.append(joints[0].cpu())
            all_verts.append(verts[0].cpu())

    return torch.cat(all_joints, dim=0), torch.cat(all_verts, dim=0)


# Face vertex indices for SMPLX (head region, ~2000 vertices)
FACE_VERTEX_INDICES = list(range(0, 2000))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate motion generation on BEAT2 test set')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-fraction', type=float, default=0.1,
                        help='Fraction of test set to evaluate (default: 0.1)')
    parser.add_argument('--chunk-length', type=int, default=150)
    parser.add_argument('--anchor-frames', type=int, default=90)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--shared-z', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-name', type=str, default='beat2_poses')
    parser.add_argument('--run-num', type=int, default=0)
    parser.add_argument('--folder-name', type=str, default='vae')
    parser.add_argument('--data-root', type=str, default='BEAT2')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--speaker', type=int, default=None,
                        help='Evaluate only this speaker (default: all). Standard BEAT2 protocol uses --speaker 2')
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
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"  Checkpoint: {args.checkpoint}")

    # ---- Load SMPLX ----
    print("Loading SMPLX layer...")
    smplx_layer = SMPLXLayer(model_path='models_smplx_v1_1/models')
    smplx_layer.to(device)

    # ---- Get test files ----
    test_files = get_test_files(
        args.data_root, args.language, args.test_fraction, args.seed, speaker=args.speaker)
    print(f"\nTest files: {len(test_files)} "
          f"({args.test_fraction*100:.0f}% of test set)")

    # ---- Initialize metrics ----
    fgd_evaluator = FGD(download_path='./emage_evaltools/', device=str(device))
    bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    l1div_evaluator = L1div()

    # Face metrics
    lvd_evaluator = LVDFace()
    mse_face_evaluator = MSEFace()

    # GT reference metrics
    gt_bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    gt_l1div_evaluator = L1div()

    # MSE / MPJPE tracking
    mse_sum = 0.0
    mse_count = 0
    mpjpe_sum = 0.0
    mpjpe_frames = 0

    # Per-recording results
    per_recording = []

    # ---- Evaluate ----
    skipped = 0
    for fi, pose_path in enumerate(tqdm(test_files, desc='Evaluating')):
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

        # Generate (suppress per-chunk prints)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            gen_poses, blend_mask = generate_autoregressive(
                model, recording, chunks, device,
                anchor_k=args.anchor_frames,
                temperature=args.temperature,
                overlap=args.overlap,
                shared_z=args.shared_z,
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

        # --- Joint positions + vertices (for BC, L1Div, MPJPE, face metrics) ---
        speaker_betas = recording.get('betas')
        gen_joints_full, gen_verts = poses_to_joints_and_vertices(
            gen_poses, smplx_layer, device, betas=speaker_betas)
        gt_joints_full, gt_verts = poses_to_joints_and_vertices(
            gt_poses, smplx_layer, device, betas=speaker_betas)

        # SMPLX returns 127 joints; use first 55 (body) to match mmae stats
        gen_joints = gen_joints_full[:, :55, :]  # (T, 55, 3)
        gt_joints = gt_joints_full[:, :55, :]

        gen_joints_np = gen_joints.reshape(T, -1).numpy()  # (T, 165)
        gt_joints_np = gt_joints.reshape(T, -1).numpy()

        # --- BC ---
        trim = 60  # 2s at 30fps
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
                    # Generated BC
                    gen_motion_beats = bc_evaluator.load_motion(
                        gen_joints_np, t_start=trim, t_end=T - trim,
                        pose_fps=30, without_file=True)
                    bc_evaluator.compute(
                        audio_beats, gen_motion_beats,
                        length=T - 2*trim, pose_fps=30)

                    # GT BC (reference)
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

        # --- L1Div (on joint positions, 165D = 55*3, matches GestureLSM) ---
        l1div_evaluator.compute(gen_joints_np)  # (T, 165)
        gt_l1div_evaluator.compute(gt_joints_np)

        # --- LVD & MSEFace ---
        gen_face_verts = gen_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        gt_face_verts = gt_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        lvd_evaluator.compute(gen_face_verts, gt_face_verts)
        mse_face_evaluator.compute(gen_face_verts, gt_face_verts)

        # --- MSE (body joint positions) ---
        joint_diff = gen_joints - gt_joints  # (T, 55, 3)
        n_joints = gen_joints.shape[1]
        mse_sum += (joint_diff ** 2).sum().item()
        mse_count += T * n_joints * 3

        # --- MPJPE ---
        per_joint_err = torch.norm(joint_diff, dim=-1)  # (T, 55)
        rec_mpjpe = per_joint_err.mean().item() * 1000  # mm
        mpjpe_sum += per_joint_err.sum().item()
        mpjpe_frames += T * n_joints

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
    print(f"Results ({n_eval} recordings, "
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

    # Mean velocity
    if per_recording:
        mean_gen_vel = np.mean([r['gen_vel'] for r in per_recording])
        mean_gt_vel = np.mean([r['gt_vel'] for r in per_recording])
        metrics['mean_vel_gen'] = round(mean_gen_vel, 6)
        metrics['mean_vel_gt'] = round(mean_gt_vel, 6)

    # Paper convention: FGD×10⁻¹ and BC×10⁻¹ (multiply raw by 10)
    fgd_paper = metrics['FGD'] * 10
    bc_paper = metrics['BC'] * 10
    bc_gt_paper = metrics['BC_GT'] * 10
    print(f"  FGD:      {metrics['FGD']:.4f}  (×10 paper scale: {fgd_paper:.3f}, lower=better)")
    print(f"  BC:       {metrics['BC']:.4f}  (×10 paper scale: {bc_paper:.3f}, GT={bc_gt_paper:.3f}, higher=better)")
    print(f"  L1Div:    {metrics['L1Div']:.4f}  (GT={metrics['L1Div_GT']:.4f}, higher=better)")
    print(f"  LVD:      {metrics['LVD']:.2e}  (lower=better)")
    print(f"  MSEFace:  {metrics['MSEFace']:.2e}  (lower=better)")
    print(f"  MSE:      {metrics['MSE']:.2e}  (lower=better)")
    print(f"  MPJPE:    {metrics['MPJPE_mm']:.2f} mm  (lower=better)")
    print(f"  Velocity: gen={metrics.get('mean_vel_gen', 0):.4f}  "
          f"gt={metrics.get('mean_vel_gt', 0):.4f}")
    print(f"{'='*60}")

    # ---- Save ----
    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(
            ckpt_dir, f'eval_{ckpt_name}_f{args.test_fraction}.json')

    results = {
        'metrics': metrics,
        'config': {
            'checkpoint': args.checkpoint,
            'test_fraction': args.test_fraction,
            'num_recordings': n_eval,
            'chunk_length': args.chunk_length,
            'anchor_frames': args.anchor_frames,
            'overlap': args.overlap,
            'temperature': args.temperature,
            'seed': args.seed,
        },
        'per_recording': per_recording,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
