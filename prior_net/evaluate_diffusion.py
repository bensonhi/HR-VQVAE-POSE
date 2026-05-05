"""
Evaluate motion generation using the diffusion prior on BEAT2 test set.

Drop-in replacement for evaluate.py but uses FlowMatchingPrior for z.
Supports both single-z and temporal diffusion priors.

Usage (temporal lean):
    conda run -n hr-vqvae-poses python prior_net/evaluate_diffusion.py \
        --vae-checkpoint checkpoint/beat2_poses/0/vae_temporal_lean/best.pt \
        --prior-checkpoint prior_net/checkpoints_diff_temporal_lean/best.pt \
        --folder-name vae_temporal_lean --temporal
"""
import argparse
import contextlib
import io
import json
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_autoregressive_v2 import load_recording, build_chunks
from prior_net.generate_diffusion import generate_autoregressive_with_diffusion
from prior_net.diffusion_model import FlowMatchingPrior, TemporalFlowMatchingPrior
from m_util import conf_parser, create_model_object, get_model_type, load_checkpoint
from m_smplx_layer import SMPLXLayer
from evaluate_generation import (
    get_test_files, axis_angle_to_rot6d, poses_to_joints_and_vertices,
)
from emage_evaltools.mertic import FGD, BC, L1div, LVDFace, MSEFace


def _get_velocity_scale(args):
    """Return velocity scale: scalar or per-joint array (165,)."""
    if args.per_joint_vel_scale:
        import numpy as np
        # Conservative per-body-part scaling (50% of measured ratio)
        scale = np.ones(165, dtype=np.float32)
        for j in range(0, 1): scale[j*3:(j+1)*3] = 0.93   # global (slight reduce)
        for j in range(1, 4): scale[j*3:(j+1)*3] = 1.12   # spine
        for j in range(12, 15): scale[j*3:(j+1)*3] = 1.17  # head
        for j in range(16, 21): scale[j*3:(j+1)*3] = 1.08  # left arm
        for j in range(21, 25): scale[j*3:(j+1)*3] = 1.25  # right arm
        for j in range(25, 40): scale[j*3:(j+1)*3] = 1.20  # left hand
        for j in range(40, 55): scale[j*3:(j+1)*3] = 1.25  # right hand
        return scale
    return args.velocity_scale


def main():
    parser = argparse.ArgumentParser(description='Evaluate diffusion prior generation')
    parser.add_argument('--vae-checkpoint', type=str,
                        default='checkpoint/beat2_poses/0/vae_temporal_lean/best.pt')
    parser.add_argument('--prior-checkpoint', type=str,
                        default='prior_net/checkpoints_diff_temporal_lean/best.pt')
    parser.add_argument('--temporal', action='store_true',
                        help='Use temporal flow matching prior')
    parser.add_argument('--use-ema', action='store_true',
                        help='Load EMA model weights (from online training)')
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--chunk-length', type=int, default=150)
    parser.add_argument('--anchor-frames', type=int, default=90)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-name', type=str, default='beat2_poses')
    parser.add_argument('--run-num', type=int, default=0)
    parser.add_argument('--folder-name', type=str, default='vae_temporal_lean')
    parser.add_argument('--data-root', type=str, default='BEAT2')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--speaker', type=int, default=None,
                        help='Evaluate only this speaker (default: all). Standard BEAT2 protocol uses --speaker 2')
    parser.add_argument('--smooth-sigma', type=float, default=0.0,
                        help='Global temporal smoothing sigma (0=off, try 0.5-2.0)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Initial noise temperature (lower = less diverse, potentially better FGD)')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of z samples to average per chunk (reduces variance)')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='Truncation factor for z (< 1 reduces diversity, may improve FGD)')
    parser.add_argument('--refine-strength', type=float, default=0.0,
                        help='SDEdit refinement: add noise at this t and re-denoise (0=off, try 0.7-0.9)')
    parser.add_argument('--euler', action='store_true', help='Use Euler ODE instead of midpoint')
    parser.add_argument('--velocity-scale', type=float, default=1.0,
                        help='Scale motion velocity (>1 = more dynamic, try 1.1-1.5)')
    parser.add_argument('--per-joint-vel-scale', action='store_true',
                        help='Use per-body-part velocity scaling based on measured GT/gen ratios')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--text-dir', type=str, default=None, help='Directory with pre-computed text embeddings (.npy). Auto-detected from checkpoint args if not set.')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'],
                        help='Which split to evaluate on (default: test). Use val for guidance sweep.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load VAE
    print("Loading VAE...")
    options, _ = conf_parser(args.dataset_name, args.run_num, args.folder_name)
    model_type = get_model_type(args.folder_name)
    vae_model = create_model_object(model_type, options)
    state_dict = load_checkpoint(args.vae_checkpoint, device)
    model_sd = vae_model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    vae_model.load_state_dict(filtered, strict=False)
    vae_model.to(device).eval()

    # Load diffusion prior
    print("Loading diffusion prior...")
    prior_ckpt = torch.load(args.prior_checkpoint, map_location=device)

    # Override VAE state if jointly trained
    if 'vae_state' in prior_ckpt:
        print("  Loading jointly-trained VAE decoder state...")
        vae_model.load_state_dict(prior_ckpt['vae_state'])
        vae_model.eval()
    prior_args = prior_ckpt.get('args', {})
    is_temporal = args.temporal or prior_ckpt.get('is_temporal', False)

    text_dim = prior_args.get('text_dim', 0)
    if is_temporal:
        diff_model = TemporalFlowMatchingPrior(
            latent_dim=prior_args.get('latent_dim', 16),
            audio_dim=prior_args.get('audio_dim', 768),
            d_model=prior_args.get('d_model', 256),
            nhead=prior_args.get('nhead', 4),
            num_audio_layers=prior_args.get('num_audio_layers', 4),
            num_denoiser_layers=prior_args.get('num_denoiser_layers', 6),
            dim_feedforward=prior_args.get('dim_feedforward', 1024),
            dropout=0.0,
            num_speakers=prior_args.get('num_speakers', 31),
            temporal_downsample=prior_args.get('temporal_downsample', getattr(vae_model, 'temporal_downsample', 8)),
            text_dim=text_dim,
        ).to(device)
    else:
        diff_model = FlowMatchingPrior(
            latent_dim=prior_args.get('latent_dim', 128),
            audio_dim=prior_args.get('audio_dim', 768),
            d_model=prior_args.get('d_model', 256),
            nhead=prior_args.get('nhead', 4),
            num_audio_layers=prior_args.get('num_audio_layers', 4),
            dim_feedforward=prior_args.get('dim_feedforward', 1024),
            dropout=0.0,
            num_speakers=prior_args.get('num_speakers', 31),
            hidden_dim=prior_args.get('hidden_dim', 512),
            num_mlp_blocks=prior_args.get('num_mlp_blocks', 6),
        ).to(device)

    # Load weights: EMA if available and requested, otherwise regular model
    weight_key = 'ema_model' if args.use_ema and 'ema_model' in prior_ckpt else 'model'
    diff_model.load_state_dict(prior_ckpt[weight_key])
    if 'z_mean' in prior_ckpt:
        diff_model.z_mean.copy_(prior_ckpt['z_mean'].to(device))
        diff_model.z_std.copy_(prior_ckpt['z_std'].to(device))
    diff_model.eval()
    print(f"  Diffusion prior epoch: {prior_ckpt.get('epoch', '?')}, "
          f"temporal={is_temporal}, weights={weight_key}")

    # SMPLX
    print("Loading SMPLX...")
    smplx_layer = SMPLXLayer(model_path='models_smplx_v1_1/models')
    smplx_layer.to(device)

    # Test files
    test_files = get_test_files(args.data_root, args.language, args.test_fraction, args.seed, speaker=args.speaker, split=args.split)
    print(f"Test files: {len(test_files)}")

    # Metrics
    fgd_evaluator = FGD(download_path='./emage_evaltools/', device=str(device))
    bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    l1div_evaluator = L1div()
    lvd_evaluator = LVDFace()
    mse_face_evaluator = MSEFace()
    gt_bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    gt_l1div_evaluator = L1div()

    mpjpe_sum = 0.0
    mpjpe_frames = 0
    per_recording = []
    skipped = 0

    FACE_VERTEX_INDICES = list(range(0, 2000))

    # Determine text directory for text-conditioned models
    text_dir = args.text_dir
    if text_dir is None and text_dim > 0:
        lang_folder = {'english': 'beat_english_v2.0.0'}.get(args.language, f'beat_{args.language}_v2.0.0')
        text_dir = os.path.join(args.data_root, lang_folder, 'bert_30')
    if text_dim > 0:
        print(f"  Text conditioning enabled (dim={text_dim}), text_dir={text_dir}")

    for fi, pose_path in enumerate(tqdm(test_files, desc='Evaluating')):
        basename = os.path.splitext(os.path.basename(pose_path))[0]

        try:
            recording = load_recording(pose_path, args.language, args.data_root)
        except Exception as e:
            tqdm.write(f"  Skip {basename}: {e}")
            skipped += 1
            continue

        # Load text features if model uses text conditioning
        text_features_np = None
        if text_dim > 0 and text_dir is not None:
            text_path = os.path.join(text_dir, f'{basename}.npy')
            if os.path.exists(text_path):
                text_features_np = np.load(text_path)
                # Trim to match audio length
                audio_len = recording['audio'].shape[0]
                text_features_np = text_features_np[:audio_len]

        chunks = build_chunks(recording, max_chunk_length=args.chunk_length)
        if not chunks:
            skipped += 1
            continue

        # Generate with diffusion prior
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            gen_poses, blend_mask = generate_autoregressive_with_diffusion(
                vae_model, diff_model, recording, chunks, device,
                anchor_k=args.anchor_frames,
                overlap=args.overlap,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                smooth_sigma=args.smooth_sigma,
                temperature=args.temperature,
                num_samples=args.num_samples,
                truncation=args.truncation,
                refine_strength=args.refine_strength,
                use_euler=args.euler,
                velocity_scale=_get_velocity_scale(args),
                text_features_np=text_features_np,
            )

        T_gen = gen_poses.shape[0]
        start_frame = chunks[0]['start']
        gt_poses = torch.FloatTensor(recording['poses'][start_frame:start_frame + T_gen])
        T = min(T_gen, gt_poses.shape[0])
        gen_poses = gen_poses[:T]
        gt_poses = gt_poses[:T]

        if T < 64:
            skipped += 1
            continue

        # FGD
        gen_r6d = axis_angle_to_rot6d(gen_poses.reshape(T, 55, 3)).reshape(1, T, 330).to(device)
        gt_r6d = axis_angle_to_rot6d(gt_poses.reshape(T, 55, 3)).reshape(1, T, 330).to(device)
        fgd_evaluator.update(gen_r6d.float(), gt_r6d.float())

        # Joints
        speaker_betas = recording.get('betas')
        gen_joints_full, gen_verts = poses_to_joints_and_vertices(gen_poses, smplx_layer, device, betas=speaker_betas)
        gt_joints_full, gt_verts = poses_to_joints_and_vertices(gt_poses, smplx_layer, device, betas=speaker_betas)
        gen_joints = gen_joints_full[:, :55, :]
        gt_joints = gt_joints_full[:, :55, :]
        gen_joints_np = gen_joints.reshape(T, -1).numpy()
        gt_joints_np = gt_joints.reshape(T, -1).numpy()

        # BC
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
                if len(audio_beats) > 0:
                    gen_motion_beats = bc_evaluator.load_motion(
                        gen_joints_np, t_start=trim, t_end=T - trim,
                        pose_fps=30, without_file=True)
                    bc_evaluator.compute(audio_beats, gen_motion_beats,
                                        length=T - 2*trim, pose_fps=30)
                    gt_audio_beats = gt_bc_evaluator.load_audio(
                        wav_path, t_start=t_start_audio, t_end=t_end_audio)
                    gt_motion_beats = gt_bc_evaluator.load_motion(
                        gt_joints_np, t_start=trim, t_end=T - trim,
                        pose_fps=30, without_file=True)
                    gt_bc_evaluator.compute(gt_audio_beats, gt_motion_beats,
                                           length=T - 2*trim, pose_fps=30)
            except Exception as e:
                tqdm.write(f"  BC error {basename}: {e}")

        # L1Div (on joint positions, matches GestureLSM)
        l1div_evaluator.compute(gen_joints_np)
        gt_l1div_evaluator.compute(gt_joints_np)

        # Face metrics
        gen_face_verts = gen_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        gt_face_verts = gt_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        lvd_evaluator.compute(gen_face_verts, gt_face_verts)
        mse_face_evaluator.compute(gen_face_verts, gt_face_verts)

        # MPJPE
        per_joint_err = torch.norm(gen_joints - gt_joints, dim=-1)
        rec_mpjpe = per_joint_err.mean().item() * 1000
        mpjpe_sum += per_joint_err.sum().item()
        mpjpe_frames += T * 55

        # Velocity
        gen_vel = torch.norm(gen_joints[1:] - gen_joints[:-1], dim=-1).mean().item()
        gt_vel = torch.norm(gt_joints[1:] - gt_joints[:-1], dim=-1).mean().item()

        per_recording.append({
            'basename': basename, 'frames': T, 'mpjpe_mm': round(rec_mpjpe, 2),
            'gen_vel': round(gen_vel, 6), 'gt_vel': round(gt_vel, 6),
        })
        tqdm.write(f"  {basename}: {T}fr, MPJPE={rec_mpjpe:.1f}mm, "
                   f"vel gen={gen_vel:.4f} gt={gt_vel:.4f}")

    # Final metrics
    n_eval = len(test_files) - skipped
    print(f"\n{'='*60}")
    print(f"DIFFUSION PRIOR Results ({n_eval} recordings, temporal={is_temporal})")
    print(f"  num_steps={args.num_steps}, guidance_scale={args.guidance_scale}")
    print(f"{'='*60}")

    metrics = {}
    fgd = fgd_evaluator.compute()
    metrics['FGD'] = round(fgd, 4)
    bc = bc_evaluator.avg() if bc_evaluator.counter > 0 else float('nan')
    gt_bc = gt_bc_evaluator.avg() if gt_bc_evaluator.counter > 0 else float('nan')
    metrics['BC'] = round(bc, 4)
    metrics['BC_GT'] = round(gt_bc, 4)
    l1div = l1div_evaluator.avg()
    gt_l1div = gt_l1div_evaluator.avg()
    metrics['L1Div'] = round(l1div, 4)
    metrics['L1Div_GT'] = round(gt_l1div, 4)
    lvd = lvd_evaluator.avg()
    mse_face = mse_face_evaluator.avg()
    metrics['LVD'] = round(lvd, 8)
    metrics['MSEFace'] = round(mse_face, 8)
    mpjpe = (mpjpe_sum / mpjpe_frames) * 1000 if mpjpe_frames > 0 else 0
    metrics['MPJPE_mm'] = round(mpjpe, 2)

    if per_recording:
        mean_gen_vel = np.mean([r['gen_vel'] for r in per_recording])
        mean_gt_vel = np.mean([r['gt_vel'] for r in per_recording])
        metrics['mean_vel_gen'] = round(mean_gen_vel, 6)
        metrics['mean_vel_gt'] = round(mean_gt_vel, 6)

    fgd_paper = metrics['FGD'] * 10
    bc_paper = metrics['BC'] * 10
    bc_gt_paper = metrics['BC_GT'] * 10
    print(f"  FGD:      {metrics['FGD']:.4f}  (paper: {fgd_paper:.3f})")
    print(f"  BC:       {metrics['BC']:.4f}  (paper: {bc_paper:.3f}, GT={bc_gt_paper:.3f})")
    print(f"  L1Div:    {metrics['L1Div']:.4f}  (GT={metrics['L1Div_GT']:.4f})")
    print(f"  LVD:      {metrics['LVD']:.2e}")
    print(f"  MSEFace:  {metrics['MSEFace']:.2e}")
    print(f"  MPJPE:    {metrics['MPJPE_mm']:.2f} mm")
    print(f"  Velocity: gen={metrics.get('mean_vel_gen', 0):.4f}  "
          f"gt={metrics.get('mean_vel_gt', 0):.4f}")

    # Save
    if args.output is None:
        prior_dir = os.path.dirname(args.prior_checkpoint)
        prior_name = os.path.splitext(os.path.basename(args.prior_checkpoint))[0]
        args.output = os.path.join(
            prior_dir, f'eval_diff_{prior_name}_s{args.num_steps}_g{args.guidance_scale}.json')

    results = {
        'metrics': metrics,
        'config': {
            'vae_checkpoint': args.vae_checkpoint,
            'prior_checkpoint': args.prior_checkpoint,
            'test_fraction': args.test_fraction,
            'num_recordings': n_eval,
            'num_steps': args.num_steps,
            'guidance_scale': args.guidance_scale,
            'temporal': is_temporal,
        },
        'per_recording': per_recording,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
