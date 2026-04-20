"""
Multi-GPU evaluation of diffusion prior on BEAT2 test set.

Distributes recordings across GPUs using multiprocessing. Each GPU independently
processes a shard of the test files, then results are merged for final metrics.

Usage (4x L40S):
    conda run -n hr-vqvae-poses python prior_net/evaluate_diffusion_multigpu.py --vae-checkpoint checkpoint/beat2_poses/0/vae_temporal_lean/best.pt --prior-checkpoint prior_net/checkpoints_diff_online_v2/229.pt --folder-name vae_temporal_lean --temporal --num-steps 50 --guidance-scale 0.5 --test-fraction 1.0 --num-gpus 4

    # Single GPU (equivalent to original script)
    conda run -n hr-vqvae-poses python prior_net/evaluate_diffusion_multigpu.py --prior-checkpoint prior_net/checkpoints_diff_online_v2/229.pt --temporal --guidance-scale 0.5 --num-gpus 1
"""
import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import torch.multiprocessing as mp

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

FACE_VERTEX_INDICES = list(range(0, 2000))


def load_models(args, device):
    """Load VAE, diffusion prior, and SMPLX on the given device."""
    options, _ = conf_parser(args.dataset_name, args.run_num, args.folder_name)
    model_type = get_model_type(args.folder_name)
    vae_model = create_model_object(model_type, options)
    state_dict = load_checkpoint(args.vae_checkpoint, device)
    # Filter out SMPLX shape params that may have different sizes across checkpoints
    model_state = vae_model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    skipped_keys = [k for k in state_dict if k not in filtered]
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} keys with shape mismatch: {skipped_keys[:5]}...")
    vae_model.load_state_dict(filtered, strict=False)
    vae_model.to(device).eval()

    prior_ckpt = torch.load(args.prior_checkpoint, map_location=device)
    prior_args = prior_ckpt.get('args', {})
    is_temporal = args.temporal or prior_ckpt.get('is_temporal', False)

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
            temporal_downsample=prior_args.get('temporal_downsample', 8),
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

    weight_key = 'ema_model' if args.use_ema and 'ema_model' in prior_ckpt else 'model'
    diff_model.load_state_dict(prior_ckpt[weight_key])
    if 'z_mean' in prior_ckpt:
        diff_model.z_mean.copy_(prior_ckpt['z_mean'].to(device))
        diff_model.z_std.copy_(prior_ckpt['z_std'].to(device))
    diff_model.eval()

    smplx_layer = SMPLXLayer(model_path='models_smplx_v1_1/models')
    smplx_layer.to(device)

    return vae_model, diff_model, smplx_layer, is_temporal


def evaluate_shard(gpu_id, args, file_shard, result_dir):
    """Evaluate a shard of test files on a single GPU. Saves intermediate results to disk.

    Face metrics (LVD, MSEFace) are computed on-the-fly per recording to avoid
    storing large face vertex arrays (~6GB per shard) in memory.
    """
    device = torch.device(f'cuda:{gpu_id}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + gpu_id)
    random.seed(args.seed + gpu_id)
    torch.cuda.manual_seed(args.seed + gpu_id)

    vae_model, diff_model, smplx_layer, is_temporal = load_models(args, device)

    # Face metrics computed on-the-fly (not stored in shard)
    lvd_evaluator = LVDFace()
    mse_face_evaluator = MSEFace()

    shard_data = {
        'gen_r6d_list': [],
        'gt_r6d_list': [],
        'gen_joints_list': [],
        'gt_joints_list': [],
        'gen_poses_list': [],
        'gt_poses_list': [],
        'bc_data': [],
        'per_recording': [],
        'skipped': 0,
        'mpjpe_sum': 0.0,
        'mpjpe_frames': 0,
        'lvd_sum': 0.0,
        'lvd_count': 0,
        'mse_face_sum': 0.0,
        'mse_face_count': 0,
    }

    desc = f'GPU {gpu_id}'
    for fi, pose_path in enumerate(tqdm(file_shard, desc=desc, position=gpu_id)):
        basename = os.path.splitext(os.path.basename(pose_path))[0]

        try:
            recording = load_recording(pose_path, args.language, args.data_root)
        except Exception as e:
            tqdm.write(f"  GPU{gpu_id} Skip {basename}: {e}")
            shard_data['skipped'] += 1
            continue

        chunks = build_chunks(recording, max_chunk_length=args.chunk_length)
        if not chunks:
            shard_data['skipped'] += 1
            continue

        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            gen_poses, blend_mask = generate_autoregressive_with_diffusion(
                vae_model, diff_model, recording, chunks, device,
                anchor_k=args.anchor_frames, overlap=args.overlap,
                num_steps=args.num_steps, guidance_scale=args.guidance_scale,
            )

        T_gen = gen_poses.shape[0]
        start_frame = chunks[0]['start']
        gt_poses = torch.FloatTensor(recording['poses'][start_frame:start_frame + T_gen])
        T = min(T_gen, gt_poses.shape[0])
        gen_poses = gen_poses[:T]
        gt_poses = gt_poses[:T]

        if T < 64:
            shard_data['skipped'] += 1
            continue

        # FGD features (rot6d)
        gen_r6d = axis_angle_to_rot6d(gen_poses.reshape(T, 55, 3)).reshape(1, T, 330).float()
        gt_r6d = axis_angle_to_rot6d(gt_poses.reshape(T, 55, 3)).reshape(1, T, 330).float()
        shard_data['gen_r6d_list'].append(gen_r6d.cpu())
        shard_data['gt_r6d_list'].append(gt_r6d.cpu())

        # Joints + vertices
        speaker_betas = recording.get('betas')
        gen_joints_full, gen_verts = poses_to_joints_and_vertices(gen_poses, smplx_layer, device, betas=speaker_betas)
        gt_joints_full, gt_verts = poses_to_joints_and_vertices(gt_poses, smplx_layer, device, betas=speaker_betas)
        gen_joints = gen_joints_full[:, :55, :]
        gt_joints = gt_joints_full[:, :55, :]

        # MPJPE
        per_joint_err = torch.norm(gen_joints - gt_joints, dim=-1)
        rec_mpjpe = per_joint_err.mean().item() * 1000
        shard_data['mpjpe_sum'] += per_joint_err.sum().item()
        shard_data['mpjpe_frames'] += T * 55

        # Velocity
        gen_vel = torch.norm(gen_joints[1:] - gen_joints[:-1], dim=-1).mean().item()
        gt_vel = torch.norm(gt_joints[1:] - gt_joints[:-1], dim=-1).mean().item()

        # Face metrics — computed on-the-fly, NOT stored in shard
        gen_face_verts = gen_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        gt_face_verts = gt_verts[:, FACE_VERTEX_INDICES, :].reshape(T, -1).numpy()
        lvd_evaluator.compute(gen_face_verts, gt_face_verts)
        mse_face_evaluator.compute(gen_face_verts, gt_face_verts)
        del gen_face_verts, gt_face_verts, gen_verts, gt_verts  # free memory immediately

        # Save lightweight data for BC, L1div (computed in main process)
        gen_joints_np = gen_joints.reshape(T, -1).numpy()
        gt_joints_np = gt_joints.reshape(T, -1).numpy()

        shard_data['gen_joints_list'].append(gen_joints_np)
        shard_data['gt_joints_list'].append(gt_joints_np)
        shard_data['gen_poses_list'].append(gen_poses.numpy())
        shard_data['gt_poses_list'].append(gt_poses.numpy())

        wav_path = recording.get('wav_path')
        shard_data['bc_data'].append({
            'wav_path': wav_path,
            'start_frame': start_frame,
            'T': T,
            'basename': basename,
        })

        shard_data['per_recording'].append({
            'basename': basename, 'frames': T, 'mpjpe_mm': round(rec_mpjpe, 2),
            'gen_vel': round(gen_vel, 6), 'gt_vel': round(gt_vel, 6),
        })

    # Store face metric accumulators (just two scalars, not arrays)
    shard_data['lvd_sum'] = float(lvd_evaluator.sum) if hasattr(lvd_evaluator, 'sum') else 0.0
    shard_data['lvd_count'] = int(lvd_evaluator.counter) if hasattr(lvd_evaluator, 'counter') else 0
    shard_data['mse_face_sum'] = float(mse_face_evaluator.sum) if hasattr(mse_face_evaluator, 'sum') else 0.0
    shard_data['mse_face_count'] = int(mse_face_evaluator.counter) if hasattr(mse_face_evaluator, 'counter') else 0

    # Save shard results to disk (atomic: write tmp then rename to prevent corruption)
    out_path = os.path.join(result_dir, f'shard_{gpu_id}.pt')
    tmp_path = out_path + '.tmp'
    torch.save(shard_data, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"GPU {gpu_id}: done, {len(file_shard) - shard_data['skipped']} recordings processed")


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU diffusion prior evaluation')
    parser.add_argument('--vae-checkpoint', type=str, default='checkpoint/beat2_poses/0/vae_temporal_lean/best.pt')
    parser.add_argument('--prior-checkpoint', type=str, default='prior_net/checkpoints_diff_temporal_lean/best.pt')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--use-ema', action='store_true')
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument('--chunk-length', type=int, default=150)
    parser.add_argument('--anchor-frames', type=int, default=90)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-name', type=str, default='beat2_poses')
    parser.add_argument('--run-num', type=int, default=0)
    parser.add_argument('--folder-name', type=str, default='vae_temporal_lean')
    parser.add_argument('--data-root', type=str, default='BEAT2')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--speaker', type=int, default=None,
                        help='Evaluate only this speaker (default: all). Standard BEAT2 protocol uses --speaker 2')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--num-gpus', type=int, default=4)
    args = parser.parse_args()

    # Clamp to available GPUs
    available_gpus = torch.cuda.device_count()
    args.num_gpus = min(args.num_gpus, available_gpus)
    print(f"Using {args.num_gpus} GPU(s) (available: {available_gpus})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    test_files = get_test_files(args.data_root, args.language, args.test_fraction, args.seed, speaker=args.speaker)
    print(f"Test files: {len(test_files)}")

    # Split files across GPUs (round-robin for balanced load)
    shards = [[] for _ in range(args.num_gpus)]
    for i, f in enumerate(test_files):
        shards[i % args.num_gpus].append(f)
    for i, s in enumerate(shards):
        print(f"  GPU {i}: {len(s)} files")

    # Temporary directory for shard results
    result_dir = tempfile.mkdtemp(prefix='eval_diff_')
    print(f"Shard results in: {result_dir}")

    # Launch workers
    if args.num_gpus == 1:
        evaluate_shard(0, args, shards[0], result_dir)
    else:
        mp.set_start_method('spawn', force=True)
        processes = []
        for gpu_id in range(args.num_gpus):
            p = mp.Process(target=evaluate_shard, args=(gpu_id, args, shards[gpu_id], result_dir))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for gpu_id, p in enumerate(processes):
            if p.exitcode != 0:
                print(f"WARNING: GPU {gpu_id} worker exited with code {p.exitcode}")

    # ---- Merge shard results ----
    print("\nMerging results...")
    all_gen_r6d = []
    all_gt_r6d = []
    all_per_recording = []
    mpjpe_sum = 0.0
    mpjpe_frames = 0
    total_skipped = 0

    bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    gt_bc_evaluator = BC(download_path='./emage_evaltools/', sigma=0.3, order=7)
    l1div_evaluator = L1div()
    gt_l1div_evaluator = L1div()
    lvd_sum = 0.0
    lvd_count = 0
    mse_face_sum = 0.0
    mse_face_count = 0

    device = torch.device('cuda:0')
    fgd_evaluator = FGD(download_path='./emage_evaltools/', device=str(device))

    failed_shards = []
    for gpu_id in range(args.num_gpus):
        shard_path = os.path.join(result_dir, f'shard_{gpu_id}.pt')
        try:
            shard = torch.load(shard_path, map_location='cpu')
        except Exception as e:
            print(f"WARNING: shard {gpu_id} failed to load ({e}), skipping")
            failed_shards.append(gpu_id)
            continue

        # FGD
        for gen_r6d, gt_r6d in zip(shard['gen_r6d_list'], shard['gt_r6d_list']):
            fgd_evaluator.update(gen_r6d.to(device), gt_r6d.to(device))

        # MPJPE
        mpjpe_sum += shard['mpjpe_sum']
        mpjpe_frames += shard['mpjpe_frames']
        total_skipped += shard['skipped']
        all_per_recording.extend(shard['per_recording'])

        # Face metrics (pre-computed scalars from shard)
        lvd_sum += shard.get('lvd_sum', 0.0)
        lvd_count += shard.get('lvd_count', 0)
        mse_face_sum += shard.get('mse_face_sum', 0.0)
        mse_face_count += shard.get('mse_face_count', 0)

        # BC, L1div
        trim = 60
        for idx, bc_info in enumerate(shard['bc_data']):
            wav_path = bc_info['wav_path']
            start_frame = bc_info['start_frame']
            T = bc_info['T']

            gen_joints_np = shard['gen_joints_list'][idx]
            gt_joints_np = shard['gt_joints_list'][idx]

            if wav_path and os.path.exists(wav_path) and T > trim * 2 + 30:
                audio_start_sec = (start_frame + trim) / 30.0
                audio_end_sec = (start_frame + T - trim) / 30.0
                t_start_audio = int(audio_start_sec * 16000)
                t_end_audio = int(audio_end_sec * 16000)
                try:
                    audio_beats = bc_evaluator.load_audio(wav_path, t_start=t_start_audio, t_end=t_end_audio)
                    if len(audio_beats) > 0:
                        gen_motion_beats = bc_evaluator.load_motion(gen_joints_np, t_start=trim, t_end=T - trim, pose_fps=30, without_file=True)
                        bc_evaluator.compute(audio_beats, gen_motion_beats, length=T - 2*trim, pose_fps=30)
                        gt_audio_beats = gt_bc_evaluator.load_audio(wav_path, t_start=t_start_audio, t_end=t_end_audio)
                        gt_motion_beats = gt_bc_evaluator.load_motion(gt_joints_np, t_start=trim, t_end=T - trim, pose_fps=30, without_file=True)
                        gt_bc_evaluator.compute(gt_audio_beats, gt_motion_beats, length=T - 2*trim, pose_fps=30)
                except Exception as e:
                    print(f"  BC error {bc_info['basename']}: {e}")

            l1div_evaluator.compute(gen_joints_np)
            gt_l1div_evaluator.compute(gt_joints_np)

    # ---- Final metrics ----
    n_eval = len(test_files) - total_skipped
    if failed_shards:
        n_failed_files = sum(len(shards[i]) for i in failed_shards if i < len(shards))
        total_skipped += n_failed_files
        n_eval -= n_failed_files
    print(f"\n{'='*60}")
    print(f"DIFFUSION PRIOR Results ({n_eval} recordings, {args.num_gpus} GPUs)")
    if failed_shards:
        print(f"  WARNING: shards {failed_shards} failed, results from {args.num_gpus - len(failed_shards)} GPUs only")
    print(f"  num_steps={args.num_steps}, guidance_scale={args.guidance_scale}")
    print(f"{'='*60}")

    if n_eval == 0:
        print("ERROR: No recordings were successfully evaluated. Check GPU memory and shard files.")
        sys.exit(1)

    metrics = {}
    fgd = fgd_evaluator.compute()
    metrics['FGD'] = round(float(fgd), 4)
    bc = bc_evaluator.avg() if bc_evaluator.counter > 0 else float('nan')
    gt_bc = gt_bc_evaluator.avg() if gt_bc_evaluator.counter > 0 else float('nan')
    metrics['BC'] = round(float(bc), 4)
    metrics['BC_GT'] = round(float(gt_bc), 4)
    l1div = l1div_evaluator.avg()
    gt_l1div = gt_l1div_evaluator.avg()
    metrics['L1Div'] = round(float(l1div), 4)
    metrics['L1Div_GT'] = round(float(gt_l1div), 4)
    lvd = lvd_sum / lvd_count if lvd_count > 0 else 0.0
    mse_face = mse_face_sum / mse_face_count if mse_face_count > 0 else 0.0
    metrics['LVD'] = round(float(lvd), 8)
    metrics['MSEFace'] = round(float(mse_face), 8)
    mpjpe = (mpjpe_sum / mpjpe_frames) * 1000 if mpjpe_frames > 0 else 0
    metrics['MPJPE_mm'] = round(mpjpe, 2)

    if all_per_recording:
        mean_gen_vel = np.mean([r['gen_vel'] for r in all_per_recording])
        mean_gt_vel = np.mean([r['gt_vel'] for r in all_per_recording])
        metrics['mean_vel_gen'] = round(float(mean_gen_vel), 6)
        metrics['mean_vel_gt'] = round(float(mean_gt_vel), 6)

    fgd_paper = metrics['FGD'] * 10
    bc_paper = metrics['BC'] * 10
    bc_gt_paper = metrics['BC_GT'] * 10
    print(f"  FGD:      {metrics['FGD']:.4f}  (paper: {fgd_paper:.3f})")
    print(f"  BC:       {metrics['BC']:.4f}  (paper: {bc_paper:.3f}, GT={bc_gt_paper:.3f})")
    print(f"  L1Div:    {metrics['L1Div']:.4f}  (GT={metrics['L1Div_GT']:.4f})")
    print(f"  LVD:      {metrics['LVD']:.2e}")
    print(f"  MSEFace:  {metrics['MSEFace']:.2e}")
    print(f"  MPJPE:    {metrics['MPJPE_mm']:.2f} mm")
    print(f"  Velocity: gen={metrics.get('mean_vel_gen', 0):.4f}  gt={metrics.get('mean_vel_gt', 0):.4f}")

    # Save
    if args.output is None:
        ckpt_dir = os.path.dirname(args.prior_checkpoint)
        ckpt_base = os.path.splitext(os.path.basename(args.prior_checkpoint))[0]
        tag = f"g{args.guidance_scale}_s{args.num_steps}_f{args.test_fraction}"
        args.output = os.path.join(ckpt_dir, f"eval_{ckpt_base}_{tag}.json")

    result = {
        'metrics': metrics,
        'fgd_paper': fgd_paper,
        'bc_paper': bc_paper,
        'config': {
            'prior_checkpoint': args.prior_checkpoint,
            'vae_checkpoint': args.vae_checkpoint,
            'num_steps': args.num_steps,
            'guidance_scale': args.guidance_scale,
            'test_fraction': args.test_fraction,
            'num_recordings': n_eval,
            'num_gpus': args.num_gpus,
            'seed': args.seed,
        },
        'per_recording': sorted(all_per_recording, key=lambda x: x['basename']),
    }

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Cleanup
    import shutil
    shutil.rmtree(result_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
