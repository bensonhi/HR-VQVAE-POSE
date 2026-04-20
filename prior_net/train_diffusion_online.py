"""
Train the temporal flow-matching diffusion prior ONLINE (with anchor conditioning).

Instead of pre-extracted latents, uses the BEAT2 dataloader directly.
A frozen VAE encoder produces z targets on-the-fly, and the diffusion prior
learns to match them conditioned on (audio, speaker, gesture, anchor_frames).

This fixes the train/test mismatch: the prior sees anchor frames during training,
matching how it's used during autoregressive generation.

Usage:
    # Single GPU
    conda activate hr-vqvae-poses && python prior_net/train_diffusion_online.py

    # Multi-GPU (e.g. 3 GPUs)
    torchrun --nproc_per_node=3 prior_net/train_diffusion_online.py

    # Resume
    conda activate hr-vqvae-poses && python prior_net/train_diffusion_online.py --resume prior_net/checkpoints_diff_online/best.pt
"""
import argparse
import contextlib
import copy
import csv
import io
import math
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m_beat_dataset import BEAT2PoseDataset, variable_length_collate_fn
from m_util import conf_parser, create_model_object, get_model_type, load_checkpoint
from prior_net.diffusion_model import TemporalFlowMatchingPrior


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


def train_epoch(diff_model, vae_model, loader, optimizer, device, grad_clip=1.0,
                cond_drop_prob=0.1, anchor_drop_prob=0.3):
    """Train one epoch with online encoding."""
    diff_model.train()
    # Unwrap DDP for method calls (gradient sync still happens via parameter hooks)
    underlying = diff_model.module if hasattr(diff_model, 'module') else diff_model
    total_loss = 0
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

        # Randomly drop anchor conditioning (so model works without anchors too)
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

        loss = underlying.compute_loss(
            mu, audio, speaker_id, gesture_type, padding_mask,
            z_padding_mask=z_padding_mask,
            cond_drop_prob=cond_drop_prob,
            anchor_frames=anchor_pool_input,
            anchor_audio=anchor_audio_input,
            text_features=text_features,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff_model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

    return total_loss / total_samples


@torch.no_grad()
def validate_epoch(diff_model, vae_model, loader, device, num_steps=20):
    """Validate: flow matching loss + sample quality."""
    diff_model.eval()
    total_loss = 0
    total_mse = 0
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

        loss = diff_model.compute_loss(
            mu, audio, speaker_id, gesture_type, padding_mask,
            z_padding_mask=z_padding_mask,
            cond_drop_prob=0.0,
            anchor_frames=anchor_pool,
            anchor_audio=anchor_audio,
            text_features=text_features,
        )

        # Sample and compare to target z
        z_sampled = diff_model.sample(
            audio, speaker_id, gesture_type, padding_mask,
            num_steps=num_steps, guidance_scale=1.0,
            anchor_frames=anchor_pool, anchor_audio=anchor_audio,
            z_seq_len=mu.shape[1],
            text_features=text_features,
        )

        if z_padding_mask is not None:
            valid = (~z_padding_mask).unsqueeze(-1).float()
            mse = ((z_sampled - mu) ** 2 * valid).sum() / valid.sum() / mu.shape[-1]
        else:
            mse = F.mse_loss(z_sampled, mu)

        total_loss += loss.item() * B
        total_mse += mse.item() * B
        total_samples += B

    return {
        'loss': total_loss / total_samples,
        'mse': total_mse / total_samples,
    }


@torch.no_grad()
def validate_fgd_autoregressive(diff_model, vae_model, device,
                                data_root='BEAT2', language='english',
                                test_fraction=0.1, seed=42,
                                chunk_length=150, anchor_frames=90,
                                num_steps=50, guidance_scale=0.5,
                                text_dim=0, text_dir=None, speaker=None,
                                rank=0, world_size=1):
    """Autoreg FGD validation using the full generation pipeline.
    When world_size > 1, distributes recordings across ranks for ~Nx speedup.
    Returns FGD on rank 0, float('nan') on other ranks.
    """
    from generate_autoregressive_v2 import load_recording, build_chunks
    from prior_net.generate_diffusion import generate_autoregressive_with_diffusion
    from evaluate_generation import get_test_files, axis_angle_to_rot6d
    from emage_evaltools.mertic import FGD

    diff_model.eval()
    actual_vae = vae_model.module if hasattr(vae_model, 'module') else vae_model
    if hasattr(actual_vae, '_z_padding_mask'):
        actual_vae._z_padding_mask = None

    # Fix seed for reproducible diffusion sampling (same result every call)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    all_test_files = get_test_files(data_root, language, test_fraction, seed, speaker=speaker)
    # Distribute files across ranks (each rank handles its slice)
    my_files = all_test_files[rank::world_size]

    # Each rank generates motion for its files and collects (gen_r6d, gt_r6d) pairs
    my_pairs = []
    for pose_path in my_files:
        basename = os.path.splitext(os.path.basename(pose_path))[0]
        try:
            recording = load_recording(pose_path, language, data_root)
        except Exception:
            continue

        text_features_np = None
        if text_dim > 0 and text_dir is not None:
            text_path = os.path.join(text_dir, basename + '.npy')
            if os.path.exists(text_path):
                text_features_np = np.load(text_path)
                text_features_np = text_features_np[:recording['audio'].shape[0]]

        chunks = build_chunks(recording, max_chunk_length=chunk_length)
        if not chunks:
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            gen_poses, _ = generate_autoregressive_with_diffusion(
                vae_model, diff_model, recording, chunks, device,
                anchor_k=anchor_frames, overlap=0,
                num_steps=num_steps, guidance_scale=guidance_scale,
                text_features_np=text_features_np,
            )

        T_gen = gen_poses.shape[0]
        start_frame = chunks[0]['start']
        gt_poses = torch.FloatTensor(recording['poses'][start_frame:start_frame + T_gen])
        T = min(T_gen, gt_poses.shape[0])
        T_trunc = (T // 32) * 32
        if T_trunc < 64:
            continue

        gen_r6d = axis_angle_to_rot6d(gen_poses[:T_trunc].reshape(T_trunc, 55, 3)).reshape(T_trunc, 330)
        gt_r6d = axis_angle_to_rot6d(gt_poses[:T_trunc].reshape(T_trunc, 55, 3)).reshape(T_trunc, 330)
        my_pairs.append((gen_r6d.cpu(), gt_r6d.cpu()))

    if world_size > 1:
        # Gather all pairs to rank 0
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(my_pairs, gathered, dst=0)
        if rank != 0:
            return float('nan')
        all_pairs = [p for rank_pairs in gathered for p in rank_pairs]
    else:
        all_pairs = my_pairs

    if not all_pairs:
        return float('inf')

    fgd_evaluator = FGD(download_path='./emage_evaltools/', device=str(device))
    for gen_r6d, gt_r6d in all_pairs:
        fgd_evaluator.update(gen_r6d.unsqueeze(0).to(device).float(),
                             gt_r6d.unsqueeze(0).to(device).float())

    fgd_val = fgd_evaluator.compute()
    fgd_evaluator.reset()
    return fgd_val


def update_ema(ema_model, model, decay=0.999):
    """Update EMA model parameters."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def main():
    parser = argparse.ArgumentParser(description='Train temporal diffusion prior (online)')
    parser.add_argument('--resume', type=str, default=None)
    # VAE
    parser.add_argument('--vae-checkpoint', type=str,
                        default='checkpoint/beat2_poses/0/vae_temporal_lean/best.pt')
    parser.add_argument('--dataset-name', type=str, default='beat2_poses')
    parser.add_argument('--run-num', type=int, default=0)
    parser.add_argument('--folder-name', type=str, default='vae_temporal_lean')
    # Diffusion model
    parser.add_argument('--d-model', type=int, default=192)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-audio-layers', type=int, default=3)
    parser.add_argument('--num-denoiser-layers', type=int, default=4)
    parser.add_argument('--dim-feedforward', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--num-speakers', type=int, default=31)
    # Training
    parser.add_argument('--checkpoint-dir', type=str, default='prior_net/checkpoints_diff_online')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--cond-drop-prob', type=float, default=0.1)
    parser.add_argument('--anchor-drop-prob', type=float, default=0.3)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--val-num-steps', type=int, default=20)
    # Data
    parser.add_argument('--data-root', type=str, default='BEAT2')
    parser.add_argument('--language', type=str, default='english')
    parser.add_argument('--min-length', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=150)
    parser.add_argument('--anchor-max-frames', type=int, default=90)
    parser.add_argument('--speaker', type=int, default=None, help='Train on single speaker only (e.g. 2 for scott)')
    parser.add_argument('--reset-best', action='store_true', help='Reset best_val_loss on resume (for fine-tuning on different data)')
    # Text conditioning
    parser.add_argument('--text-dim', type=int, default=0, help='Text embedding dim (768 for BERT, 0 to disable)')
    parser.add_argument('--text-dir', type=str, default=None, help='Path to BERT text features (e.g. BEAT2/beat_english_v2.0.0/bert_30)')
    # Autoreg FGD validation
    parser.add_argument('--val-autoreg-every', type=int, default=5, help='Run autoreg FGD validation every N epochs (0 to disable)')
    parser.add_argument('--val-autoreg-fraction', type=float, default=0.1, help='Fraction of test files for autoreg FGD')
    parser.add_argument('--val-autoreg-guidance', type=float, default=0.5, help='Guidance scale for autoreg FGD validation')
    args = parser.parse_args()

    # ===== DDP setup =====
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    log_csv = os.path.join(args.checkpoint_dir, 'loss_log.csv')

    # ===== Load frozen VAE =====
    if rank == 0:
        print("Loading frozen VAE...")
    options, _ = conf_parser(args.dataset_name, args.run_num, args.folder_name)
    model_type = get_model_type(args.folder_name)
    vae_model = create_model_object(model_type, options)
    state_dict = load_checkpoint(args.vae_checkpoint, device)
    # Filter out SMPLX params with shape mismatch (num_betas/expression differ)
    model_sd = vae_model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    vae_model.load_state_dict(filtered, strict=False)
    vae_model.to(device).eval()
    for p in vae_model.parameters():
        p.requires_grad_(False)

    latent_dim = vae_model.latent_dim
    temporal_downsample = vae_model.temporal_downsample
    if rank == 0:
        print(f"  VAE: latent_dim={latent_dim}, temporal_downsample={temporal_downsample}")

    # ===== Datasets =====
    if rank == 0:
        print("Loading datasets...")
    use_text = args.text_dim > 0
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

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=variable_length_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=variable_length_collate_fn,
    )
    if rank == 0:
        print(f"  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # ===== Compute z normalization stats (rank 0 only, then broadcast) =====
    if rank == 0:
        print("Computing z normalization stats from training data...")
    z_mean = torch.zeros(latent_dim, device=device)
    z_std = torch.ones(latent_dim, device=device)
    if rank == 0:
        z_tokens = []
        with torch.no_grad():
            for i, (batch_data, _) in enumerate(tqdm(train_loader, desc="Encoding for stats")):
                if i >= 50:  # Sample ~50 batches for stats
                    break
                mu, z_padding_mask = encode_batch(vae_model, batch_data, device)
                mu_cpu = mu.cpu()
                if z_padding_mask is not None:
                    for b in range(mu_cpu.shape[0]):
                        valid_len = (~z_padding_mask[b]).sum().item()
                        z_tokens.append(mu_cpu[b, :valid_len, :].reshape(-1, latent_dim))
                else:
                    z_tokens.append(mu_cpu.reshape(-1, latent_dim))
        all_z = torch.cat(z_tokens, dim=0)
        z_mean = all_z.mean(dim=0).to(device)
        z_std = all_z.std(dim=0).clamp(min=1e-6).to(device)
        print(f"  z stats ({all_z.shape[0]} tokens): mean_norm={z_mean.norm():.2f}, mean_std={z_std.mean():.4f}")
    if world_size > 1:
        dist.broadcast(z_mean, src=0)
        dist.broadcast(z_std, src=0)

    # ===== Diffusion model =====
    diff_model = TemporalFlowMatchingPrior(
        latent_dim=latent_dim,
        audio_dim=768,
        d_model=args.d_model,
        nhead=args.nhead,
        num_audio_layers=args.num_audio_layers,
        num_denoiser_layers=args.num_denoiser_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_speakers=args.num_speakers,
        temporal_downsample=temporal_downsample,
        cond_drop_prob=args.cond_drop_prob,
        text_dim=args.text_dim,
    ).to(device)

    diff_model.z_mean.copy_(z_mean)
    diff_model.z_std.copy_(z_std)

    # EMA model (rank 0 only — only needed for saving/validation)
    ema_model = None
    if rank == 0:
        ema_model = copy.deepcopy(diff_model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    if rank == 0:
        n_params = sum(p.numel() for p in diff_model.parameters() if p.requires_grad)
        print(f"  Diffusion model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        diff_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        diff_model.load_state_dict(ckpt['model'])
        if rank == 0 and ema_model is not None:
            ema_model.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = float('inf') if args.reset_best else ckpt['best_val_loss']
        # Re-create scheduler for remaining epochs (avoid dead LR from old cosine)
        remaining = args.epochs - start_epoch
        if remaining > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-6)
            # Reset optimizer LR to args.lr for fresh cosine cycle
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
            if rank == 0:
                print(f"  Fresh cosine schedule: {remaining} epochs, lr={args.lr}")

    best_ar_fgd = float('inf')

    # CSV log (rank 0 only)
    csv_fields = ['epoch', 'train_loss', 'val_loss', 'val_mse', 'ema_val_loss', 'ema_val_mse', 'lr', 'ar_fgd']
    if rank == 0:
        if not os.path.exists(log_csv) or (start_epoch == 0 and not args.resume):
            with open(log_csv, 'w', newline='') as f:
                csv.writer(f).writerow(csv_fields)

    # Wrap in DDP after loading checkpoint
    if world_size > 1:
        diff_model_ddp = torch.nn.parallel.DistributedDataParallel(
            diff_model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        diff_model_ddp = diff_model

    if rank == 0:
        print(f"\nTraining for {args.epochs - start_epoch} epochs (world_size={world_size})...")
        print(f"  d_model={args.d_model}, audio_layers={args.num_audio_layers}, "
              f"denoiser_layers={args.num_denoiser_layers}")
        print(f"  lr={args.lr}, dropout={args.dropout}, wd={args.weight_decay}")
        print(f"  cond_drop={args.cond_drop_prob}, anchor_drop={args.anchor_drop_prob}, "
              f"ema_decay={args.ema_decay}")
        if args.text_dim > 0:
            print(f"  text_dim={args.text_dim}, text_dir={args.text_dir}")

    for epoch in range(start_epoch, args.epochs):
        # Set epoch for reproducible shuffling in DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            diff_model_ddp, vae_model, train_loader, optimizer, device,
            grad_clip=args.grad_clip,
            cond_drop_prob=args.cond_drop_prob,
            anchor_drop_prob=args.anchor_drop_prob,
        )

        # Aggregate train loss across ranks
        if world_size > 1:
            t = torch.tensor([train_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            train_loss = t.item()

        # EMA update (rank 0 only)
        if rank == 0 and ema_model is not None:
            update_ema(ema_model, diff_model, args.ema_decay)

        # Autoreg FGD validation — all ranks participate (distributed generation)
        ar_fgd = float('nan')
        if args.val_autoreg_every > 0 and (epoch % args.val_autoreg_every == 0 or epoch == args.epochs - 1):
            if rank == 0:
                tqdm.write(f"  Running autoreg FGD validation (distributed across {world_size} GPUs)...")
            ar_fgd = validate_fgd_autoregressive(
                diff_model, vae_model, device,
                data_root=args.data_root, language=args.language,
                test_fraction=args.val_autoreg_fraction, seed=42,
                num_steps=50, guidance_scale=args.val_autoreg_guidance,
                text_dim=args.text_dim, text_dir=args.text_dir,
                speaker=args.speaker,
                rank=rank, world_size=world_size,
            )
            if rank == 0:
                tqdm.write(f"  ar_fgd={ar_fgd:.4f}")

        # ===== Rank 0 only: validation, logging, checkpointing =====
        if rank == 0:
            # Validate every 5 epochs
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                val_metrics = validate_epoch(
                    diff_model, vae_model, val_loader, device,
                    num_steps=args.val_num_steps)
                ema_metrics = validate_epoch(
                    ema_model, vae_model, val_loader, device,
                    num_steps=args.val_num_steps)
                val_loss = val_metrics['loss']
                val_mse = val_metrics['mse']
                ema_val_loss = ema_metrics['loss']
                ema_val_mse = ema_metrics['mse']
            else:
                # Quick val (loss only, no sampling)
                diff_model.eval()
                quick_loss = 0
                quick_n = 0
                with torch.no_grad():
                    for batch_data, _ in val_loader:
                        mu, z_padding_mask = encode_batch(vae_model, batch_data, device)
                        audio = batch_data['audio'].to(device)
                        spk = batch_data['speaker_id'].to(device)
                        gtype = batch_data['gesture_type'].to(device)
                        pmask = batch_data['padding_mask'].to(device)
                        anchor_pool = batch_data['anchor_pool'].to(device)
                        anchor_audio = batch_data['anchor_audio'].to(device)
                        txt = batch_data.get('text')
                        if txt is not None:
                            txt = txt.to(device)
                        B = mu.shape[0]
                        loss = diff_model.compute_loss(
                            mu, audio, spk, gtype, pmask,
                            z_padding_mask=z_padding_mask, cond_drop_prob=0,
                            anchor_frames=anchor_pool, anchor_audio=anchor_audio,
                            text_features=txt)
                        quick_loss += loss.item() * B
                        quick_n += B
                val_loss = quick_loss / quick_n
                val_mse = float('nan')
                ema_val_loss = float('nan')
                ema_val_mse = float('nan')
                diff_model.train()

            lr = optimizer.param_groups[0]['lr']

            # Log
            ar_fgd_str = f'{ar_fgd:.6f}' if not math.isnan(ar_fgd) else ''
            with open(log_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch, f'{train_loss:.6f}', f'{val_loss:.6f}',
                    f'{val_mse:.6f}', f'{ema_val_loss:.6f}', f'{ema_val_mse:.6f}',
                    f'{lr:.6f}', ar_fgd_str])

            tqdm.write(
                f"Ep {epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} "
                f"mse={val_mse:.4f} ema_val={ema_val_loss:.4f} ema_mse={ema_val_mse:.4f} "
                f"lr={lr:.6f}")

            # Save best_ar.pt based on autoreg FGD
            if not math.isnan(ar_fgd) and ar_fgd < best_ar_fgd:
                best_ar_fgd = ar_fgd
                save_dict = {
                    'model': diff_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'best_ar_fgd': best_ar_fgd,
                    'z_mean': diff_model.z_mean,
                    'z_std': diff_model.z_std,
                    'args': vars(args),
                    'is_temporal': True,
                }
                torch.save(save_dict, os.path.join(args.checkpoint_dir, 'best_ar.pt'))
                tqdm.write(f"  -> Saved best_ar (ar_fgd={best_ar_fgd:.4f})")

            # Save best.pt based on FM val loss
            check_loss = ema_val_loss if not math.isnan(ema_val_loss) else val_loss
            if check_loss < best_val_loss:
                best_val_loss = check_loss
                save_dict = {
                    'model': diff_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'z_mean': diff_model.z_mean,
                    'z_std': diff_model.z_std,
                    'args': vars(args),
                    'is_temporal': True,
                }
                torch.save(save_dict, os.path.join(args.checkpoint_dir, 'best.pt'))
                tqdm.write(f"  -> Saved best (val_loss={best_val_loss:.6f})")

            # Periodic save
            if (epoch + 1) % args.save_every == 0:
                save_dict = {
                    'model': diff_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'z_mean': diff_model.z_mean,
                    'z_std': diff_model.z_std,
                    'args': vars(args),
                    'is_temporal': True,
                }
                torch.save(save_dict, os.path.join(
                    args.checkpoint_dir, f'{epoch:03d}.pt'))

        scheduler.step()

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print("Done.")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
