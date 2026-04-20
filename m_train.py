import argparse
import csv
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from m_trainer import train
from m_sample import vae_sampler
from m_beat_dataset import BEAT2PoseDataset, variable_length_collate_fn

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true',
                    help='Resume training from latest checkpoint (reads loss_log.csv)')
parser.add_argument('--reset-best', action='store_true',
                    help='Reset best val_loss on resume (for fine-tuning on different data)')
parser.add_argument('--speaker', type=int, default=None,
                    help='Train on single speaker only (e.g. 2 for scott)')
parser.add_argument('--text-dim', type=int, default=0,
                    help='Text embedding dim (768 for BERT, 0 to disable)')
parser.add_argument('--text-dir', type=str, default=None,
                    help='Path to BERT text features (e.g. BEAT2/beat_english_v2.0.0/bert_30)')
parser.add_argument('--folder-name', type=str, default=None,
                    help='Override checkpoint folder name (e.g. vae_temporal_4x)')
cli_args = parser.parse_args()

# ===== DDP setup =====
# Works with both `torchrun` (sets env vars) and plain `python` (single GPU)
if 'RANK' in os.environ:
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
else:
    rank = 0
    local_rank = 0
    world_size = 1
    device = 'cuda'

dataset_name = 'beat2_poses'
if cli_args.text_dim > 0:
    base_folder = 'vae_temporal_lean_text'
else:
    base_folder = 'vae_temporal_lean'
if cli_args.folder_name is not None:
    folder_name = cli_args.folder_name
elif cli_args.speaker is not None:
    folder_name = f'{base_folder}_spk{cli_args.speaker}'
else:
    folder_name = base_folder

n_run = 0
start_epoch = -1
end_epoch = -1
batch_size = -1
lr = -1
sched = None
size = 28
amp = None

sample_period = 1
sampler = vae_sampler

# Early stopping: -1 to disable, positive number for patience (epochs without improvement)
patience = 30  # Default: stop if no improvement for 30 epochs

# ===== VAE KL Annealing Configuration =====
kl_anneal_epochs = 0
max_kl_weight = 1e-5
vel_weight = 3.0  # Velocity (frame-difference) loss weight
cont_vel_weight = 3.0  # Continuation velocity loss weight (anchor→first frame)
anchor_recon_weight = 0.1  # Anchor reconstruction loss weight

# ===== Progressive Training Configuration =====
use_progressive = True

# Weights for each level in progressive training (face, body, hands, whole-body, ...)
level_weights = [1.0, 1.0, 1.0, 1.0]

anchor_max_frames = 90
per_gpu_batch_size = 8  # vae_temporal_2x is larger (d_model=768, n_level=4)

# ===== Create datasets =====
use_text = cli_args.text_dim > 0
train_dataset = BEAT2PoseDataset(
    data_path='BEAT2',
    language='english',
    min_length=5,
    max_length=150,
    use_axis_angle=True,
    split='train',
    anchor_max_frames=anchor_max_frames,
    speaker=cli_args.speaker,
    load_text=use_text, text_dir=cli_args.text_dir,
)
val_dataset = BEAT2PoseDataset(
    data_path='BEAT2',
    language='english',
    min_length=5,
    max_length=150,
    use_axis_angle=True,
    split='val',
    anchor_max_frames=anchor_max_frames,
    speaker=cli_args.speaker,
    load_text=use_text, text_dir=cli_args.text_dir,
)

# ===== Create data loaders (DDP-aware) =====
if world_size > 1:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
else:
    train_sampler = None
    val_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=per_gpu_batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=variable_length_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=per_gpu_batch_size,
    shuffle=False,
    sampler=val_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=variable_length_collate_fn,
)

# ===== Resume from latest checkpoint =====
if cli_args.resume:
    log_path = f'checkpoint/{dataset_name}/{n_run}/{folder_name}/loss_log.csv'
    if os.path.exists(log_path):
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            latest_epoch = int(rows[-1]['epoch'])  # 1-indexed in CSV
            start_epoch = latest_epoch  # loop starts at this 0-indexed value
            if rank == 0:
                print(f"Resuming from epoch {latest_epoch} (will load checkpoint {latest_epoch-1:03d}.pt)")
        else:
            if rank == 0:
                print("loss_log.csv is empty, starting from scratch")
    else:
        if rank == 0:
            print("No loss_log.csv found, starting from scratch")

train(
       folder_name,
       train_loader,
       dataset_name,
       n_run,
       sample_period,
       sampler,
       start_epoch=start_epoch,
       end_epoch=end_epoch,
       batch_size=per_gpu_batch_size,
       sched=sched,
       device=device,
       size=size,
       lr=lr,
       amp=amp,
       use_progressive=use_progressive,
       level_weights=level_weights,
       patience=patience,
       kl_anneal_epochs=kl_anneal_epochs,
       max_kl_weight=max_kl_weight,
       val_loader=val_loader,
       vel_weight=vel_weight,
       cont_vel_weight=cont_vel_weight,
       anchor_recon_weight=anchor_recon_weight,
       reset_best=cli_args.reset_best,
       rank=rank,
       local_rank=local_rank,
       world_size=world_size)

# ===== Cleanup =====
if world_size > 1:
    dist.destroy_process_group()
