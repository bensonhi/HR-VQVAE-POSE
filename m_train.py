import argparse
import csv
import os

from m_trainer import train
from m_sample import vae_sampler

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true',
                    help='Resume training from latest checkpoint (reads loss_log.csv)')
cli_args = parser.parse_args()

dataset_name = 'beat2_poses'
folder_name = 'vae'

n_run = 0
start_epoch = -1
end_epoch = -1
batch_size = -1
lr = -1
sched=None
device='cuda'
size=28
amp=None

sample_period = 1
sampler = vae_sampler

# Early stopping: -1 to disable, positive number for patience (epochs without improvement)
patience = 30  # Default: stop if no improvement for 30 epochs

# ===== VAE KL Annealing Configuration =====
kl_anneal_epochs = 0
max_kl_weight = 1e-5
vel_weight = 3.0  # Velocity (frame-difference) loss weight
cont_vel_weight = 3.0  # Continuation velocity loss weight (anchor→first frame)

# ===== Progressive Training Configuration =====
use_progressive = True

# Weights for each level in progressive training
level_1_weight = 1.0  # Face pose loss weight
level_2_weight = 1.0  # Body pose loss weight
level_3_weight = 1.0  # Hands pose loss weight

from m_beat_dataset import get_beat_variable_length_loader
anchor_max_frames = 30

train_loader = get_beat_variable_length_loader(
    data_path='BEAT2',
    language='english',
    batch_size=16,  # reduced from 32: 3 decoder graphs w/o detach need more memory
    min_length=5,
    max_length=150,
    shuffle=True,
    num_workers=8,
    use_axis_angle=True,
    split='train',
    anchor_max_frames=anchor_max_frames,
)
val_loader = get_beat_variable_length_loader(
    data_path='BEAT2',
    language='english',
    batch_size=16,
    min_length=5,
    max_length=150,
    shuffle=False,
    num_workers=8,
    use_axis_angle=True,
    split='val',
    anchor_max_frames=anchor_max_frames,
)

# ===== Resume from latest checkpoint =====
if cli_args.resume:
    log_path = f'checkpoint/{dataset_name}/{n_run}/vae/loss_log.csv'
    if os.path.exists(log_path):
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            latest_epoch = int(rows[-1]['epoch'])  # 1-indexed in CSV
            start_epoch = latest_epoch  # loop starts at this 0-indexed value
            print(f"Resuming from epoch {latest_epoch} (will load checkpoint {latest_epoch-1:03d}.pt)")
        else:
            print("loss_log.csv is empty, starting from scratch")
    else:
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
       batch_size=batch_size,
       sched=sched,
       device=device,
       size=size,
       lr=lr,
       amp=amp,
       use_progressive=use_progressive,
       level_1_weight=level_1_weight,
       level_2_weight=level_2_weight,
       level_3_weight=level_3_weight,
       patience=patience,
       kl_anneal_epochs=kl_anneal_epochs,
       max_kl_weight=max_kl_weight,
       val_loader=val_loader,
       vel_weight=vel_weight,
       cont_vel_weight=cont_vel_weight)
