from m_trainer import train
from m_sample import vae_sampler

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
kl_anneal_epochs = 100
max_kl_weight = 0.05

# ===== Progressive Training Configuration =====
use_progressive = True

# Weights for each level in progressive training
level_1_weight = 1.0  # Face pose loss weight
level_2_weight = 1.0  # Body pose loss weight
level_3_weight = 1.0  # Hands pose loss weight

from m_beat_dataset import get_beat_variable_length_loader
train_loader = get_beat_variable_length_loader(
    data_path='BEAT2',
    language='english',
    batch_size=32,
    min_length=5,
    max_length=300,
    shuffle=True,
    num_workers=8,
    use_axis_angle=True,
    split='train',
)
val_loader = get_beat_variable_length_loader(
    data_path='BEAT2',
    language='english',
    batch_size=32,
    min_length=5,
    max_length=300,
    shuffle=False,
    num_workers=8,
    use_axis_angle=True,
    split='val',
)

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
       val_loader=val_loader)
