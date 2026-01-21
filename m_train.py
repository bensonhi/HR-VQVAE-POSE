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
# For continuous VAE training: gradually increase KL loss weight from 0 to max_kl_weight
# This helps prevent KL collapse and improves reconstruction quality
kl_anneal_epochs = 100  # Number of epochs to anneal from 0 to max (100 epochs = gradual increase)
max_kl_weight = 0.05    # Maximum KL weight (beta in beta-VAE). Lower = better reconstruction, higher = better regularization

# ===== Progressive Training Configuration =====
# Set to True to enable progressive training with level-specific losses:
#   - Level 1: Focus on face pose
#   - Level 2: Focus on body pose
#   - Level 3: Focus on hands + overall reconstruction
# Set to False for standard training (all levels contribute to final output)
use_progressive = True

# Weights for each level in progressive training (only used if use_progressive=True)
level_1_weight = 1.0  # Face pose loss weight
level_2_weight = 1.0  # Body pose loss weight
level_3_weight = 1.0  # Hands pose loss weight

from m_beat_dataset import get_combined_pose_loader
loader = get_combined_pose_loader(
    beat2_path='BEAT2_joints_vertices',  # Has both poses and pre-computed GT
    amass_path='AMASS',  # AMASS directory
    amass_subsets=['BMLrub'],  # Include BMLrub subset from AMASS
    language='english',
    batch_size=32,
    sequence_length=25,
    shuffle=True,
    num_workers=8,  # Parallel data loading
    use_axis_angle=True,  # Load axis-angle poses (165D)
    load_gt_geometry=True,  # Load ground truth vertices and joints for supervision
    compute_gt_on_fly=True,  # Compute GT on-the-fly from poses using SMPLX
    smplx_model_path='models_smplx_v1_1/models'
)

train(
       folder_name,
       loader,
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
       max_kl_weight=max_kl_weight)
