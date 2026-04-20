"""Compute val loss of a prior checkpoint on speaker 2 val data only."""
import os, sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader
from m_beat_dataset import BEAT2PoseDataset, variable_length_collate_fn
from m_util import conf_parser, create_model_object, get_model_type, load_checkpoint
from prior_net.diffusion_model import TemporalFlowMatchingPrior


@torch.no_grad()
def encode_batch(vae_model, batch, device):
    poses = batch['poses'].to(device)
    audio = batch['audio'].to(device)
    gesture_type = batch['gesture_type'].to(device)
    speaker_id = batch['speaker_id'].to(device)
    padding_mask = batch['padding_mask'].to(device)
    z, mu, logvar, kl_loss, raw_kl, kl_per_level = vae_model.encode(
        poses, padding_mask, gesture_type=gesture_type,
        audio_features=audio, speaker_id=speaker_id)
    z_padding_mask = vae_model._z_padding_mask
    return mu, z_padding_mask


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VAE
    options, _ = conf_parser('beat2_poses', 0, 'vae_temporal_lean')
    model_type = get_model_type('vae_temporal_lean')
    vae = create_model_object(model_type, options)
    state_dict = load_checkpoint('checkpoint/beat2_poses/0/vae_temporal_lean/best.pt', device)
    model_sd = vae.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
    vae.load_state_dict(filtered, strict=False)
    vae.to(device).eval()

    # Load prior
    ckpt = torch.load('prior_net/checkpoints_diff_online_v2/229.pt', map_location=device)
    diff_model = TemporalFlowMatchingPrior(
        latent_dim=16, audio_dim=768, d_model=192, nhead=4,
        num_audio_layers=3, num_denoiser_layers=4, dim_feedforward=768,
        dropout=0.15, num_speakers=31, temporal_downsample=8, cond_drop_prob=0.1,
    ).to(device)
    diff_model.load_state_dict(ckpt['model'])
    diff_model.z_mean.copy_(ckpt['z_mean'].to(device))
    diff_model.z_std.copy_(ckpt['z_std'].to(device))
    diff_model.eval()

    # Speaker 2 val data
    val_dataset = BEAT2PoseDataset(
        data_path='BEAT2', language='english', min_length=5, max_length=150,
        use_axis_angle=True, split='val', anchor_max_frames=90, speaker=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=4, collate_fn=variable_length_collate_fn)

    total_loss = 0
    total_samples = 0
    for batch_data, _ in val_loader:
        mu, z_padding_mask = encode_batch(vae, batch_data, device)
        audio = batch_data['audio'].to(device)
        speaker_id = batch_data['speaker_id'].to(device)
        gesture_type = batch_data['gesture_type'].to(device)
        padding_mask = batch_data['padding_mask'].to(device)
        anchor_pool = batch_data['anchor_pool'].to(device)
        anchor_audio = batch_data['anchor_audio'].to(device)
        B = mu.shape[0]

        loss = diff_model.compute_loss(
            mu, audio, speaker_id, gesture_type, padding_mask,
            z_padding_mask=z_padding_mask, cond_drop_prob=0.0,
            anchor_frames=anchor_pool, anchor_audio=anchor_audio)
        total_loss += loss.item() * B
        total_samples += B

    val_loss = total_loss / total_samples
    print(f"Speaker 2 val samples: {len(val_dataset)}")
    print(f"Val loss (speaker 2): {val_loss:.6f}")


if __name__ == '__main__':
    main()
