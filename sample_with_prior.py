"""
Sample poses using trained PixelSNAIL prior models.

This script generates high-quality poses by:
1. Sampling codes from PixelSNAIL priors (autoregressive)
2. Decoding codes with VQ-VAE to get poses

The sampling is hierarchical:
  - Sample z1 from p(z1) using top PixelSNAIL
  - Sample z2 from p(z2|z1) using middle PixelSNAIL
  - Sample z3 from p(z3|z2) using bottom PixelSNAIL
  - Decode [z1, z2, z3] to poses using VQ-VAE

Usage:
    python sample_with_prior.py --vqvae-checkpoint checkpoint/beat2_poses/0/vqvae/010.pt --num-samples 10
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from m_util import model_object_parser, load_checkpoint
from m_pixelsnail import PixelSNAIL


@torch.no_grad()
def sample_pixelsnail(model, batch_size, seq_height, seq_width, device, condition=None, temperature=1.0):
    """
    Sample codes from PixelSNAIL autoregressively.

    Args:
        model: Trained PixelSNAIL model
        batch_size: Number of samples
        seq_height: Height of code sequence
        seq_width: Width of code sequence
        device: Device
        condition: Conditioning codes (batch, 1, height, width) if conditional
        temperature: Sampling temperature (higher = more diverse, lower = more conservative)

    Returns:
        sampled_codes: (batch, seq_height, seq_width) integer codes
    """
    model.eval()

    # Initialize with zeros (or random)
    codes = torch.zeros(batch_size, 1, seq_height, seq_width, dtype=torch.long, device=device)

    # Convert condition to one-hot if provided
    cond_onehot = None
    if condition is not None:
        cond_flat = condition.view(-1)
        cond_onehot = F.one_hot(cond_flat, model.n_class).float()
        cond_onehot = cond_onehot.view(batch_size, seq_height, seq_width, model.n_class).permute(0, 3, 1, 2)

    cache = {}

    # Sample autoregressively
    for i in tqdm(range(seq_height), desc='Sampling rows'):
        for j in range(seq_width):
            # Convert current codes to one-hot
            codes_flat = codes.view(-1)
            codes_onehot = F.one_hot(codes_flat, model.n_class).float()
            codes_onehot = codes_onehot.view(batch_size, seq_height, seq_width, model.n_class).permute(0, 3, 1, 2)

            # Forward through PixelSNAIL
            out, cache = model(codes_onehot, condition=cond_onehot, cache=cache)

            # Get logits for current position
            logits = out[:, :, i, j]  # (batch, n_class)

            # Apply temperature
            logits = logits / temperature

            # Sample from distribution
            probs = F.softmax(logits, dim=1)
            sampled = torch.multinomial(probs, 1).squeeze(1)  # (batch,)

            # Update codes
            codes[:, 0, i, j] = sampled

    return codes.squeeze(1)  # (batch, height, width)


@torch.no_grad()
def sample_hierarchical(vqvae_model, pixelsnail_models, batch_size, seq_height, seq_width, device, temperature=1.0):
    """
    Sample poses hierarchically using all PixelSNAIL levels.

    Args:
        vqvae_model: Trained VQ-VAE model
        pixelsnail_models: Dict with keys 'level_1', 'level_2', 'level_3' containing PixelSNAIL models
        batch_size: Number of samples
        seq_height: Height of code sequence
        seq_width: Width of code sequence
        device: Device
        temperature: Sampling temperature

    Returns:
        generated_poses: (batch, seq_len, 165) generated poses
        codes_all: List of codes for each level
    """
    print(f"\nGenerating {batch_size} samples hierarchically...")

    # Level 1: Sample top level (unconditional)
    print("  Sampling Level 1 (top)...")
    codes_1 = sample_pixelsnail(
        pixelsnail_models['level_1'],
        batch_size, seq_height, seq_width,
        device, condition=None, temperature=temperature
    )
    print(f"    Codes shape: {codes_1.shape}")

    # Level 2: Sample middle level (conditional on Level 1)
    print("  Sampling Level 2 (middle)...")
    codes_1_cond = codes_1.unsqueeze(1)  # Add channel dim for conditioning
    codes_2 = sample_pixelsnail(
        pixelsnail_models['level_2'],
        batch_size, seq_height, seq_width,
        device, condition=codes_1_cond, temperature=temperature
    )
    print(f"    Codes shape: {codes_2.shape}")

    # Level 3: Sample bottom level (conditional on Level 2)
    print("  Sampling Level 3 (bottom)...")
    codes_2_cond = codes_2.unsqueeze(1)
    codes_3 = sample_pixelsnail(
        pixelsnail_models['level_3'],
        batch_size, seq_height, seq_width,
        device, condition=codes_2_cond, temperature=temperature
    )
    print(f"    Codes shape: {codes_3.shape}")

    # Flatten codes from 2D back to 1D sequence
    # (batch, height, width) -> (batch, seq_len)
    codes_1_flat = codes_1.view(batch_size, -1)
    codes_2_flat = codes_2.view(batch_size, -1)
    codes_3_flat = codes_3.view(batch_size, -1)

    codes_all = [codes_1_flat, codes_2_flat, codes_3_flat]

    # Decode with VQ-VAE
    print("  Decoding with VQ-VAE...")
    actual_model = vqvae_model.module if hasattr(vqvae_model, 'module') else vqvae_model
    generated_poses = actual_model.decode_code(codes_all)

    print(f"    Generated poses shape: {generated_poses.shape}")

    return generated_poses, codes_all


def load_pixelsnail_model(checkpoint_path, n_class, seq_height, seq_width, level, device):
    """Load a trained PixelSNAIL model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get architecture from saved args
    args = checkpoint.get('args', {})

    # Create model with saved architecture
    model = PixelSNAIL(
        shape=[seq_height, seq_width],
        n_class=n_class,
        channel=args.get('channel', 256),
        kernel_size=args.get('kernel_size', 5),
        n_block=args.get('n_block', 4),
        n_res_block=args.get('n_res_block', 4),
        res_channel=args.get('res_channel', 256),
        attention=args.get('attention', True),
        dropout=args.get('dropout', 0.1),
        cond_channel=n_class if level > 1 else 0,
        n_cond_res_block=args.get('n_res_block', 4) if level > 1 else 0,
        cond_res_channel=args.get('res_channel', 256) if level > 1 else 0,
        cond_res_kernel=3,
        n_out_res_block=0
    )

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description='Sample poses using PixelSNAIL priors')

    # Models
    parser.add_argument('--vqvae-checkpoint', type=str, required=True,
                       help='Path to trained VQ-VAE checkpoint')
    parser.add_argument('--pixelsnail-dir', type=str, default='checkpoint/beat2_poses/0/pixelsnail',
                       help='Directory containing PixelSNAIL checkpoints')
    parser.add_argument('--use-best', action='store_true',
                       help='Use best.pt checkpoints instead of latest')

    # Sampling
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (0.8-1.2 recommended)')
    parser.add_argument('--seq-height', type=int, default=5,
                       help='Height of code sequence')
    parser.add_argument('--seq-width', type=int, default=5,
                       help='Width of code sequence')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    # Output
    parser.add_argument('--output', type=str, default='generated_poses_with_prior.npz',
                       help='Output file for generated poses')

    args = parser.parse_args()

    print("="*60)
    print("SAMPLING WITH PIXELSNAIL PRIORS")
    print("="*60)

    # Load VQ-VAE
    print("\n1. Loading VQ-VAE...")
    vqvae_model = model_object_parser('beat2_poses', 0, 'vqvae')
    state_dict = load_checkpoint(args.vqvae_checkpoint, args.device)
    vqvae_model.load_state_dict(state_dict)
    vqvae_model = vqvae_model.to(args.device)
    vqvae_model.eval()
    print(f"   ✓ Loaded from: {args.vqvae_checkpoint}")

    # Load PixelSNAIL models
    print("\n2. Loading PixelSNAIL models...")
    codebook_sizes = {1: 8, 2: 64, 3: 512}
    pixelsnail_models = {}

    checkpoint_name = 'best.pt' if args.use_best else '*.pt'

    for level in [1, 2, 3]:
        level_dir = os.path.join(args.pixelsnail_dir, f'level_{level}')

        if args.use_best:
            ckpt_path = os.path.join(level_dir, 'best.pt')
        else:
            # Find latest checkpoint
            import glob
            checkpoints = glob.glob(os.path.join(level_dir, '*.pt'))
            checkpoints = [c for c in checkpoints if 'best' not in c]
            if not checkpoints:
                print(f"   ✗ No checkpoints found for level {level} in {level_dir}")
                print(f"   Please train level {level} first:")
                print(f"     python train_pixelsnail_hierarchical.py --level {level} --codes codes_dataset.npz")
                return
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            ckpt_path = checkpoints[0]

        if not os.path.exists(ckpt_path):
            print(f"   ✗ Checkpoint not found: {ckpt_path}")
            print(f"   Please train level {level} first")
            return

        print(f"   Loading Level {level} from: {ckpt_path}")
        pixelsnail_models[f'level_{level}'] = load_pixelsnail_model(
            ckpt_path,
            n_class=codebook_sizes[level],
            seq_height=args.seq_height,
            seq_width=args.seq_width,
            level=level,
            device=args.device
        )

    print("   ✓ All PixelSNAIL models loaded!")

    # Generate samples
    print("\n3. Generating samples...")
    generated_poses, codes_all = sample_hierarchical(
        vqvae_model,
        pixelsnail_models,
        batch_size=args.num_samples,
        seq_height=args.seq_height,
        seq_width=args.seq_width,
        device=args.device,
        temperature=args.temperature
    )

    # Save results
    print("\n4. Saving results...")
    np.savez(
        args.output,
        poses=generated_poses.cpu().numpy(),
        codes_level_1=codes_all[0].cpu().numpy(),
        codes_level_2=codes_all[1].cpu().numpy(),
        codes_level_3=codes_all[2].cpu().numpy()
    )
    print(f"   ✓ Saved to: {args.output}")

    print("\n" + "="*60)
    print("SAMPLING COMPLETE!")
    print("="*60)
    print(f"Generated {args.num_samples} poses")
    print(f"Temperature: {args.temperature}")
    print(f"\nVisualize with:")
    print(f"  python visualize_pose_comparison.py --sample-file {args.output}")


if __name__ == '__main__':
    main()
