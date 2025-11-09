"""
Extract discrete codes from all training data.

This script processes the entire BEAT2 dataset and extracts the discrete
latent codes from the VQ-VAE. These codes can be used to:
1. Train a PixelSNAIL prior model for better sampling
2. Analyze codebook usage and detect collapse
3. Study what patterns the model has learned

Usage:
    python extract_all_codes.py --checkpoint checkpoint/beat2_poses/0/vqvae/010.pt --output codes_dataset.npz
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from m_util import model_object_parser, find_latest_checkpoint
from m_beat_dataset import get_beat_pose_loader
from m_sample import extract_codes_from_poses


def extract_all_codes(model, loader, device='cuda', max_batches=None):
    """
    Extract codes from entire dataset.

    Args:
        model: Trained VQ-VAE model
        loader: DataLoader for dataset
        device: Device to use
        max_batches: Maximum number of batches to process (None = all)

    Returns:
        all_codes: Dictionary with codes for each level
        statistics: Usage statistics for each level
    """
    model.eval()

    # Get actual model (unwrap DataParallel if needed)
    actual_model = model.module if hasattr(model, 'module') else model
    n_levels = actual_model.n_level

    # Initialize storage for codes
    all_codes = {f'level_{i+1}': [] for i in range(n_levels)}
    all_poses = []

    print(f"Extracting codes from dataset...")
    print(f"Number of levels: {n_levels}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Processing batches")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Extract poses from batch
            # BEAT2 dataset returns (dict, labels) tuple
            if isinstance(batch, tuple) and len(batch) == 2:
                batch_dict, _ = batch
                poses = batch_dict['poses']
            elif isinstance(batch, dict):
                poses = batch['poses']
            else:
                poses = batch[0] if isinstance(batch, (list, tuple)) else batch

            # Handle case where poses is still a dict (nested structure)
            if isinstance(poses, dict):
                # Try common keys
                if 'axis_angle' in poses:
                    poses = poses['axis_angle']
                elif 'rotmat' in poses:
                    poses = poses['rotmat']
                else:
                    # Just take the first tensor value
                    poses = list(poses.values())[0]

            poses = poses.to(device)

            # Extract codes
            codes, _ = extract_codes_from_poses(model, poses, device)

            # Store codes for each level
            for i, level_codes in enumerate(codes):
                all_codes[f'level_{i+1}'].append(level_codes.cpu().numpy())

            # Optionally store poses for reference
            all_poses.append(poses.cpu().numpy())

    # Concatenate all codes
    print("\nConcatenating codes...")
    for level_name in all_codes:
        all_codes[level_name] = np.concatenate(all_codes[level_name], axis=0)
        print(f"  {level_name}: {all_codes[level_name].shape}")

    all_poses = np.concatenate(all_poses, axis=0)
    print(f"  poses: {all_poses.shape}")

    # Compute global statistics
    print("\nComputing statistics...")
    statistics = compute_code_statistics(all_codes, actual_model)

    return all_codes, all_poses, statistics


def compute_code_statistics(all_codes, model):
    """
    Compute detailed statistics about code usage.

    Args:
        all_codes: Dictionary with codes for each level
        model: VQ-VAE model (to get codebook sizes)

    Returns:
        statistics: Dictionary with usage stats per level
    """
    statistics = {}

    for i, level_name in enumerate(all_codes):
        level_codes = all_codes[level_name].flatten()
        n_embed = model.quantizes[i].n_embed

        # Get unique codes and counts
        unique_codes, counts = np.unique(level_codes, return_counts=True)

        # Sort by frequency
        sort_idx = np.argsort(counts)[::-1]
        unique_codes = unique_codes[sort_idx]
        counts = counts[sort_idx]

        # Find unused codes
        used_set = set(unique_codes)
        all_codes_set = set(range(n_embed))
        unused_codes = sorted(list(all_codes_set - used_set))

        statistics[level_name] = {
            'codebook_size': n_embed,
            'n_unique': len(unique_codes),
            'usage_rate': len(unique_codes) / n_embed,
            'unique_codes': unique_codes,
            'counts': counts,
            'frequencies': counts / len(level_codes),
            'unused_codes': np.array(unused_codes),
            'total_codes': len(level_codes)
        }

        # Print statistics
        print(f"\n{level_name}:")
        print(f"  Codebook size: {n_embed}")
        print(f"  Unique codes used: {len(unique_codes)} / {n_embed} ({statistics[level_name]['usage_rate']*100:.1f}%)")

        if unused_codes:
            print(f"  ⚠️  WARNING: {len(unused_codes)} codes never used (codebook collapse!)")
            print(f"      Unused codes: {unused_codes[:20]}{'...' if len(unused_codes) > 20 else ''}")
        else:
            print(f"  ✓ All codes are used!")

        # Show most frequent codes
        print(f"  Top 10 most frequent codes:")
        for j in range(min(10, len(unique_codes))):
            code = unique_codes[j]
            count = counts[j]
            freq = statistics[level_name]['frequencies'][j]
            print(f"    Code {code:3d}: {count:8d} times ({freq*100:5.2f}%)")

    return statistics


def save_codes_dataset(output_path, all_codes, all_poses, statistics):
    """
    Save codes and statistics to file.

    Args:
        output_path: Path to save .npz file
        all_codes: Dictionary with codes for each level
        all_poses: Original poses (for reference)
        statistics: Usage statistics
    """
    print(f"\nSaving to {output_path}...")

    # Prepare data for saving
    save_dict = {}

    # Save codes
    for level_name, codes in all_codes.items():
        save_dict[f'codes_{level_name}'] = codes

    # Save poses
    save_dict['poses'] = all_poses

    # Save statistics (flatten nested dicts)
    for level_name, stats in statistics.items():
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                save_dict[f'stats_{level_name}_{key}'] = value
            elif isinstance(value, (int, float)):
                save_dict[f'stats_{level_name}_{key}'] = np.array([value])

    # Save to npz
    np.savez_compressed(output_path, **save_dict)

    print(f"✓ Saved successfully!")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Extract codes from entire dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained VQ-VAE checkpoint (default: latest checkpoint)')
    parser.add_argument('--output', type=str, default='codes_dataset.npz',
                       help='Output path for extracted codes')
    parser.add_argument('--data-path', type=str, default='BEAT2_joints_vertices',
                       help='Path to BEAT2 dataset')
    parser.add_argument('--language', type=str, default='english',
                       help='Language subset to use')
    parser.add_argument('--sequence-length', type=int, default=25,
                       help='Sequence length to extract')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum batches to process (None = all)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    args = parser.parse_args()

    print("="*60)
    print("EXTRACTING VQ-VAE CODES FROM DATASET")
    print("="*60)

    # Find checkpoint if not specified
    if args.checkpoint is None:
        print("\n0. Finding latest checkpoint...")
        args.checkpoint = find_latest_checkpoint(dataset_name='beat2_poses', run_num=0, folder_name='vqvae')
        if args.checkpoint is None:
            print("   ✗ No checkpoints found!")
            print("   Please train a model first or specify --checkpoint manually")
            return
        print(f"   ✓ Found latest checkpoint: {args.checkpoint}")

    # Load model
    print("\n1. Loading model...")
    model = model_object_parser('beat2_poses', 0, 'vqvae')

    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Extract state dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.eval()
    print(f"   ✓ Model loaded from: {args.checkpoint}")

    # Load dataset
    print("\n2. Loading dataset...")
    loader = get_beat_pose_loader(
        data_path=args.data_path,
        language=args.language,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        shuffle=False,  # Keep order for reproducibility
        num_workers=args.num_workers,
        use_axis_angle=True
    )
    print(f"   ✓ Dataset loaded: {len(loader)} batches")
    if args.max_batches:
        print(f"   ℹ  Processing only first {args.max_batches} batches")

    # Extract codes
    print("\n3. Extracting codes...")
    all_codes, all_poses, statistics = extract_all_codes(
        model, loader, args.device, args.max_batches
    )

    # Save results
    print("\n4. Saving results...")
    save_codes_dataset(args.output, all_codes, all_poses, statistics)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total poses processed: {all_poses.shape[0]}")
    print(f"Sequence length: {all_poses.shape[1]}")
    print(f"Pose dimensions: {all_poses.shape[2]}")
    print(f"\nCodes extracted:")
    for level_name in all_codes:
        print(f"  {level_name}: {all_codes[level_name].shape}")

    print(f"\nOutput file: {args.output}")
    print("\nNext steps:")
    print("  1. Use these codes to train a PixelSNAIL prior model")
    print("  2. Check for codebook collapse (unused codes)")
    print("  3. Analyze code frequency distributions")

    print("\nExample PixelSNAIL training (TODO):")
    print(f"  python train_pixelsnail.py --codes {args.output} --level 1")


if __name__ == '__main__':
    main()
