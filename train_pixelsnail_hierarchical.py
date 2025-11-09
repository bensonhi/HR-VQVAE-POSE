"""
Train PixelSNAIL prior models for hierarchical VQ-VAE.

This script trains autoregressive prior models (PixelSNAIL) on the discrete codes
extracted from the VQ-VAE. The prior learns the distribution of codes, enabling
high-quality unconditional generation.

For a 3-level VQ-VAE, we need to train 3 PixelSNAIL models:
  1. Top level (Level 1): p(z1) - unconditional
  2. Middle level (Level 2): p(z2|z1) - conditional on Level 1
  3. Bottom level (Level 3): p(z3|z2) - conditional on Level 2

Usage:
    # Train top level (unconditional)
    python train_pixelsnail_hierarchical.py --level 1 --codes codes_dataset.npz

    # Train middle level (conditional on top)
    python train_pixelsnail_hierarchical.py --level 2 --codes codes_dataset.npz

    # Train bottom level (conditional on middle)
    python train_pixelsnail_hierarchical.py --level 3 --codes codes_dataset.npz
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from m_pixelsnail import PixelSNAIL


class CodeDataset(Dataset):
    """Dataset of discrete codes for PixelSNAIL training."""

    def __init__(self, codes_file, level, seq_height=5, seq_width=5):
        """
        Args:
            codes_file: Path to .npz file with extracted codes
            level: Which level to train (1, 2, or 3)
            seq_height: Height to reshape sequence (height * width should = sequence_length)
            seq_width: Width to reshape sequence
        """
        self.level = level
        self.seq_height = seq_height
        self.seq_width = seq_width

        # Load codes
        data = np.load(codes_file)

        # Get codes for this level
        self.codes = data[f'codes_level_{level}']  # Shape: (N, seq_len)

        # Get conditioning codes (from previous level) if not level 1
        self.condition_codes = None
        if level > 1:
            self.condition_codes = data[f'codes_level_{level-1}']

        print(f"Loaded {len(self.codes)} code sequences for level {level}")
        print(f"  Code shape: {self.codes.shape}")
        if self.condition_codes is not None:
            print(f"  Condition shape: {self.condition_codes.shape}")

        # Verify dimensions
        seq_len = self.codes.shape[1]
        if seq_height * seq_width != seq_len:
            # Auto-adjust to make it work
            import math
            # Try to make it as square as possible
            seq_width = int(math.sqrt(seq_len))
            while seq_len % seq_width != 0:
                seq_width += 1
            seq_height = seq_len // seq_width
            self.seq_height = seq_height
            self.seq_width = seq_width
            print(f"  Auto-adjusted shape to: {seq_height}x{seq_width}")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        # Get codes for this sample
        codes = torch.LongTensor(self.codes[idx])  # (seq_len,)

        # Reshape to 2D for PixelSNAIL: (seq_len,) -> (1, height, width)
        codes_2d = codes.view(1, self.seq_height, self.seq_width)

        # Get conditioning codes if available
        if self.condition_codes is not None:
            cond = torch.LongTensor(self.condition_codes[idx])
            cond_2d = cond.view(1, self.seq_height, self.seq_width)
            return codes_2d, cond_2d
        else:
            # Return dummy condition for unconditional case
            return codes_2d, torch.zeros(1, 1, 1, dtype=torch.long)


def train_epoch(epoch, loader, model, optimizer, device, level):
    """Train for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_acc = 0

    pbar = tqdm(loader, desc=f'Level {level} Epoch {epoch+1}')

    for batch_idx, (codes, cond) in enumerate(pbar):
        codes = codes.to(device)
        cond = cond.to(device) if cond.shape[1] > 1 else None

        optimizer.zero_grad()

        # Forward pass
        # PixelSNAIL expects (batch, n_class, height, width)
        # We have (batch, 1, height, width) with integer codes

        # Convert codes to one-hot for input
        batch, _, height, width = codes.shape
        codes_flat = codes.view(-1)  # Flatten for one-hot

        # Get number of classes (codebook size)
        n_class = model.n_class

        # One-hot encode: (batch*height*width,) -> (batch*height*width, n_class)
        codes_onehot = F.one_hot(codes_flat, n_class).float()
        # Reshape: (batch*height*width, n_class) -> (batch, height, width, n_class) -> (batch, n_class, height, width)
        codes_onehot = codes_onehot.view(batch, height, width, n_class).permute(0, 3, 1, 2)

        # Process condition similarly if it exists
        cond_onehot = None
        if cond is not None:
            cond_flat = cond.view(-1)
            cond_onehot = F.one_hot(cond_flat, n_class).float()
            cond_onehot = cond_onehot.view(batch, height, width, n_class).permute(0, 3, 1, 2)

        # Forward
        out, _ = model(codes_onehot, condition=cond_onehot)

        # Loss: predict next code autoregressively
        # out shape: (batch, n_class, height, width)
        # codes shape: (batch, 1, height, width)
        target = codes.squeeze(1)  # (batch, height, width)

        loss = criterion(out, target)

        # Backward
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, pred = out.max(1)  # (batch, height, width)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        total_loss += loss.item()
        total_acc += accuracy.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy.item():.4f}'
        })

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Train PixelSNAIL for hierarchical VQ-VAE')

    # Data
    parser.add_argument('--codes', type=str, default='codes_dataset.npz',
                       help='Path to extracted codes file')
    parser.add_argument('--level', type=int, required=True, choices=[1, 2, 3],
                       help='Which level to train (1=top, 2=middle, 3=bottom)')

    # Model architecture
    parser.add_argument('--channel', type=int, default=256,
                       help='Number of channels in PixelSNAIL')
    parser.add_argument('--n-block', type=int, default=4,
                       help='Number of PixelSNAIL blocks')
    parser.add_argument('--n-res-block', type=int, default=4,
                       help='Number of residual blocks per PixelSNAIL block')
    parser.add_argument('--res-channel', type=int, default=256,
                       help='Residual channel size')
    parser.add_argument('--kernel-size', type=int, default=5,
                       help='Kernel size')
    parser.add_argument('--attention', action='store_true',
                       help='Use attention in PixelSNAIL')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')

    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    # Sequence shape
    parser.add_argument('--seq-height', type=int, default=5,
                       help='Height of sequence (height * width = seq_len)')
    parser.add_argument('--seq-width', type=int, default=5,
                       help='Width of sequence')

    # Output
    parser.add_argument('--save-dir', type=str, default='checkpoint/beat2_poses/0/pixelsnail',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    print("="*60)
    print(f"TRAINING PIXELSNAIL - LEVEL {args.level}")
    print("="*60)

    # Create save directory
    level_dir = os.path.join(args.save_dir, f'level_{args.level}')
    os.makedirs(level_dir, exist_ok=True)

    # Load codebook sizes from extracted codes
    data = np.load(args.codes)
    codebook_sizes = {
        1: 8,    # Level 1 codebook size
        2: 64,   # Level 2 codebook size
        3: 512   # Level 3 codebook size
    }
    n_class = codebook_sizes[args.level]

    print(f"\n1. Creating dataset...")
    print(f"   Level: {args.level}")
    print(f"   Codebook size: {n_class}")

    # Create dataset
    dataset = CodeDataset(
        args.codes,
        level=args.level,
        seq_height=args.seq_height,
        seq_width=args.seq_width
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n2. Creating PixelSNAIL model...")

    # Create model
    model = PixelSNAIL(
        shape=[args.seq_height, args.seq_width],
        n_class=n_class,
        channel=args.channel,
        kernel_size=args.kernel_size,
        n_block=args.n_block,
        n_res_block=args.n_res_block,
        res_channel=args.res_channel,
        attention=args.attention,
        dropout=args.dropout,
        # Conditional parameters (for levels 2 and 3)
        cond_channel=n_class if args.level > 1 else 0,
        n_cond_res_block=args.n_res_block if args.level > 1 else 0,
        cond_res_channel=args.res_channel if args.level > 1 else 0,
        cond_res_kernel=3,
        n_out_res_block=0
    )

    model = model.to(args.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler (optional)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(level_dir, 'logs'))

    print(f"\n3. Training for {args.epochs} epochs...")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Save directory: {level_dir}")

    best_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        avg_loss, avg_acc = train_epoch(epoch, loader, model, optimizer, args.device, args.level)

        # Learning rate schedule
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log to tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {avg_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % args.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_acc,
                'args': vars(args)
            }

            # Save regular checkpoint
            ckpt_path = os.path.join(level_dir, f'{str(epoch+1).zfill(3)}.pt')
            torch.save(checkpoint, ckpt_path)
            print(f"  Saved: {ckpt_path}")

            # Save best checkpoint
            if is_best:
                best_path = os.path.join(level_dir, 'best.pt')
                torch.save(checkpoint, best_path)
                print(f"  New best model! Loss: {best_loss:.4f}")

    writer.close()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {level_dir}")

    print("\nNext steps:")
    if args.level < 3:
        print(f"  1. Train level {args.level + 1}:")
        print(f"     python train_pixelsnail_hierarchical.py --level {args.level + 1} --codes {args.codes}")
    else:
        print("  1. All levels trained! You can now use the prior for sampling.")
        print("  2. See sample_with_prior.py for generation examples")


if __name__ == '__main__':
    main()
