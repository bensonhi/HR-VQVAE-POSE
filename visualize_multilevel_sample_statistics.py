"""
Visualize multi-level VQ-VAE samples to compare reconstruction quality across levels.

This script loads the runtime samples generated during training and visualizes
the reconstruction from each quantization level, allowing you to see:
- What each level learns
- How much improvement each level adds
- The progressive refinement from Level 1 → Level 2 → Level 3

Usage:
    python visualize_multilevel_samples.py --sample-file checkpoint/beat2_poses/0/vqvae/runtime_samples/000.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# SMPLX pose parameter structure (165 dimensions total)
SMPLX_BODY_PARTS = {
    'Global Orient': (0, 3),        # Root orientation
    'Body': (3, 66),                # Body joints (21 joints * 3)
    'Jaw': (66, 69),                # Jaw
    'Left Eye': (69, 72),           # Left eye
    'Right Eye': (72, 75),          # Right eye
    'Left Hand': (75, 120),         # Left hand (15 joints * 3)
    'Right Hand': (120, 165),       # Right hand (15 joints * 3)
}


def compute_reconstruction_error(original, reconstructed):
    """Compute MSE between original and reconstructed poses."""
    return np.mean((original - reconstructed) ** 2)


def compute_bodypart_errors(original, reconstructed, body_parts=SMPLX_BODY_PARTS):
    """
    Compute MSE for each body part separately.

    Args:
        original: (batch, seq_len, 165) original poses
        reconstructed: (batch, seq_len, 165) reconstructed poses
        body_parts: Dictionary mapping body part names to (start, end) indices

    Returns:
        Dictionary mapping body part names to MSE values
    """
    errors = {}
    for part_name, (start, end) in body_parts.items():
        orig_part = original[:, :, start:end]
        recon_part = reconstructed[:, :, start:end]
        errors[part_name] = np.mean((orig_part - recon_part) ** 2)
    return errors


def visualize_multilevel_comparison(sample_file):
    """
    Visualize the reconstruction quality across different levels.

    Args:
        sample_file: Path to .npz file containing samples
    """
    # Load the samples
    data = np.load(sample_file)

    print(f"\nLoaded: {sample_file}")
    print(f"Available keys: {list(data.keys())}\n")

    # Get original and full reconstruction
    original = data['original']
    full_recon = data['reconstructed']

    # Detect number of levels
    levels = []
    for key in data.keys():
        if key.startswith('reconstructed_level_'):
            level_num = int(key.split('_')[-1])
            levels.append(level_num)

    levels = sorted(levels)

    if not levels:
        print("No per-level reconstructions found. Make sure you're using the updated training code.")
        return

    print(f"Found {len(levels)} levels: {levels}")
    print(f"Original shape: {original.shape}")
    print(f"Full reconstruction shape: {full_recon.shape}\n")

    # Compute errors for each level
    print("Reconstruction Errors (MSE):")
    print("-" * 60)

    errors = {}
    for level in levels:
        level_recon = data[f'reconstructed_level_{level}']
        error = compute_reconstruction_error(original, level_recon)
        errors[level] = error
        print(f"  Level {level}: {error:.6f}")

    full_error = compute_reconstruction_error(original, full_recon)
    print(f"  Full ({max(levels)} levels): {full_error:.6f}")
    print("-" * 60)

    # Compute improvement between consecutive levels
    print("\nImprovement from Previous Level:")
    print("-" * 60)
    for i, level in enumerate(levels):
        if i == 0:
            print(f"  Level {level}: baseline")
        else:
            prev_error = errors[levels[i-1]]
            curr_error = errors[level]
            improvement = prev_error - curr_error
            improvement_pct = (improvement / prev_error) * 100
            print(f"  Level {level}: {improvement:.6f} ({improvement_pct:.2f}% improvement)")
    print("-" * 60)

    # Select a sample to visualize (first sample, first frame)
    sample_idx = 0
    frame_idx = 0

    # Create visualization
    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Error comparison across levels
    ax1 = plt.subplot(2, 2, 1)
    level_labels = [f"L{l}" for l in levels]
    level_errors = [errors[l] for l in levels]

    bars = ax1.bar(level_labels, level_errors, color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(levels)])
    ax1.axhline(y=full_error, color='green', linestyle='--', label=f'Full Model ({full_error:.6f})')
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Quantization Level')
    ax1.set_title('Reconstruction Error by Level')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, error in zip(bars, level_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.6f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Improvement from previous level
    if len(levels) > 1:
        ax2 = plt.subplot(2, 2, 2)
        improvements = [0]  # First level has no improvement
        for i in range(1, len(levels)):
            prev_error = errors[levels[i-1]]
            curr_error = errors[levels[i]]
            improvement_pct = ((prev_error - curr_error) / prev_error) * 100
            improvements.append(improvement_pct)

        bars2 = ax2.bar(level_labels, improvements, color=['#95E1D3', '#F38181', '#AA96DA'][:len(levels)])
        ax2.set_ylabel('Improvement (%)')
        ax2.set_xlabel('Quantization Level')
        ax2.set_title('Relative Improvement from Previous Level')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, imp in zip(bars2, improvements):
            if imp > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{imp:.1f}%',
                        ha='center', va='bottom', fontsize=9)

    # Compute body part errors for each level
    bodypart_errors_by_level = {}
    print("\nBody Part Errors by Level:")
    print("-" * 80)

    for level in levels:
        level_recon = data[f'reconstructed_level_{level}']
        bodypart_errors = compute_bodypart_errors(original, level_recon)
        bodypart_errors_by_level[level] = bodypart_errors

        print(f"\nLevel {level}:")
        for part_name, error in bodypart_errors.items():
            print(f"  {part_name:15s}: {error:.6f}")

    print("-" * 80)

    # Plot 3: MSE by body part for each level (grouped bar chart)
    ax3 = plt.subplot(2, 2, 3)

    body_part_names = list(SMPLX_BODY_PARTS.keys())
    x = np.arange(len(body_part_names))
    width = 0.8 / len(levels)  # Width of bars

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']
    for idx, level in enumerate(levels):
        errors_list = [bodypart_errors_by_level[level][part] for part in body_part_names]
        offset = (idx - len(levels)/2 + 0.5) * width
        bars = ax3.bar(x + offset, errors_list, width, label=f'Level {level}',
                      color=colors[idx % len(colors)], alpha=0.8)

    ax3.set_ylabel('MSE')
    ax3.set_xlabel('Body Part')
    ax3.set_title('Reconstruction Error by Body Part')
    ax3.set_xticks(x)
    ax3.set_xticklabels(body_part_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Improvement in body part errors from Level 1 to final level
    ax4 = plt.subplot(2, 2, 4)

    if len(levels) > 1:
        # Compare first level vs last level
        first_level = levels[0]
        last_level = levels[-1]

        first_errors = [bodypart_errors_by_level[first_level][part] for part in body_part_names]
        last_errors = [bodypart_errors_by_level[last_level][part] for part in body_part_names]

        # Compute percentage improvement
        improvements = []
        for first_err, last_err in zip(first_errors, last_errors):
            if first_err > 0:
                improvement_pct = ((first_err - last_err) / first_err) * 100
            else:
                improvement_pct = 0
            improvements.append(improvement_pct)

        # Color bars based on improvement (green=good, red=worse)
        bar_colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]

        bars = ax4.bar(x, improvements, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_ylabel('Improvement (%)')
        ax4.set_xlabel('Body Part')
        ax4.set_title(f'Improvement from Level {first_level} to Level {last_level}')
        ax4.set_xticks(x)
        ax4.set_xticklabels(body_part_names, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            label_y = height + (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.01
            if height < 0:
                label_y = height - (ax4.get_ylim()[1] - ax4.get_ylim()[0]) * 0.03
            ax4.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{imp:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')
    else:
        # If only one level, show the errors directly
        errors_list = [bodypart_errors_by_level[levels[0]][part] for part in body_part_names]
        ax4.bar(x, errors_list, color='#4ECDC4', alpha=0.7)
        ax4.set_ylabel('MSE')
        ax4.set_xlabel('Body Part')
        ax4.set_title(f'Body Part Errors (Level {levels[0]})')
        ax4.set_xticks(x)
        ax4.set_xticklabels(body_part_names, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save the figure
    output_path = Path(sample_file).parent / f"{Path(sample_file).stem}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-level VQ-VAE samples')
    parser.add_argument('--sample-file', type=str,
                       default='checkpoint/beat2_poses/0/vqvae/runtime_samples/00002_new_structure_loss.npz',
                       help='Path to sample .npz file')

    args = parser.parse_args()

    sample_file = Path(args.sample_file)

    if not sample_file.exists():
        print(f"Error: Sample file not found: {sample_file}")
        print("\nMake sure to train the model first to generate runtime samples.")
        print("Runtime samples are saved in: checkpoint/<dataset>/<run>/vqvae/runtime_samples/")
        return

    visualize_multilevel_comparison(sample_file)


if __name__ == '__main__':
    main()