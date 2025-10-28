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


def compute_reconstruction_error(original, reconstructed):
    """Compute MSE between original and reconstructed poses."""
    return np.mean((original - reconstructed) ** 2)


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

    # Plot 3: Pose comparison for a few dimensions
    ax3 = plt.subplot(2, 2, 3)

    # Select first 10 pose dimensions to visualize
    n_dims = min(10, original.shape[-1])
    x = np.arange(n_dims)

    orig_vals = original[sample_idx, frame_idx, :n_dims]
    ax3.plot(x, orig_vals, 'ko-', label='Original', linewidth=2, markersize=8)

    colors = ['red', 'orange', 'blue', 'green']
    for idx, level in enumerate(levels):
        level_recon = data[f'reconstructed_level_{level}']
        recon_vals = level_recon[sample_idx, frame_idx, :n_dims]
        ax3.plot(x, recon_vals, 'o--', label=f'Level {level}',
                color=colors[idx % len(colors)], alpha=0.7, markersize=6)

    ax3.set_xlabel('Pose Dimension')
    ax3.set_ylabel('Value')
    ax3.set_title(f'Pose Reconstruction (First {n_dims} dimensions)')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Error heatmap across pose dimensions
    ax4 = plt.subplot(2, 2, 4)

    # Compute per-dimension errors for each level
    n_dims_heatmap = min(50, original.shape[-1])  # Show first 50 dimensions
    error_matrix = []

    for level in levels:
        level_recon = data[f'reconstructed_level_{level}']
        # Average over batch and sequence, compute error per dimension
        dim_errors = np.mean((original[:, :, :n_dims_heatmap] -
                             level_recon[:, :, :n_dims_heatmap]) ** 2, axis=(0, 1))
        error_matrix.append(dim_errors)

    error_matrix = np.array(error_matrix)

    im = ax4.imshow(error_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax4.set_yticks(range(len(levels)))
    ax4.set_yticklabels([f'Level {l}' for l in levels])
    ax4.set_xlabel('Pose Dimension')
    ax4.set_ylabel('Quantization Level')
    ax4.set_title(f'Error per Dimension (First {n_dims_heatmap} dims)')
    plt.colorbar(im, ax=ax4, label='MSE')

    plt.tight_layout()

    # Save the figure
    output_path = Path(sample_file).parent / f"{Path(sample_file).stem}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize multi-level VQ-VAE samples')
    parser.add_argument('--sample-file', type=str,
                       default='checkpoint/beat2_poses/0/vqvae/runtime_samples/000.npz',
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