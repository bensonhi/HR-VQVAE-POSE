"""
Pre-extract and cache Wav2Vec 2.0 features for BEAT2 audio files.

Extracts hidden states from facebook/wav2vec2-base-960h (768-dim at 50Hz),
resamples to 30fps to match pose frame rate, and saves as .npy files.

Usage:
    python extract_wav2vec_features.py --beat2-dir BEAT2 --language english --device cuda
    python extract_wav2vec_features.py --beat2-dir BEAT2 --language english --max-files 5  # test run
"""

import os
import argparse
import glob
import numpy as np
import torch
import torchaudio
from tqdm import tqdm


def resample_features(features, target_length):
    """
    Resample features from 50Hz (Wav2Vec 2.0 output rate) to match target frame count.

    Args:
        features: (T_audio, 768) numpy array at ~50Hz
        target_length: desired number of frames (at 30fps)

    Returns:
        (target_length, 768) numpy array
    """
    # Use linear interpolation via torch
    # features: (T, D) -> (1, D, T) for interpolate -> (1, D, target_length) -> (target_length, D)
    feat_tensor = torch.from_numpy(features).float().unsqueeze(0).permute(0, 2, 1)  # (1, D, T)
    resampled = torch.nn.functional.interpolate(
        feat_tensor, size=target_length, mode='linear', align_corners=True
    )
    return resampled.squeeze(0).permute(1, 0).numpy()  # (target_length, D)


def extract_wav2vec_features(
    beat2_dir: str,
    language: str = 'english',
    device: str = 'cuda',
    max_files: int = -1,
):
    """
    Extract Wav2Vec 2.0 features for all audio files in a BEAT2 language subset.

    Args:
        beat2_dir: Path to BEAT2 root directory
        language: Language subset
        device: Device for model inference
        max_files: Maximum files to process (-1 for all)
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    lang_folders = {
        'english': 'beat_english_v2.0.0',
        'chinese': 'beat_chinese_v2.0.0',
        'spanish': 'beat_spanish_v2.0.0',
        'japanese': 'beat_japanese_v2.0.0',
    }
    lang_folder = lang_folders.get(language, f'beat_{language}_v2.0.0')
    lang_path = os.path.join(beat2_dir, lang_folder)

    wave_dir = os.path.join(lang_path, 'wave16k')
    pose_dir = os.path.join(lang_path, 'smplxflame_30')
    output_dir = os.path.join(lang_path, 'wav2vec_30')
    os.makedirs(output_dir, exist_ok=True)

    # Find all wav files
    wav_files = sorted(glob.glob(os.path.join(wave_dir, '*.wav')))
    if max_files > 0:
        wav_files = wav_files[:max_files]

    print(f"Found {len(wav_files)} wav files in {wave_dir}")
    print(f"Output directory: {output_dir}")

    # Load Wav2Vec 2.0 model
    print("Loading Wav2Vec 2.0 model (facebook/wav2vec2-base-960h)...")
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    model = model.to(device)
    model.eval()
    print("Model loaded.")

    skipped = 0
    processed = 0

    for wav_path in tqdm(wav_files, desc="Extracting features"):
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        output_path = os.path.join(output_dir, f'{basename}.npy')

        # Skip if already extracted
        if os.path.exists(output_path):
            processed += 1
            continue

        # Check for matching pose file to get target frame count
        pose_path = os.path.join(pose_dir, f'{basename}.npz')
        if not os.path.exists(pose_path):
            print(f"  Skipping {basename}: no matching pose file")
            skipped += 1
            continue

        try:
            # Get target frame count from pose data
            pose_data = np.load(pose_path)
            target_frames = pose_data['poses'].shape[0]

            # Load audio at 16kHz
            waveform, sample_rate = torchaudio.load(wav_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Process through Wav2Vec 2.0
            # Process in chunks to avoid OOM for long audio
            waveform_np = waveform.squeeze(0).numpy()
            inputs = processor(waveform_np, sampling_rate=16000, return_tensors='pt', padding=True)
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                # Extract hidden states (last hidden state)
                outputs = model(input_values)
                features = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (T_audio, 768)

            # Resample from ~50Hz to 30fps (target_frames)
            resampled = resample_features(features, target_frames)

            # Verify shape
            assert resampled.shape == (target_frames, 768), \
                f"Shape mismatch: {resampled.shape} vs expected ({target_frames}, 768)"

            # Save
            np.save(output_path, resampled.astype(np.float32))
            processed += 1

        except Exception as e:
            print(f"  Error processing {basename}: {e}")
            skipped += 1
            continue

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Features saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Wav2Vec 2.0 features for BEAT2')
    parser.add_argument('--beat2-dir', type=str, default='BEAT2', help='Path to BEAT2 directory')
    parser.add_argument('--language', type=str, default='english', help='Language subset')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--max-files', type=int, default=-1, help='Max files to process (-1 for all)')
    args = parser.parse_args()

    extract_wav2vec_features(
        beat2_dir=args.beat2_dir,
        language=args.language,
        device=args.device,
        max_files=args.max_files,
    )
