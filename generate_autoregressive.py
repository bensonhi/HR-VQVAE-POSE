"""
Autoregressive motion generation from a full test recording.

Loads a complete recording (poses + audio + sem labels), splits it into
gesture-type-consistent chunks, and generates motion autoregressively
with anchor-frame bridging between chunks.  Renders an SMPLX avatar
MP4 with the original audio.

Usage:
  # Specific pose file
  python generate_autoregressive.py \
      --checkpoint checkpoint/beat2_poses/0/vae/191.pt \
      --pose-file BEAT2/beat_english_v2.0.0/smplxflame_30/1_wayne_0_1_1.npz

  # Random test file
  python generate_autoregressive.py \
      --checkpoint checkpoint/beat2_poses/0/vae/191.pt
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from m_util import conf_parser, create_model_object, load_checkpoint
from m_beat_dataset import parse_sem_file, BEAT_LABELS
from m_smplx_layer import SMPLXLayer
from visualize_motion_animation import (
    poses_to_vertices, render_avatar_frames, write_mp4, _cut_wav,
)

FPS = 30
MIN_CHUNK_FRAMES = 5


# ============================================================================
# Sem file parsing (handles BEAT2 actual format: label \t start \t end ...)
# ============================================================================

def parse_sem_file_beat2(sem_path, total_frames, fps=30):
    """Parse BEAT2 .txt sem file (label\\tstart\\tend\\tduration\\tweight...).

    Falls back to parse_sem_file() if format doesn't match.
    Returns (valid_regions, gesture_labels) same as parse_sem_file().
    """
    gesture_labels = np.zeros(total_frames, dtype=np.int64)
    need_cut_mask = np.zeros(total_frames, dtype=bool)

    if not os.path.exists(sem_path):
        return [(0, total_frames)], gesture_labels

    parsed_any = False
    with open(sem_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue

            # Try BEAT2 format: label \t start \t end ...
            label = parts[0].strip()
            try:
                start_time = float(parts[1])
                end_time = float(parts[2])
            except (ValueError, IndexError):
                # Fall back to original format: start \t end \t label
                try:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    label = parts[2].strip()
                except (ValueError, IndexError):
                    continue

            start_frame = int(start_time * fps)
            end_frame = min(int(end_time * fps), total_frames)
            if start_frame >= total_frames or start_frame >= end_frame:
                continue

            parsed_any = True
            if label == 'need_cut':
                need_cut_mask[start_frame:end_frame] = True
            elif label in BEAT_LABELS:
                gesture_labels[start_frame:end_frame] = 0
            else:
                gesture_labels[start_frame:end_frame] = 1  # semantic

    if not parsed_any:
        return [(0, total_frames)], gesture_labels

    # Build valid regions (contiguous non-need_cut)
    valid_regions = []
    in_region = False
    region_start = 0
    for i in range(total_frames):
        if not need_cut_mask[i]:
            if not in_region:
                region_start = i
                in_region = True
        else:
            if in_region:
                valid_regions.append((region_start, i))
                in_region = False
    if in_region:
        valid_regions.append((region_start, total_frames))

    return valid_regions, gesture_labels


# ============================================================================
# Loading
# ============================================================================

def load_recording(pose_path, language='english', data_root='BEAT2'):
    """Load a full recording: poses, audio features, sem labels, wav path.

    Returns dict with keys: poses, audio, gesture_labels, valid_regions,
    speaker_id, wav_path, basename, total_frames.
    """
    basename = os.path.splitext(os.path.basename(pose_path))[0]
    lang_folders = {
        'english': 'beat_english_v2.0.0',
        'chinese': 'beat_chinese_v2.0.0',
        'spanish': 'beat_spanish_v2.0.0',
        'japanese': 'beat_japanese_v2.0.0',
    }
    lang_folder = lang_folders.get(language, f'beat_{language}_v2.0.0')
    lang_path = os.path.join(data_root, lang_folder)

    # Load poses
    data = np.load(pose_path)
    poses = data['poses'].astype(np.float32)  # (T, 165)
    total_frames = poses.shape[0]

    # Load audio features
    audio_path = os.path.join(lang_path, 'wav2vec_30', f'{basename}.npy')
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio features not found: {audio_path}")
    audio = np.load(audio_path).astype(np.float32)  # (T, 768)

    # Align lengths
    min_len = min(total_frames, audio.shape[0])
    poses = poses[:min_len]
    audio = audio[:min_len]
    total_frames = min_len

    # Load sem labels (try .sem then .txt)
    sem_path = os.path.join(lang_path, 'sem', f'{basename}.sem')
    if not os.path.exists(sem_path):
        sem_path = os.path.join(lang_path, 'sem', f'{basename}.txt')
    valid_regions, gesture_labels = parse_sem_file_beat2(sem_path, total_frames)

    # Wav path
    wav_path = os.path.join(lang_path, 'wave16k', f'{basename}.wav')
    if not os.path.exists(wav_path):
        wav_path = None

    # Speaker ID from filename
    speaker_id = int(basename.split('_')[0])

    return {
        'poses': poses,
        'audio': audio,
        'gesture_labels': gesture_labels,
        'valid_regions': valid_regions,
        'speaker_id': speaker_id,
        'wav_path': wav_path,
        'basename': basename,
        'total_frames': total_frames,
    }


def pick_random_test_file(data_root='BEAT2', language='english'):
    """Pick a random pose file from the test split."""
    lang_folders = {
        'english': 'beat_english_v2.0.0',
    }
    lang_folder = lang_folders.get(language, f'beat_{language}_v2.0.0')
    lang_path = os.path.join(data_root, lang_folder)

    csv_path = os.path.join(lang_path, 'train_test_split.csv')
    test_ids = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['type'] == 'test':
                test_ids.append(row['id'])

    chosen = random.choice(test_ids)
    pose_path = os.path.join(lang_path, 'smplxflame_30', f'{chosen}.npz')
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    return pose_path


# ============================================================================
# Chunking
# ============================================================================

def build_chunks(recording, max_chunk_length=150):
    """Split recording into gesture-type-consistent chunks.

    Each chunk is a dict: {start, end, gesture_type, length}.
    Chunks longer than max_chunk_length are split into sub-chunks.
    Chunks shorter than MIN_CHUNK_FRAMES are skipped.
    """
    valid_regions = recording['valid_regions']
    gesture_labels = recording['gesture_labels']
    chunks = []

    for reg_start, reg_end in valid_regions:
        # Segment this region by gesture type transitions
        segments = []
        seg_start = reg_start
        current_type = int(gesture_labels[reg_start])

        for i in range(reg_start + 1, reg_end):
            if int(gesture_labels[i]) != current_type:
                segments.append((seg_start, i, current_type))
                seg_start = i
                current_type = int(gesture_labels[i])
        segments.append((seg_start, reg_end, current_type))

        # Split long segments, skip short ones
        for seg_start, seg_end, gtype in segments:
            seg_len = seg_end - seg_start
            if seg_len < MIN_CHUNK_FRAMES:
                continue

            pos = seg_start
            while pos < seg_end:
                chunk_end = min(pos + max_chunk_length, seg_end)
                chunk_len = chunk_end - pos
                if chunk_len < MIN_CHUNK_FRAMES:
                    break
                chunks.append({
                    'start': pos,
                    'end': chunk_end,
                    'gesture_type': gtype,
                    'length': chunk_len,
                })
                pos = chunk_end

    return chunks


# ============================================================================
# Generation
# ============================================================================

def generate_autoregressive(model, recording, chunks, device,
                            anchor_k=30, temperature=1.0):
    """Generate motion autoregressively, chunk by chunk.

    Returns (T_total, 165) tensor of generated poses and the total frame count.
    """
    actual_model = model.module if hasattr(model, 'module') else model
    actual_model.eval()

    poses_np = recording['poses']
    audio_np = recording['audio']
    speaker_id = recording['speaker_id']
    speaker_id_tensor = torch.tensor([speaker_id], dtype=torch.long, device=device)

    generated_chunks = []
    prev_anchor = None  # (1, K, 165) from previous chunk

    for ci, chunk in enumerate(chunks):
        start, end = chunk['start'], chunk['end']
        chunk_len = chunk['length']
        gtype = chunk['gesture_type']

        # Audio for this chunk
        audio_slice = torch.FloatTensor(audio_np[start:end]).unsqueeze(0).to(device)
        # Gesture type
        gesture_type_tensor = torch.tensor([gtype], dtype=torch.long, device=device)

        # Anchor frames
        anchor_frames = None
        if anchor_k > 0 and prev_anchor is not None:
            anchor_frames = prev_anchor
        elif anchor_k > 0 and ci == 0 and start >= anchor_k:
            # First chunk: use GT preceding frames if available
            anchor_frames = torch.FloatTensor(
                poses_np[start - anchor_k:start]
            ).unsqueeze(0).to(device)

        # Generate
        with torch.no_grad():
            gen = actual_model.sample(
                1, chunk_len, device=device,
                temperature=temperature,
                gesture_type=gesture_type_tensor,
                audio_features=audio_slice,
                anchor_frames=anchor_frames,
                speaker_id=speaker_id_tensor,
            )
        chunk_poses = gen[0].cpu()  # (chunk_len, 165)
        generated_chunks.append(chunk_poses)

        # Update anchor for next chunk
        if anchor_k > 0:
            tail = chunk_poses[-anchor_k:] if chunk_poses.shape[0] >= anchor_k else chunk_poses
            prev_anchor = tail.unsqueeze(0).to(device)

        gtype_str = 'semantic' if gtype else 'beat'
        print(f"  Chunk {ci+1}/{len(chunks)}: frames {start}-{end} "
              f"({chunk_len}fr, {gtype_str})")

    all_poses = torch.cat(generated_chunks, dim=0)
    return all_poses


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Autoregressive motion generation from a full recording')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoint/beat2_poses/0/vae/best.pt')
    parser.add_argument('--pose-file', type=str, default=None,
                        help='Path to .npz pose file (random test file if omitted)')
    parser.add_argument('--chunk-length', type=int, default=150,
                        help='Max frames per generation chunk (default: 150)')
    parser.add_argument('--anchor-frames', type=int, default=30,
                        help='Anchor frames between chunks (default: 30)')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--output-dir', type=str, default='visualization_output/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Cap total output length')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-name', type=str, default='beat2_poses')
    parser.add_argument('--run-num', type=int, default=0)
    parser.add_argument('--data-root', type=str, default='BEAT2')
    parser.add_argument('--language', type=str, default='english')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Select pose file ----
    if args.pose_file is not None:
        pose_path = args.pose_file
    else:
        pose_path = pick_random_test_file(args.data_root, args.language)
        print(f"Randomly selected test file: {pose_path}")

    # ---- Load recording ----
    print(f"Loading recording: {pose_path}")
    recording = load_recording(pose_path, args.language, args.data_root)
    print(f"  Total frames: {recording['total_frames']} "
          f"({recording['total_frames']/FPS:.1f}s)")
    print(f"  Valid regions: {len(recording['valid_regions'])}")
    print(f"  Speaker ID: {recording['speaker_id']}")
    if recording['wav_path']:
        print(f"  Wav: {recording['wav_path']}")

    # ---- Build chunks ----
    chunks = build_chunks(recording, max_chunk_length=args.chunk_length)
    total_gen_frames = sum(c['length'] for c in chunks)
    n_beat = sum(1 for c in chunks if c['gesture_type'] == 0)
    n_semantic = sum(1 for c in chunks if c['gesture_type'] == 1)
    print(f"\nChunking: {len(chunks)} chunks, {total_gen_frames} frames "
          f"({total_gen_frames/FPS:.1f}s)")
    print(f"  Beat chunks: {n_beat}, Semantic chunks: {n_semantic}")

    if not chunks:
        print("No valid chunks found. Exiting.")
        return

    # ---- Load model ----
    print("\nLoading model...")
    options, _ = conf_parser(args.dataset_name, args.run_num, 'vae')
    model = create_model_object('vae', options)
    state_dict = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"  Checkpoint: {args.checkpoint}")

    # ---- Generate ----
    print(f"\nGenerating autoregressively "
          f"(chunk_len≤{args.chunk_length}, anchor={args.anchor_frames}, "
          f"temp={args.temperature})...")
    gen_poses = generate_autoregressive(
        model, recording, chunks, device,
        anchor_k=args.anchor_frames,
        temperature=args.temperature,
    )

    if args.max_frames and gen_poses.shape[0] > args.max_frames:
        gen_poses = gen_poses[:args.max_frames]

    total_frames = gen_poses.shape[0]
    print(f"\nGenerated {total_frames} frames ({total_frames/FPS:.1f}s)")

    # ---- Render ----
    print("\nLoading SMPLX layer...")
    smplx_layer = SMPLXLayer(model_path='models_smplx_v1_1/models')
    smplx_layer.to(device)
    smplx_faces = smplx_layer.smplx_model.faces_tensor.cpu().numpy()

    print("Converting poses to vertices...")
    gen_verts = poses_to_vertices(gen_poses, smplx_layer, device)

    print("Rendering frames...")
    frames = render_avatar_frames(
        [gen_verts], smplx_faces,
        ['Generated'], [(0.3, 0.6, 0.9)],
    )

    # ---- Cut audio ----
    clip_wav_path = None
    if recording['wav_path']:
        # Audio starts at the first chunk's start frame
        audio_start_frame = chunks[0]['start']
        clip_wav_path = os.path.join(args.output_dir, '.autoreg_audio_tmp.wav')
        try:
            _cut_wav(recording['wav_path'], audio_start_frame, total_frames,
                     clip_wav_path)
            print(f"  Audio clip: frame {audio_start_frame} + {total_frames}fr "
                  f"({total_frames/FPS:.1f}s)")
        except Exception as e:
            print(f"  Warning: could not cut audio ({e})")
            clip_wav_path = None

    # ---- Save MP4 ----
    basename = recording['basename']
    suffix = (f"autoreg_{basename}_t{args.temperature:.1f}"
              f"_chunk{args.chunk_length}_anchor{args.anchor_frames}")
    out_path = os.path.join(args.output_dir, f'{suffix}.mp4')
    print(f"Writing MP4: {out_path}")
    write_mp4(frames, out_path, fps=FPS, audio_path=clip_wav_path)

    # Clean up temp audio
    if clip_wav_path and os.path.exists(clip_wav_path):
        os.remove(clip_wav_path)

    print(f"\nDone! Output: {out_path}")


if __name__ == '__main__':
    main()
