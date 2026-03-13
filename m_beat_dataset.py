import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from typing import Optional, Union, List, Tuple
from tqdm import tqdm


# =============================================================================
# Gesture label parsing
# =============================================================================

# Labels that map to beat (0)
BEAT_LABELS = {'01_beat_align', '00_nogesture', 'habit'}


def parse_sem_file(sem_path: str, total_frames: int, fps: int = 30) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Parse a .sem tab-separated file into per-frame gesture labels and valid regions.

    Format expected: each line is "start_time\tend_time\tlabel" (times in seconds).

    Args:
        sem_path: Path to .sem file
        total_frames: Total number of frames in the corresponding pose sequence
        fps: Frame rate (default 30)

    Returns:
        valid_regions: list of (start_frame, end_frame) tuples excluding need_cut segments
        gesture_labels: np.array of shape (total_frames,) with 0=beat, 1=semantic
    """
    gesture_labels = np.zeros(total_frames, dtype=np.int64)  # default beat
    need_cut_mask = np.zeros(total_frames, dtype=bool)

    if not os.path.exists(sem_path):
        # No sem file: treat entire sequence as one valid region with beat label
        return [(0, total_frames)], gesture_labels

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

            if label == 'need_cut':
                need_cut_mask[start_frame:end_frame] = True
            elif label in BEAT_LABELS:
                gesture_labels[start_frame:end_frame] = 0
            else:
                gesture_labels[start_frame:end_frame] = 1  # semantic

    # Build valid regions (contiguous non-need_cut regions)
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


# =============================================================================
# Variable-length BEAT2 Dataset
# =============================================================================

class BEAT2PoseDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 language: str = 'english',
                 min_length: int = 5,
                 max_length: int = 300,
                 pose_dims: int = 165,
                 use_axis_angle: bool = True,
                 audio_dir: Optional[str] = None,
                 split: Optional[str] = None,
                 anchor_max_frames: int = 30,
                 # Legacy parameters (ignored, kept for backward compat)
                 sequence_length: int = 120,
                 stride: int = 30,
                 normalize: bool = False,
                 load_gt_geometry: bool = False,
                 compute_gt_on_fly: bool = False,
                 smplx_model_path: str = 'models_smplx_v1_1/models'):
        """
        BEAT2 Pose Dataset with variable-length clips, gesture type, and audio conditioning.

        Args:
            data_path: Path to BEAT2 directory
            language: Language subset
            min_length: Minimum clip length in frames
            max_length: Maximum clip length in frames
            pose_dims: Dimension of pose data (165 for SMPLX axis-angle)
            use_axis_angle: If True, load axis-angle poses (165D)
            audio_dir: Path to wav2vec_30 features directory. If None, auto-constructed.
            split: 'train', 'val', or 'test' to filter by BEAT2 split. None = all files.
        """
        self.data_path = data_path
        self.language = language
        self.min_length = min_length
        self.max_length = max_length
        self.pose_dims = pose_dims
        self.use_axis_angle = use_axis_angle
        self.split = split
        self.anchor_max_frames = anchor_max_frames

        # Determine the correct language folder
        lang_folders = {
            'english': 'beat_english_v2.0.0',
            'chinese': 'beat_chinese_v2.0.0',
            'spanish': 'beat_spanish_v2.0.0',
            'japanese': 'beat_japanese_v2.0.0'
        }
        lang_folder = lang_folders.get(language, f'beat_{language}_v2.0.0')
        self.lang_path = os.path.join(data_path, lang_folder)

        self.pose_dir = os.path.join(self.lang_path, 'smplxflame_30')
        self.sem_dir = os.path.join(self.lang_path, 'sem')
        self.audio_dir = audio_dir or os.path.join(self.lang_path, 'wav2vec_30')

        self.pose_files = sorted(glob.glob(os.path.join(self.pose_dir, '*.npz')))

        # Per-file metadata storage
        self.file_data = []  # list of dicts: {poses, audio, valid_regions, gesture_labels}
        self.sample_index = []  # list of (file_idx, region_idx) for sampling

        self._load_pose_sequences()

        split_str = f", split={split}" if split else ""
        print(f"Loaded {len(self.file_data)} files, {len(self.sample_index)} sample entries "
              f"from {language} BEAT2 data (variable-length [{min_length}, {max_length}]{split_str})")

    def _load_pose_sequences(self):
        """Load all pose sequences, sem labels, and audio features."""
        # Build allowed basenames from train_test_split.csv if split is specified
        allowed_basenames = None
        if self.split is not None:
            csv_path = os.path.join(self.lang_path, 'train_test_split.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Split CSV not found: {csv_path}")
            allowed_basenames = set()
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['type'] == self.split:
                        allowed_basenames.add(row['id'])
            print(f"Split '{self.split}': {len(allowed_basenames)} files allowed by CSV")

        print(f"Loading pose sequences from {len(self.pose_files)} files...")

        avg_clip_length = (self.min_length + self.max_length) / 2.0

        for file_path in tqdm(self.pose_files, desc="Loading files"):
            try:
                basename = os.path.splitext(os.path.basename(file_path))[0]

                # Skip files not in the requested split
                if allowed_basenames is not None and basename not in allowed_basenames:
                    continue

                # Load poses
                data = np.load(file_path)
                if self.use_axis_angle:
                    poses = data['poses']  # (T, 165)
                else:
                    poses = data['joints']
                    if poses.ndim > 2:
                        poses = poses.reshape(poses.shape[0], -1)

                total_frames = poses.shape[0]

                # Load audio features
                audio_path = os.path.join(self.audio_dir, f'{basename}.npy')
                if not os.path.exists(audio_path):
                    continue  # Skip files without audio features

                audio = np.load(audio_path)  # (T, 768)

                # Ensure audio and pose lengths match
                min_len = min(total_frames, audio.shape[0])
                poses = poses[:min_len]
                audio = audio[:min_len]
                total_frames = min_len

                if total_frames < self.min_length:
                    continue

                # Load sem file and parse gesture labels
                sem_path = os.path.join(self.sem_dir, f'{basename}.sem')
                valid_regions, gesture_labels = parse_sem_file(sem_path, total_frames)

                # Filter regions shorter than min_length
                valid_regions = [(s, e) for s, e in valid_regions if (e - s) >= self.min_length]

                if not valid_regions:
                    continue

                # Extract speaker ID from filename (e.g. "1_wayne_0_1_1.npz" → 1)
                speaker_id = int(basename.split('_')[0])

                file_idx = len(self.file_data)
                self.file_data.append({
                    'poses': poses.astype(np.float32),
                    'audio': audio.astype(np.float32),
                    'valid_regions': valid_regions,
                    'gesture_labels': gesture_labels,
                    'audio_path': audio_path,
                    'pose_path': file_path,
                    'speaker_id': speaker_id,
                })

                # Build sample index: weight each region by its length / avg_clip_length
                for region_idx, (start, end) in enumerate(valid_regions):
                    region_length = end - start
                    num_samples = max(1, int(region_length / avg_clip_length))
                    for _ in range(num_samples):
                        self.sample_index.append((file_idx, region_idx))

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_idx, region_idx = self.sample_index[idx]
        file_data = self.file_data[file_idx]

        region_start, region_end = file_data['valid_regions'][region_idx]
        region_length = region_end - region_start

        # Sample random clip length
        clip_length = np.random.randint(
            self.min_length,
            min(self.max_length, region_length) + 1
        )

        # Sample random start within region
        max_start = region_end - clip_length
        clip_start = np.random.randint(region_start, max_start + 1)

        # Extract clip
        poses = file_data['poses'][clip_start:clip_start + clip_length]
        audio = file_data['audio'][clip_start:clip_start + clip_length]

        # Majority gesture type (tie -> semantic=1)
        gesture_labels = file_data['gesture_labels'][clip_start:clip_start + clip_length]
        semantic_count = np.sum(gesture_labels == 1)
        beat_count = np.sum(gesture_labels == 0)
        gesture_type = 1 if semantic_count >= beat_count else 0

        # Extract anchor pool: up to anchor_max_frames preceding the clip
        K = self.anchor_max_frames
        anchor_start = max(0, clip_start - K)
        anchor_pool = file_data['poses'][anchor_start:clip_start]  # (≤K, 165)
        anchor_audio = file_data['audio'][anchor_start:clip_start]  # (≤K, 768)
        # Left-zero-pad if clip is near the start of the recording
        if anchor_pool.shape[0] < K:
            pad_poses = np.zeros((K - anchor_pool.shape[0], self.pose_dims), dtype=np.float32)
            anchor_pool = np.concatenate([pad_poses, anchor_pool], axis=0)  # (K, 165)
            pad_audio = np.zeros((K - anchor_audio.shape[0], file_data['audio'].shape[-1]), dtype=np.float32)
            anchor_audio = np.concatenate([pad_audio, anchor_audio], axis=0)  # (K, 768)

        return {
            'poses': torch.FloatTensor(poses),
            'audio': torch.FloatTensor(audio),
            'gesture_type': gesture_type,
            'speaker_id': file_data['speaker_id'],
            'length': clip_length,
            'clip_start': clip_start,
            'anchor_pool': anchor_pool,  # (K, 165) real preceding frames, zero-padded at start
            'anchor_audio': anchor_audio,  # (K, 768) audio for anchor frames, zero-padded at start
            'audio_path': file_data['audio_path'],
            'pose_path': file_data['pose_path'],
        }


# =============================================================================
# Collate function for variable-length batches
# =============================================================================

def variable_length_collate_fn(batch):
    """
    Collate variable-length samples into a padded batch.

    Returns:
        (data_dict, dummy_label) for backward compat with (data, label) unpacking.
        data_dict contains: poses, audio, gesture_type, lengths, padding_mask
    """
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    gesture_types = torch.tensor([item['gesture_type'] for item in batch], dtype=torch.long)
    t_max = lengths.max().item()
    batch_size = len(batch)

    pose_dim = batch[0]['poses'].shape[-1]
    audio_dim = batch[0]['audio'].shape[-1]

    # Pad poses and audio
    padded_poses = torch.zeros(batch_size, t_max, pose_dim)
    padded_audio = torch.zeros(batch_size, t_max, audio_dim)
    padding_mask = torch.ones(batch_size, t_max, dtype=torch.bool)  # True = padded

    for i, item in enumerate(batch):
        l = item['length']
        padded_poses[i, :l] = item['poses']
        padded_audio[i, :l] = item['audio']
        padding_mask[i, :l] = False  # valid positions

    # Stack anchor pools — all same shape (K, 165) due to zero-padding in __getitem__
    anchor_pool = torch.stack(
        [torch.FloatTensor(item['anchor_pool']) for item in batch], dim=0
    )  # (B, K, 165)
    anchor_audio = torch.stack(
        [torch.FloatTensor(item['anchor_audio']) for item in batch], dim=0
    )  # (B, K, 768)

    speaker_ids = torch.tensor([item['speaker_id'] for item in batch], dtype=torch.long)

    data_dict = {
        'poses': padded_poses,
        'audio': padded_audio,
        'gesture_type': gesture_types,
        'speaker_id': speaker_ids,
        'lengths': lengths,
        'padding_mask': padding_mask,
        'clip_start': torch.tensor([item['clip_start'] for item in batch], dtype=torch.long),
        'anchor_pool': anchor_pool,
        'anchor_audio': anchor_audio,
        'audio_path': [item['audio_path'] for item in batch],
        'pose_path': [item['pose_path'] for item in batch],
    }

    return data_dict, torch.zeros(batch_size)  # dummy label


# =============================================================================
# DataLoader factory
# =============================================================================

def get_beat_variable_length_loader(
    data_path: str = 'BEAT2',
    language: str = 'english',
    batch_size: int = 32,
    min_length: int = 5,
    max_length: int = 300,
    shuffle: bool = True,
    num_workers: int = 8,
    use_axis_angle: bool = True,
    audio_dir: Optional[str] = None,
    split: Optional[str] = None,
    anchor_max_frames: int = 30,
):
    """
    Create a DataLoader for BEAT2 with variable-length clips, gesture type, and audio.

    Args:
        data_path: Path to BEAT2 directory
        language: Language subset
        batch_size: Batch size
        min_length: Minimum clip length
        max_length: Maximum clip length
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        use_axis_angle: Load axis-angle poses (165D)
        audio_dir: Path to wav2vec features. If None, auto-constructed.
        split: 'train', 'val', or 'test' to filter by BEAT2 split. None = all files.
    """
    dataset = BEAT2PoseDataset(
        data_path=data_path,
        language=language,
        min_length=min_length,
        max_length=max_length,
        use_axis_angle=use_axis_angle,
        audio_dir=audio_dir,
        split=split,
        anchor_max_frames=anchor_max_frames,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate_fn,
    )


# =============================================================================
# Legacy classes and functions (kept for backward compatibility)
# =============================================================================

class AMASSDataset(Dataset):
    """AMASS Dataset (e.g., BMLrub) for SMPLX pose sequences"""

    def __init__(self,
                 data_path: str,
                 subsets: list = ['BMLrub'],
                 sequence_length: int = 120,
                 stride: int = 30,
                 target_fps: int = 30,
                 normalize: bool = False,
                 load_gt_geometry: bool = False,
                 compute_gt_on_fly: bool = False,
                 smplx_model_path: str = 'models_smplx_v1_1/models'):
        self.data_path = data_path
        self.subsets = subsets
        self.sequence_length = sequence_length
        self.stride = stride
        self.target_fps = target_fps
        self.normalize = normalize
        self.load_gt_geometry = load_gt_geometry
        self.compute_gt_on_fly = compute_gt_on_fly
        self.smplx_model_path = smplx_model_path
        self.smplx_model = None

        self.pose_files = []
        for subset in subsets:
            subset_path = os.path.join(data_path, subset)
            if os.path.exists(subset_path):
                for subject_dir in os.listdir(subset_path):
                    subject_path = os.path.join(subset_path, subject_dir)
                    if os.path.isdir(subject_path):
                        npz_files = glob.glob(os.path.join(subject_path, '*.npz'))
                        self.pose_files.extend(npz_files)

        self.sequences = []
        self._load_sequences()
        print(f"Loaded {len(self.sequences)} pose sequences from AMASS {subsets}")

    def _load_sequences(self):
        print(f"Loading AMASS sequences from {len(self.pose_files)} files...")
        for file_path in tqdm(self.pose_files, desc="Loading AMASS files"):
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'poses' not in data:
                    continue
                poses = data['poses']
                source_fps = float(data.get('mocap_frame_rate', 120.0))
                downsample_factor = max(1, int(round(source_fps / self.target_fps)))
                if downsample_factor > 1:
                    poses = poses[::downsample_factor]
                if len(poses) < self.sequence_length:
                    continue
                for i in range(0, len(poses) - self.sequence_length + 1, self.stride):
                    sequence = poses[i:i + self.sequence_length]
                    self.sequences.append({'poses': sequence.astype(np.float32)})
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        pose_sequence = torch.FloatTensor(sequence_data['poses'])
        return {'poses': pose_sequence}, torch.zeros(1)


class CombinedPoseDataset(Dataset):
    """Combined dataset that merges multiple pose datasets"""

    def __init__(self, datasets: list):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)
        print(f"Combined dataset: {total} total sequences from {len(datasets)} datasets")

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        for i, cumsize in enumerate(self.cumulative_sizes):
            if idx < cumsize:
                if i == 0:
                    return self.datasets[i][idx]
                else:
                    return self.datasets[i][idx - self.cumulative_sizes[i-1]]
        raise IndexError(f"Index {idx} out of range")


def get_combined_pose_loader(beat2_path='BEAT2_joints_vertices', amass_path='AMASS',
                             amass_subsets=['BMLrub'], language='english',
                             batch_size=32, sequence_length=25, shuffle=True,
                             num_workers=8, use_axis_angle=True,
                             load_gt_geometry=True, compute_gt_on_fly=True,
                             smplx_model_path='models_smplx_v1_1/models'):
    """Legacy loader combining BEAT2 + AMASS. Kept for backward compatibility."""
    datasets = []
    beat2_dataset = BEAT2PoseDataset(
        data_path=beat2_path, language=language, sequence_length=sequence_length,
        use_axis_angle=use_axis_angle, load_gt_geometry=load_gt_geometry,
        compute_gt_on_fly=compute_gt_on_fly, smplx_model_path=smplx_model_path
    )
    datasets.append(beat2_dataset)
    amass_dataset = AMASSDataset(
        data_path=amass_path, subsets=amass_subsets, sequence_length=sequence_length,
        target_fps=30, load_gt_geometry=load_gt_geometry, compute_gt_on_fly=compute_gt_on_fly,
        smplx_model_path=smplx_model_path
    )
    datasets.append(amass_dataset)
    combined_dataset = CombinedPoseDataset(datasets)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def get_beat_pose_loader(data_path, language='chinese', batch_size=32, sequence_length=120,
                         shuffle=True, num_workers=0, use_axis_angle=False,
                         load_gt_geometry=False, compute_gt_on_fly=False,
                         smplx_model_path='models_smplx_v1_1/models'):
    """Legacy BEAT2 pose loader. Kept for backward compatibility."""
    dataset = BEAT2PoseDataset(
        data_path=data_path, language=language, sequence_length=sequence_length,
        use_axis_angle=use_axis_angle, load_gt_geometry=load_gt_geometry,
        compute_gt_on_fly=compute_gt_on_fly, smplx_model_path=smplx_model_path
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
