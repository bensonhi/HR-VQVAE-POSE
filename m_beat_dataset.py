import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from typing import Optional, Union
from tqdm import tqdm


class BEAT2PoseDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 language: str = 'english',
                 sequence_length: int = 120,
                 stride: int = 30,
                 pose_dims: int = 165,
                 normalize: bool = False,
                 use_axis_angle: bool = False,
                 load_gt_geometry: bool = False):
        """
        BEAT2 Pose Sequence Dataset

        Args:
            data_path: Path to BEAT2 directory
            language: Language subset ('english', 'chinese', 'spanish', 'japanese')
            sequence_length: Length of pose sequences to extract
            stride: Stride between sequences
            pose_dims: Dimension of pose data (165 for SMPLX axis-angle, 381 for joints)
            normalize: Whether to normalize pose data
            use_axis_angle: If True, load axis-angle poses (165D). If False, load joint positions (381D)
            load_gt_geometry: If True, also load ground truth vertices and joints for supervision
        """
        self.data_path = data_path
        self.language = language
        self.sequence_length = sequence_length
        self.stride = stride
        self.pose_dims = pose_dims
        self.normalize = normalize
        self.use_axis_angle = use_axis_angle
        self.load_gt_geometry = load_gt_geometry
        
        # Determine the correct language folder
        lang_folders = {
            'english': 'beat_english_v2.0.0',
            'chinese': 'beat_chinese_v2.0.0', 
            'spanish': 'beat_spanish_v2.0.0',
            'japanese': 'beat_japanese_v2.0.0'
        }
        
        # All languages have pose data in smplxflame_30 folder
        lang_folder = lang_folders.get(language, 'beat_chinese_v2.0.0')
        self.pose_files = glob.glob(os.path.join(data_path, lang_folder, 'smplxflame_30', '*.npz'))
        self.use_semantic = False
        
        # Load and process sequences
        self.sequences = []
        self._load_sequences()
        
        print(f"Loaded {len(self.sequences)} pose sequences from {language} BEAT2 data")
    
    def _load_sequences(self):
        """Load all pose sequences from files"""
        if self.use_semantic:
            self._load_semantic_sequences()
        else:
            self._load_pose_sequences()
    
    def _load_semantic_sequences(self):
        """Load semantic feature sequences for English"""
        print(f"Loading semantic sequences from {len(self.pose_files)} files...")
        for file_path in tqdm(self.pose_files, desc="Loading files"):
            try:
                # Read semantic features (assume they're text files with numerical data)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Convert text to numerical features (simple approach)
                # In practice, you'd use proper text-to-feature conversion
                features = []
                for line in lines:
                    # Simple word count features (replace with proper semantic features)
                    words = line.strip().split()
                    word_features = [len(words), len(line.strip())] + [hash(w) % 100 for w in words[:10]]
                    # Pad or truncate to fixed size
                    while len(word_features) < self.pose_dims:
                        word_features.append(0.0)
                    features.append(word_features[:self.pose_dims])
                
                if len(features) < self.sequence_length:
                    continue
                    
                # Extract sequences with stride
                for i in range(0, len(features) - self.sequence_length + 1, self.stride):
                    sequence = np.array(features[i:i + self.sequence_length])
                    if self.normalize:
                        sequence = self._normalize_sequence(sequence)
                    self.sequences.append(sequence)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    def _load_pose_sequences(self):
        """Load actual pose sequences"""
        print(f"Loading semantic sequences from {len(self.pose_files)} files...")
        for file_path in tqdm(self.pose_files, desc="Loading files"):
            try:
                data = np.load(file_path)

                # Choose between axis-angle poses or joint positions
                if self.use_axis_angle:
                    poses = data['poses']  # Shape: (T, 165) - axis-angle representation
                else:
                    poses = data['joints']  # Shape: (T, num_joints, 3) - joint positions

                if len(poses) < self.sequence_length:
                    continue

                # Load ground truth geometry if needed for supervision
                gt_joints = data['joints'] if self.load_gt_geometry and 'joints' in data else None
                gt_vertices = data['vertices'] if self.load_gt_geometry and 'vertices' in data else None

                # Extract sequences with stride
                for i in range(0, len(poses) - self.sequence_length + 1, self.stride):
                    sequence = poses[i:i + self.sequence_length]

                    # Flatten if needed
                    if sequence.ndim > 2:
                        # Joint positions: (seq_len, num_joints, 3) -> (seq_len, num_joints*3)
                        sequence = sequence.reshape(sequence.shape[0], -1)

                    sequence_data = {
                        'poses': sequence.astype(np.float32)
                    }

                    # Add ground truth geometry if available
                    if self.load_gt_geometry:
                        if gt_joints is not None:
                            gt_joints_seq = gt_joints[i:i + self.sequence_length]
                            sequence_data['gt_joints'] = gt_joints_seq.astype(np.float32)
                        if gt_vertices is not None:
                            gt_vertices_seq = gt_vertices[i:i + self.sequence_length]
                            sequence_data['gt_vertices'] = gt_vertices_seq.astype(np.float32)

                    self.sequences.append(sequence_data)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
    
    def _normalize_sequence(self, sequence):
        """Normalize pose sequence to [-1, 1] range"""
        # Simple min-max normalization per sequence
        seq_min = sequence.min(axis=0, keepdims=True)
        seq_max = sequence.max(axis=0, keepdims=True)
        normalized = 2.0 * (sequence - seq_min) / (seq_max - seq_min + 1e-8) - 1.0
        return normalized.astype(np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]

        if isinstance(sequence_data, dict):
            # New format with ground truth geometry
            pose_sequence = torch.FloatTensor(sequence_data['poses'])

            # Prepare return dict
            ret_dict = {'poses': pose_sequence}

            if 'gt_joints' in sequence_data:
                ret_dict['gt_joints'] = torch.FloatTensor(sequence_data['gt_joints'])
            if 'gt_vertices' in sequence_data:
                ret_dict['gt_vertices'] = torch.FloatTensor(sequence_data['gt_vertices'])

            return ret_dict, torch.zeros(1)  # dummy label for compatibility
        else:
            # Legacy format (backward compatibility)
            return torch.FloatTensor(sequence_data), torch.zeros(1)


def get_beat_pose_loader(data_path: str,
                        language: str = 'chinese',
                        batch_size: int = 32,
                        sequence_length: int = 120,
                        shuffle: bool = True,
                        num_workers: int = 0,
                        use_axis_angle: bool = False,
                        load_gt_geometry: bool = False):
    """Create a DataLoader for BEAT2 pose sequences"""
    dataset = BEAT2PoseDataset(
        data_path=data_path,
        language=language,
        sequence_length=sequence_length,
        use_axis_angle=use_axis_angle,
        load_gt_geometry=load_gt_geometry
    )

    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Faster CPU-to-GPU transfer
    )