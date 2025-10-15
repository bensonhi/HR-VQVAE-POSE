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
                 load_gt_geometry: bool = False,
                 compute_gt_on_fly: bool = False,
                 smplx_model_path: str = 'models_smplx_v1_1/models'):
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
            load_gt_geometry: If True, load/compute ground truth vertices and joints for supervision
            compute_gt_on_fly: If True, compute GT geometry from poses on-the-fly instead of loading precomputed.
                              Requires use_axis_angle=True and axis-angle poses in dataset.
            smplx_model_path: Path to SMPLX models (used when compute_gt_on_fly=True)
        """
        self.data_path = data_path
        self.language = language
        self.sequence_length = sequence_length
        self.stride = stride
        self.pose_dims = pose_dims
        self.normalize = normalize
        self.use_axis_angle = use_axis_angle
        self.load_gt_geometry = load_gt_geometry
        self.compute_gt_on_fly = compute_gt_on_fly
        self.smplx_model_path = smplx_model_path

        # Initialize SMPLX model if computing GT on-the-fly
        self.smplx_model = None
        if self.compute_gt_on_fly and self.load_gt_geometry:
            if not self.use_axis_angle:
                raise ValueError("compute_gt_on_fly requires use_axis_angle=True")

            print(f"Initializing SMPLX model from {smplx_model_path} for on-the-fly GT computation...")
            try:
                import smplx
                self.smplx_model = smplx.create(
                    smplx_model_path,
                    model_type='smplx',
                    gender='neutral',
                    use_face_contour=False,
                    use_pca=False,
                    ext='npz'
                )
                # Freeze parameters
                for param in self.smplx_model.parameters():
                    param.requires_grad = False
                print("SMPLX model loaded successfully!")
            except Exception as e:
                print(f"Warning: Failed to load SMPLX model: {e}")
                print("Falling back to loading pre-computed geometry if available.")
                self.compute_gt_on_fly = False
        
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
        print(f"Loading pose sequences from {len(self.pose_files)} files...")
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
                # (only if not computing on-the-fly)
                gt_joints = None
                gt_vertices = None
                if self.load_gt_geometry and not self.compute_gt_on_fly:
                    gt_joints = data['joints'] if 'joints' in data else None
                    gt_vertices = data['vertices'] if 'vertices' in data else None

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

                    # Add ground truth geometry if available (pre-computed)
                    if self.load_gt_geometry and not self.compute_gt_on_fly:
                        if gt_joints is not None:
                            gt_joints_seq = gt_joints[i:i + self.sequence_length]
                            sequence_data['gt_joints'] = gt_joints_seq.astype(np.float32)
                        if gt_vertices is not None:
                            gt_vertices_seq = gt_vertices[i:i + self.sequence_length]
                            sequence_data['gt_vertices'] = gt_vertices_seq.astype(np.float32)
                    elif self.load_gt_geometry and self.compute_gt_on_fly:
                        # Mark for on-the-fly computation (will compute in __getitem__)
                        sequence_data['compute_gt'] = True

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

            # Check if we need to compute GT on-the-fly
            if sequence_data.get('compute_gt', False) and self.smplx_model is not None:
                # Compute vertices and joints from axis-angle poses
                with torch.no_grad():
                    # Parse 165D pose into SMPLX parameters
                    batch_size, seq_len, _ = pose_sequence.shape
                    pose_flat = pose_sequence.reshape(batch_size * seq_len, 165)

                    global_orient = pose_flat[:, :3]
                    body_pose = pose_flat[:, 3:66]
                    jaw_pose = pose_flat[:, 66:69]
                    leye_pose = pose_flat[:, 69:72]
                    reye_pose = pose_flat[:, 72:75]
                    left_hand_pose = pose_flat[:, 75:120]
                    right_hand_pose = pose_flat[:, 120:165]

                    # SMPLX forward pass
                    output = self.smplx_model(
                        global_orient=global_orient,
                        body_pose=body_pose,
                        jaw_pose=jaw_pose,
                        leye_pose=leye_pose,
                        reye_pose=reye_pose,
                        left_hand_pose=left_hand_pose,
                        right_hand_pose=right_hand_pose,
                        betas=torch.zeros(batch_size * seq_len, 10),
                        expression=torch.zeros(batch_size * seq_len, 10),
                        return_verts=True
                    )

                    # Reshape back to (batch, seq_len, ...)
                    vertices = output.vertices.reshape(batch_size, seq_len, -1, 3)
                    joints = output.joints.reshape(batch_size, seq_len, -1, 3)

                    ret_dict['gt_vertices'] = vertices.squeeze(0)  # Remove batch dim if single sample
                    ret_dict['gt_joints'] = joints.squeeze(0)
            else:
                # Use pre-computed GT if available
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
                        load_gt_geometry: bool = False,
                        compute_gt_on_fly: bool = False,
                        smplx_model_path: str = 'models_smplx_v1_1/models'):
    """
    Create a DataLoader for BEAT2 pose sequences

    Args:
        data_path: Path to BEAT2 data directory
        language: Language subset
        batch_size: Batch size
        sequence_length: Length of pose sequences
        shuffle: Whether to shuffle data
        num_workers: Number of parallel data loading workers
        use_axis_angle: Load axis-angle (165D) instead of joint positions (381D)
        load_gt_geometry: Load/compute ground truth vertices and joints
        compute_gt_on_fly: Compute GT geometry on-the-fly instead of loading pre-computed
        smplx_model_path: Path to SMPLX models (for on-the-fly computation)
    """
    dataset = BEAT2PoseDataset(
        data_path=data_path,
        language=language,
        sequence_length=sequence_length,
        use_axis_angle=use_axis_angle,
        load_gt_geometry=load_gt_geometry,
        compute_gt_on_fly=compute_gt_on_fly,
        smplx_model_path=smplx_model_path
    )

    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Faster CPU-to-GPU transfer
    )