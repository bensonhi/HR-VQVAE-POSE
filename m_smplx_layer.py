import torch
import torch.nn as nn
import smplx
import numpy as np


# SMPLX body part joint indices (127 joints total in SMPLX)
# Reference: SMPLX joint naming convention
SMPLX_JOINT_INDICES = {
    # Face: head, neck, jaw, eyes (joints 12, 15, 22, 23)
    'face': [12, 15, 22, 23],

    # Body: pelvis, spine, shoulders, hips, legs (joints 0-21 excluding face)
    'body': list(range(0, 12)) + [13, 14] + list(range(16, 22)),

    # Hands: finger joints (25-54 for left hand, 55-84 for right hand in some configurations)
    # This is approximate - adjust based on your SMPLX model configuration
    'hands': list(range(25, 55)) + list(range(55, 85)) if 85 <= 127 else list(range(25, min(55, 127))),

    # For more granular control
    'left_hand': list(range(25, 40)),
    'right_hand': list(range(40, 55)),
}

# Note: Vertex indices are mesh-specific. SMPLX has ~10,475 vertices.
# These are rough approximations - ideally use a proper segmentation
SMPLX_VERTEX_INDICES = {
    'face': list(range(0, 2000)),  # Approximate head region
    'body': list(range(2000, 7000)),  # Approximate torso/limbs
    'hands': list(range(7000, 10475)),  # Approximate hand regions
}


class SMPLXLayer(nn.Module):
    """
    Differentiable SMPLX layer for pose to mesh/joints conversion.

    Takes 165D axis-angle pose vectors and outputs vertices and joints.
    """
    def __init__(self,
                 model_path: str = 'models_smplx_v1_1/models',
                 gender: str = 'neutral',
                 use_pca: bool = False,
                 num_betas: int = 10,
                 num_expression_coeffs: int = 10,
                 batch_size: int = 1):
        """
        Args:
            model_path: Path to SMPLX model files
            gender: Model gender ('male', 'female', 'neutral')
            use_pca: Whether to use PCA for hand pose
            num_betas: Number of shape parameters
            num_expression_coeffs: Number of expression parameters
            batch_size: Batch size (for initialization)
        """
        super().__init__()

        # Create SMPLX model
        self.smplx_model = smplx.create(
            model_path,
            model_type='smplx',
            gender=gender,
            use_face_contour=False,
            use_pca=use_pca,
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
            ext='npz'
        )

        # Freeze SMPLX parameters (we only use it for forward kinematics)
        for param in self.smplx_model.parameters():
            param.requires_grad = False

        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs

        # Register buffers for default shape and expression
        self.register_buffer('default_betas', torch.zeros(1, num_betas))
        self.register_buffer('default_expression', torch.zeros(1, num_expression_coeffs))

    def forward(self, pose_params, betas=None, expression=None, return_joints_only=False):
        """
        Forward pass through SMPLX layer.

        Args:
            pose_params: (batch, seq_len, 165) axis-angle pose parameters
                        165D = global_orient(3) + body_pose(63) + jaw(3) + leye(3) + reye(3)
                              + left_hand(45) + right_hand(45)
            betas: Optional shape parameters (batch, num_betas). If None, uses zeros.
            expression: Optional expression parameters (batch, num_expression_coeffs). If None, uses zeros.
            return_joints_only: If True, only return joints (for efficiency)

        Returns:
            vertices: (batch, seq_len, num_vertices, 3) if return_joints_only=False
            joints: (batch, seq_len, num_joints, 3)
        """
        batch_size, seq_len, pose_dim = pose_params.shape
        assert pose_dim == 165, f"Expected 165D pose parameters, got {pose_dim}D"

        # Flatten batch and sequence dimensions
        pose_flat = pose_params.reshape(batch_size * seq_len, 165)

        # Parse pose parameters
        global_orient = pose_flat[:, :3]
        body_pose = pose_flat[:, 3:66]
        jaw_pose = pose_flat[:, 66:69]
        leye_pose = pose_flat[:, 69:72]
        reye_pose = pose_flat[:, 72:75]
        left_hand_pose = pose_flat[:, 75:120]
        right_hand_pose = pose_flat[:, 120:165]

        # Expand betas and expression for all frames
        if betas is None:
            betas = self.default_betas.expand(batch_size * seq_len, -1)
        else:
            betas = betas.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)

        if expression is None:
            expression = self.default_expression.expand(batch_size * seq_len, -1)
        else:
            expression = expression.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)

        # SMPLX forward pass
        output = self.smplx_model(
            global_orient=global_orient,
            body_pose=body_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas,
            expression=expression,
            return_verts=not return_joints_only
        )

        # Extract joints
        joints = output.joints  # (batch*seq_len, num_joints, 3)
        joints = joints.reshape(batch_size, seq_len, -1, 3)

        if return_joints_only:
            return None, joints

        # Extract vertices
        vertices = output.vertices  # (batch*seq_len, num_vertices, 3)
        vertices = vertices.reshape(batch_size, seq_len, -1, 3)

        return vertices, joints

    def to(self, device):
        """Override to method to also move SMPLX model"""
        super().to(device)
        self.smplx_model = self.smplx_model.to(device)
        return self


class SMPLXWrapper(nn.Module):
    """
    Wrapper that can be used with or without SMPLX layer.
    Useful for models that optionally use SMPLX supervision.
    """
    def __init__(self,
                 use_smplx: bool = False,
                 smplx_model_path: str = 'models_smplx_v1_1/models',
                 gender: str = 'neutral'):
        super().__init__()

        self.use_smplx = use_smplx

        if use_smplx:
            self.smplx_layer = SMPLXLayer(
                model_path=smplx_model_path,
                gender=gender
            )
        else:
            self.smplx_layer = None

    def forward(self, pose_params, **kwargs):
        """
        Args:
            pose_params: (batch, seq_len, 165) axis-angle poses

        Returns:
            vertices, joints if use_smplx=True, else (None, None)
        """
        if self.use_smplx:
            return self.smplx_layer(pose_params, **kwargs)
        else:
            return None, None