import sys
import torch
from torch import nn
from torch.nn import functional as F
from m_smplx_layer import SMPLXLayer

sys.path.append('../')


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot.sum(0), alpha=1 - self.decay
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock1D(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        # For single frames, use kernel=1 to avoid size issues
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel, channel, 1),  # kernel=1 for single frames
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        
        # For single frames, use stride=1 and kernel=1 to avoid size issues
        blocks = [
            nn.Conv1d(in_channel, channel // 2, 1),  # kernel=1 for single frames
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // 2, channel, 1),     # kernel=1 for single frames
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, channel, 1),          # kernel=1 for single frames
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock1D(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        # For single frames, use kernel=1 to avoid size issues
        blocks = [nn.Conv1d(in_channel, channel, 1)]  # kernel=1 for single frames

        for i in range(n_res_block):
            blocks.append(ResBlock1D(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        
        # For single frames, use simple 1x1 convolutions
        blocks.extend([
            nn.Conv1d(channel, channel // 2, 1),  # kernel=1 for single frames
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // 2, out_channel, 1),  # kernel=1 for single frames
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE_Pose_1(nn.Module):
    """Single-level VQ-VAE for pose sequences"""
    def __init__(
            self,
            in_channel=165,  # Pose dimension
            channel=256,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
            decay=0.99,
            use_smplx=False,
            smplx_model_path='models_smplx_v1_1/models',
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_conv = nn.Conv1d(channel, embed_dim, 1)
        self.quantize = Quantize(embed_dim, n_embed)
        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

        # Optional SMPLX layer for geometry supervision
        self.use_smplx = use_smplx
        if use_smplx:
            self.smplx_layer = SMPLXLayer(model_path=smplx_model_path)
        else:
            self.smplx_layer = None

    def forward(self, input, compute_geometry=False):
        # Input: (batch, sequence_length, pose_dim)
        # Transpose to (batch, pose_dim, sequence_length) for conv1d
        input = input.transpose(1, 2)

        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)

        # Transpose back to (batch, sequence_length, pose_dim)
        dec = dec.transpose(1, 2)

        # Optionally compute geometry through SMPLX
        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(dec)
            return dec, diff, vertices, joints

        return dec, diff

    def encode(self, input):
        enc = self.enc(input)
        quant = self.quantize_conv(enc).permute(0, 2, 1)  # (B, T, D)
        quant, diff, id = self.quantize(quant)
        quant = quant.permute(0, 2, 1)  # (B, D, T)
        diff = diff.unsqueeze(0)
        return quant, diff, id

    def decode(self, quant):
        dec = self.dec(quant)
        return dec

    def decode_code(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 2, 1)  # (B, D, T)
        dec = self.decode(quant)
        return dec.transpose(1, 2)


class VQVAE_Pose_ML(nn.Module):
    """Multi-level hierarchical VQ-VAE for pose sequences"""
    def __init__(
            self,
            in_channel=165,  # Pose dimension
            channel=256,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_level=4,
            n_embed=512,
            n_embeds=None,  # List of different codebook sizes per layer {8, 64, 512}
            decay=0.99,
            stride=4,
            use_smplx=False,
            smplx_model_path='models_smplx_v1_1/models',
    ):
        super().__init__()
        self.device = 'cpu'

        # Main encoder
        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride)
        self.quantize_conv = nn.Conv1d(channel, embed_dim, 1)

        # Multiple quantization levels
        self.n_level = n_level
        self.quantizes = nn.ModuleList()
        self.quantizes_conv = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Use different codebook sizes per layer if provided (for paper's {8, 64, 512} specification)
        if n_embeds is not None:
            assert len(n_embeds) == n_level, f"n_embeds length {len(n_embeds)} must match n_level {n_level}"
            for i in range(n_level):
                self.quantizes.append(Quantize(embed_dim, n_embeds[i], decay=decay))
                self.quantizes_conv.append(nn.Conv1d(embed_dim, embed_dim, 1))
                self.bns.append(nn.BatchNorm1d(embed_dim))
        else:
            # Use same codebook size for all layers (backward compatibility)
            for i in range(n_level):
                self.quantizes.append(Quantize(embed_dim, n_embed, decay=decay))
                self.quantizes_conv.append(nn.Conv1d(embed_dim, embed_dim, 1))
                self.bns.append(nn.BatchNorm1d(embed_dim))

        # Decoder
        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=stride)

        # Optional SMPLX layer for geometry supervision
        self.use_smplx = use_smplx
        if use_smplx:
            self.smplx_layer = SMPLXLayer(model_path=smplx_model_path)
        else:
            self.smplx_layer = None

    def forward(self, input, compute_geometry=False, return_intermediate=False):
        # Input: (batch, sequence_length, pose_dim)
        # Transpose to (batch, pose_dim, sequence_length) for conv1d
        input = input.transpose(1, 2)

        if return_intermediate:
            # Return intermediate reconstructions for progressive training
            return self.forward_with_intermediate_outputs(input, compute_geometry)

        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)

        # Transpose back to (batch, sequence_length, pose_dim)
        dec = dec.transpose(1, 2)

        # Optionally compute geometry through SMPLX
        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(dec)
            return dec, diff, vertices, joints

        return dec, diff

    def forward_with_intermediate_outputs(self, input, compute_geometry=False):
        """
        Forward pass that returns intermediate reconstructions at each level.
        Useful for progressive training with level-specific losses.

        Args:
            input: (batch, pose_dim, seq_len) in conv1d format
            compute_geometry: Whether to compute SMPLX geometry for final output

        Returns:
            intermediate_outputs: List of reconstructions [level_1, level_2, ..., level_n]
            diffs: List of quantization losses per level
            pred_vertices: SMPLX vertices for final output (if compute_geometry=True)
            pred_joints: SMPLX joints for final output (if compute_geometry=True)
        """
        enc = self.enc(input)
        quant = self.quantize_conv(enc)

        residual = quant.permute(0, 2, 1)  # (B, T, D)
        accumulated_quant = torch.zeros_like(residual)

        intermediate_outputs = []
        diffs = []

        for i in range(self.n_level):
            # Quantize at this level
            quantized, diff, id = self.quantizes[i](residual)
            diffs.append(diff)

            # Accumulate quantizations
            accumulated_quant = accumulated_quant + quantized

            # Decode from accumulated quantization up to this level
            level_quant = accumulated_quant.permute(0, 2, 1)  # (B, D, T)
            level_dec = self.decode(level_quant)
            level_dec = level_dec.transpose(1, 2)  # (B, T, pose_dim)

            intermediate_outputs.append(level_dec)

            # Update residual for next level
            residual = residual - quantized

        # Optionally compute geometry for final output
        pred_vertices, pred_joints = None, None
        if compute_geometry and self.use_smplx:
            pred_vertices, pred_joints = self.smplx_layer(intermediate_outputs[-1])

        # Stack diffs for backward compatibility
        combined_diff = torch.stack(diffs).mean().unsqueeze(0)

        if compute_geometry and self.use_smplx:
            return intermediate_outputs, combined_diff, pred_vertices, pred_joints

        return intermediate_outputs, combined_diff

    def encode(self, input):
        enc = self.enc(input)
        quant = self.quantize_conv(enc)

        # Multi-level hierarchical residual quantization
        diffs = []
        ids = []
        residual = quant.permute(0, 2, 1)  # (B, T, D)
        accumulated_quant = torch.zeros_like(residual)  # Accumulate all levels

        for i in range(self.n_level):
            quantized, diff, id = self.quantizes[i](residual)
            diffs.append(diff)
            ids.append(id)

            # Accumulate quantized values (hierarchical residual learning)
            accumulated_quant = accumulated_quant + quantized

            # Update residual (subtract quantized version for next level)
            residual = residual - quantized

        # Use accumulated quantization from all levels
        final_quant = accumulated_quant.permute(0, 2, 1)  # (B, D, T)
        combined_diff = torch.stack(diffs).mean()

        return final_quant, combined_diff.unsqueeze(0), ids

    def decode(self, quant):
        dec = self.dec(quant)
        return dec

    def decode_partial_levels(self, input, num_levels=None):
        """
        Encode and decode using only the first num_levels quantization levels.
        This allows visualizing what each level learns.

        Args:
            input: Input tensor (batch, pose_dim, sequence_length) in conv1d format
            num_levels: Number of levels to use (1 to n_level). If None, uses all levels.

        Returns:
            dec: Decoded output (batch, sequence_length, pose_dim)
            diff: Quantization loss
        """
        if num_levels is None:
            num_levels = self.n_level

        num_levels = min(num_levels, self.n_level)

        enc = self.enc(input)
        quant = self.quantize_conv(enc)

        # Multi-level hierarchical residual quantization (up to num_levels)
        diffs = []
        residual = quant.permute(0, 2, 1)  # (B, T, D)
        accumulated_quant = torch.zeros_like(residual)

        for i in range(num_levels):
            quantized, diff, id = self.quantizes[i](residual)
            diffs.append(diff)
            accumulated_quant = accumulated_quant + quantized
            residual = residual - quantized

        final_quant = accumulated_quant.permute(0, 2, 1)  # (B, D, T)
        combined_diff = torch.stack(diffs).mean()

        dec = self.decode(final_quant)
        dec = dec.transpose(1, 2)  # (B, T, pose_dim)

        return dec, combined_diff.unsqueeze(0)

    def decode_code(self, codes):
        """
        Decode from hierarchical codes.
        Args:
            codes: List of code tensors [code_L1, code_L2, ..., code_Ln] or single code tensor
        """
        if not isinstance(codes, list):
            # Backward compatibility - single code uses only first level
            code = codes
            quant = self.quantizes[0].embed_code(code)
            quant = quant.permute(0, 2, 1)  # (B, D, T)
        else:
            # Hierarchical decoding - accumulate all levels
            accumulated_quant = None
            for i, code in enumerate(codes):
                if i >= self.n_level:
                    break
                level_quant = self.quantizes[i].embed_code(code)  # (B, T, D)
                if accumulated_quant is None:
                    accumulated_quant = level_quant
                else:
                    accumulated_quant = accumulated_quant + level_quant
            quant = accumulated_quant.permute(0, 2, 1)  # (B, D, T)

        dec = self.decode(quant)
        return dec.transpose(1, 2)