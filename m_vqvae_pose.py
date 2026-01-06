import sys
import torch
from torch import nn
from torch.nn import functional as F
from m_smplx_layer import SMPLXLayer

sys.path.append('../')


class Quantize(nn.Module):
    """Vector Quantization for VQ-VAE (kept for backward compatibility)"""
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


class VAEEncode(nn.Module):
    """
    VAE encoding module that outputs mu and logvar for continuous latent space.
    Replaces the discrete Quantize module with continuous Gaussian encoding.
    """
    def __init__(self, dim, latent_dim=None):
        super().__init__()

        self.dim = dim
        # latent_dim can be same as dim or different for compression
        self.latent_dim = latent_dim if latent_dim is not None else dim

        # Linear layers to output mu and logvar
        self.fc_mu = nn.Linear(dim, self.latent_dim)
        self.fc_logvar = nn.Linear(dim, self.latent_dim)

    def forward(self, input):
        """
        Args:
            input: (batch, seq_len, dim)

        Returns:
            z: sampled latent (batch, seq_len, latent_dim)
            kl_loss: KL divergence loss
            mu: mean (batch, seq_len, latent_dim)
        """
        # input shape: (B, T, D)
        batch_size, seq_len, dim = input.shape

        # Flatten for linear layers
        flat_input = input.reshape(-1, dim)  # (B*T, D)

        # Encode to mu and logvar
        mu = self.fc_mu(flat_input)  # (B*T, latent_dim)
        logvar = self.fc_logvar(flat_input)  # (B*T, latent_dim)

        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Compute KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Reshape back
        z = z.reshape(batch_size, seq_len, self.latent_dim)
        mu = mu.reshape(batch_size, seq_len, self.latent_dim)

        return z, kl_loss, mu

    def sample(self, mu):
        """Sample from the distribution (for inference/generation)"""
        return mu  # For deterministic inference, just use the mean


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
    """Multi-level hierarchical VAE for pose sequences (continuous latent space)"""
    def __init__(
            self,
            in_channel=165,  # Pose dimension
            channel=256,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_level=4,
            n_embed=512,  # No longer used, kept for backward compatibility
            n_embeds=None,  # No longer used, kept for backward compatibility
            decay=0.99,  # No longer used, kept for backward compatibility
            stride=4,
            use_smplx=False,
            smplx_model_path='models_smplx_v1_1/models',
            latent_dims=None,  # List of latent dimensions per level, if None uses embed_dim
    ):
        super().__init__()
        self.device = 'cpu'

        # Main encoder
        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride)
        self.quantize_conv = nn.Conv1d(channel, embed_dim, 1)

        # Multiple VAE encoding levels (continuous latent space)
        self.n_level = n_level
        self.embed_dim = embed_dim
        self.vae_encoders = nn.ModuleList()

        # Set up latent dimensions per level
        if latent_dims is not None:
            assert len(latent_dims) == n_level, f"latent_dims length {len(latent_dims)} must match n_level {n_level}"
            self.latent_dims = latent_dims
        else:
            # Use same latent dimension for all levels
            self.latent_dims = [embed_dim] * n_level

        # Create VAE encoders for each level
        for i in range(n_level):
            self.vae_encoders.append(VAEEncode(embed_dim, self.latent_dims[i]))

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
            kl_losses: List of KL divergence losses per level
            pred_vertices: SMPLX vertices for final output (if compute_geometry=True)
            pred_joints: SMPLX joints for final output (if compute_geometry=True)
        """
        enc = self.enc(input)
        latent = self.quantize_conv(enc)

        residual = latent.permute(0, 2, 1)  # (B, T, D)
        accumulated_latent = torch.zeros_like(residual)

        intermediate_outputs = []
        kl_losses = []

        for i in range(self.n_level):
            # VAE encode at this level
            z, kl_loss, mu = self.vae_encoders[i](residual)
            kl_losses.append(kl_loss)

            # Accumulate latent values
            accumulated_latent = accumulated_latent + z

            # Decode from accumulated latent up to this level
            level_latent = accumulated_latent.permute(0, 2, 1)  # (B, D, T)
            level_dec = self.decode(level_latent)
            level_dec = level_dec.transpose(1, 2)  # (B, T, pose_dim)

            intermediate_outputs.append(level_dec)

            # Update residual for next level
            residual = residual - z

        # Optionally compute geometry for final output
        pred_vertices, pred_joints = None, None
        if compute_geometry and self.use_smplx:
            pred_vertices, pred_joints = self.smplx_layer(intermediate_outputs[-1])

        # Stack KL losses for backward compatibility
        combined_kl_loss = torch.stack(kl_losses).mean().unsqueeze(0)

        if compute_geometry and self.use_smplx:
            return intermediate_outputs, combined_kl_loss, pred_vertices, pred_joints

        return intermediate_outputs, combined_kl_loss

    def encode(self, input):
        """
        Hierarchical VAE encoding with continuous latent space.

        Args:
            input: (batch, pose_dim, seq_len) in conv1d format

        Returns:
            final_latent: accumulated latent representation (batch, embed_dim, seq_len)
            combined_kl_loss: combined KL divergence loss
            mus: list of mu values per level (for analysis)
        """
        enc = self.enc(input)
        latent = self.quantize_conv(enc)

        # Multi-level hierarchical residual VAE encoding
        kl_losses = []
        mus = []
        residual = latent.permute(0, 2, 1)  # (B, T, D)
        accumulated_latent = torch.zeros_like(residual)  # Accumulate all levels

        for i in range(self.n_level):
            # VAE encode at this level: get z, kl_loss, mu
            z, kl_loss, mu = self.vae_encoders[i](residual)
            kl_losses.append(kl_loss)
            mus.append(mu)

            # Accumulate latent values (hierarchical residual learning)
            accumulated_latent = accumulated_latent + z

            # Update residual (subtract sampled latent for next level)
            residual = residual - z

        # Use accumulated latent from all levels
        final_latent = accumulated_latent.permute(0, 2, 1)  # (B, D, T)
        combined_kl_loss = torch.stack(kl_losses).mean()

        return final_latent, combined_kl_loss.unsqueeze(0), mus

    def decode(self, quant):
        dec = self.dec(quant)
        return dec

    def decode_partial_levels(self, input, num_levels=None):
        """
        Encode and decode using only the first num_levels VAE encoding levels.
        This allows visualizing what each level learns.

        Args:
            input: Input tensor (batch, pose_dim, sequence_length) in conv1d format
            num_levels: Number of levels to use (1 to n_level). If None, uses all levels.

        Returns:
            dec: Decoded output (batch, sequence_length, pose_dim)
            kl_loss: Combined KL divergence loss
        """
        if num_levels is None:
            num_levels = self.n_level

        num_levels = min(num_levels, self.n_level)

        enc = self.enc(input)
        latent = self.quantize_conv(enc)

        # Multi-level hierarchical residual VAE encoding (up to num_levels)
        kl_losses = []
        residual = latent.permute(0, 2, 1)  # (B, T, D)
        accumulated_latent = torch.zeros_like(residual)

        for i in range(num_levels):
            z, kl_loss, mu = self.vae_encoders[i](residual)
            kl_losses.append(kl_loss)
            accumulated_latent = accumulated_latent + z
            residual = residual - z

        final_latent = accumulated_latent.permute(0, 2, 1)  # (B, D, T)
        combined_kl_loss = torch.stack(kl_losses).mean()

        dec = self.decode(final_latent)
        dec = dec.transpose(1, 2)  # (B, T, pose_dim)

        return dec, combined_kl_loss.unsqueeze(0)

    def sample(self, batch_size=1, seq_len=25, device='cpu'):
        """
        Generate poses by sampling from the prior distribution.
        For VAE, we sample from standard Gaussian and decode.

        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            device: Device to generate on

        Returns:
            Generated poses (batch_size, seq_len, pose_dim)
        """
        # Sample from standard Gaussian prior
        z = torch.randn(batch_size, self.embed_dim, seq_len, device=device)

        # Decode
        dec = self.decode(z)
        return dec.transpose(1, 2)  # (B, T, pose_dim)