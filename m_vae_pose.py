import sys
import torch
from torch import nn
from torch.nn import functional as F
from m_smplx_layer import SMPLXLayer

sys.path.append('../')


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


class VAELevel(nn.Module):
    """Single VAE level with continuous latent encoding (mean + logvar)"""
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        # Project to mean and logvar for VAE
        self.fc_mu = nn.Conv1d(embed_dim, embed_dim, 1)
        self.fc_logvar = nn.Conv1d(embed_dim, embed_dim, 1)

    def encode(self, x):
        """
        Encode to mean and logvar
        Args:
            x: (B, D, T) format
        Returns:
            mu: (B, T, D)
            logvar: (B, T, D)
        """
        mu = self.fc_mu(x).permute(0, 2, 1)  # (B, T, D)
        logvar = self.fc_logvar(x).permute(0, 2, 1)  # (B, T, D)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        Args:
            mu: (B, T, D)
            logvar: (B, T, D)
        Returns:
            z: (B, T, D)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input):
        """
        Args:
            input: (B, D, T) format from encoder
        Returns:
            z: sampled latent (B, T, D)
            mu: mean (B, T, D)
            logvar: log variance (B, T, D)
        """
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAE_Pose_1(nn.Module):
    """Single-level VAE for pose sequences"""
    def __init__(
            self,
            in_channel=165,  # Pose dimension
            channel=256,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            decay=0.99,
            use_smplx=False,
            smplx_model_path='models_smplx_v1_1/models',
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_conv = nn.Conv1d(channel, embed_dim, 1)
        self.vae_level = VAELevel(embed_dim)
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

        z, mu, logvar = self.encode(input)
        dec = self.decode(z)

        # Transpose back to (batch, sequence_length, pose_dim)
        dec = dec.transpose(1, 2)

        # Compute KL divergence
        kl_loss = self.kl_divergence(mu, logvar)

        # Optionally compute geometry through SMPLX
        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(dec)
            return dec, kl_loss, vertices, joints

        return dec, kl_loss

    def encode(self, input):
        enc = self.enc(input)
        quant = self.quantize_conv(enc)  # (B, D, T)
        z, mu, logvar = self.vae_level(quant)
        return z, mu, logvar

    def decode(self, z):
        # z is (B, T, D), need (B, D, T) for conv1d
        z = z.permute(0, 2, 1)
        dec = self.dec(z)
        return dec

    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        Args:
            mu: (B, T, D)
            logvar: (B, T, D)
        Returns:
            kl_loss: scalar
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()


class VAE_Pose_ML(nn.Module):
    """Multi-level hierarchical VAE for pose sequences"""
    def __init__(
            self,
            in_channel=165,  # Pose dimension
            channel=256,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_level=3,  # Changed default to 3 levels
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

        # Multiple VAE levels (continuous latent space)
        self.n_level = n_level
        self.vae_levels = nn.ModuleList()
        self.level_convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_level):
            self.vae_levels.append(VAELevel(embed_dim))
            self.level_convs.append(nn.Conv1d(embed_dim, embed_dim, 1))
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

        z, kl_losses = self.encode(input)
        dec = self.decode(z)

        # Transpose back to (batch, sequence_length, pose_dim)
        dec = dec.transpose(1, 2)

        # Combine KL losses from all levels
        total_kl = sum(kl_losses)

        # Optionally compute geometry through SMPLX
        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(dec)
            return dec, total_kl, vertices, joints

        return dec, total_kl

    def forward_with_intermediate_outputs(self, input, compute_geometry=False):
        """
        Forward pass that returns intermediate reconstructions at each level.
        Useful for progressive training with level-specific losses.

        Args:
            input: (batch, pose_dim, seq_len) in conv1d format
            compute_geometry: Whether to compute SMPLX geometry for final output

        Returns:
            intermediate_outputs: List of reconstructions [level_1, level_2, ..., level_n]
            total_kl: Combined KL divergence from all levels
            pred_vertices: SMPLX vertices for final output (if compute_geometry=True)
            pred_joints: SMPLX joints for final output (if compute_geometry=True)
        """
        enc = self.enc(input)
        quant = self.quantize_conv(enc)  # (B, D, T)

        intermediate_outputs = []
        kl_losses = []

        # Hierarchical residual learning in latent space (like VQ-VAE)
        # Start with encoded latent representation
        residual_latent = quant  # (B, D, T)
        accumulated_z = torch.zeros(quant.shape[0], quant.shape[2], self.vae_levels[0].embed_dim).to(quant.device)  # (B, T, D)

        for i in range(self.n_level):
            # VAE encode at this level: get z, mu, logvar
            # residual_latent is (B, D, T), vae_level expects (B, D, T)
            z_i, mu_i, logvar_i = self.vae_levels[i](residual_latent)
            # z_i, mu_i, logvar_i are all (B, T, D)

            # KL divergence for this level
            kl_i = self.kl_divergence(mu_i, logvar_i)
            kl_losses.append(kl_i)

            # Accumulate latent codes (hierarchical residual)
            accumulated_z = accumulated_z + z_i  # (B, T, D)

            # Decode from accumulated latent up to this level
            level_z = accumulated_z.permute(0, 2, 1)  # (B, D, T)
            level_dec = self.decode(level_z)  # (B, pose_dim, T)
            level_dec = level_dec.transpose(1, 2)  # (B, T, pose_dim)

            intermediate_outputs.append(level_dec)

            # Update residual for next level (in latent space)
            # Subtract what we just encoded from the residual
            if i < self.n_level - 1:
                # Convert z_i back to (B, D, T) format and subtract
                residual_latent = residual_latent - z_i.permute(0, 2, 1)  # (B, D, T)

        # Combine KL losses
        total_kl = sum(kl_losses)

        # Optionally compute geometry for final output
        pred_vertices, pred_joints = None, None
        if compute_geometry and self.use_smplx:
            pred_vertices, pred_joints = self.smplx_layer(intermediate_outputs[-1])

        if compute_geometry and self.use_smplx:
            return intermediate_outputs, total_kl, pred_vertices, pred_joints

        return intermediate_outputs, total_kl

    def encode(self, input):
        enc = self.enc(input)
        quant = self.quantize_conv(enc)  # (B, D, T)

        # Multi-level hierarchical VAE encoding (in latent space)
        kl_losses = []
        residual_latent = quant  # (B, D, T)
        accumulated_z = torch.zeros(quant.shape[0], quant.shape[2], self.vae_levels[0].embed_dim).to(quant.device)  # (B, T, D)

        for i in range(self.n_level):
            # Encode residual at this level
            z_i, mu_i, logvar_i = self.vae_levels[i](residual_latent)

            # KL divergence for this level
            kl_i = self.kl_divergence(mu_i, logvar_i)
            kl_losses.append(kl_i)

            # Accumulate latent codes (hierarchical residual)
            accumulated_z = accumulated_z + z_i  # (B, T, D)

            # Update residual for next level (in latent space)
            if i < self.n_level - 1:
                residual_latent = residual_latent - z_i.permute(0, 2, 1)  # (B, D, T)

        return accumulated_z, kl_losses

    def decode(self, z):
        # z is (B, T, D) or (B, D, T)
        if z.dim() == 3 and z.shape[1] != z.shape[2]:
            # Check which dimension is likely the embedding dim
            if z.shape[2] > z.shape[1]:
                # Likely (B, T, D)
                z = z.permute(0, 2, 1)
        dec = self.dec(z)
        return dec

    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        Args:
            mu: (B, T, D)
            logvar: (B, T, D)
        Returns:
            kl_loss: scalar
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def decode_partial_levels(self, input, num_levels=None):
        """
        Encode and decode using only the first num_levels.
        This allows visualizing what each level learns.

        Args:
            input: Input tensor (batch, pose_dim, sequence_length) in conv1d format
            num_levels: Number of levels to use (1 to n_level). If None, uses all levels.

        Returns:
            dec: Decoded output (batch, sequence_length, pose_dim)
            kl_loss: KL divergence
        """
        if num_levels is None:
            num_levels = self.n_level

        num_levels = min(num_levels, self.n_level)

        enc = self.enc(input)
        quant = self.quantize_conv(enc)

        kl_losses = []
        residual_latent = quant  # (B, D, T)
        accumulated_z = torch.zeros(quant.shape[0], quant.shape[2], self.vae_levels[0].embed_dim).to(quant.device)  # (B, T, D)

        for i in range(num_levels):
            z_i, mu_i, logvar_i = self.vae_levels[i](residual_latent)

            kl_i = self.kl_divergence(mu_i, logvar_i)
            kl_losses.append(kl_i)

            accumulated_z = accumulated_z + z_i  # (B, T, D)

            if i < num_levels - 1:
                residual_latent = residual_latent - z_i.permute(0, 2, 1)  # (B, D, T)

        final_z = accumulated_z.permute(0, 2, 1)  # (B, D, T)
        dec = self.decode(final_z)
        dec = dec.transpose(1, 2)  # (B, T, pose_dim)

        total_kl = sum(kl_losses)

        return dec, total_kl

    def sample(self, batch_size, seq_len, device='cuda'):
        """
        Sample from the prior p(z) = N(0, I) and decode.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            device: Device to generate samples on

        Returns:
            samples: (batch_size, seq_len, pose_dim)
        """
        # Sample from standard normal for all levels and accumulate
        accumulated_z = torch.zeros(batch_size, seq_len, self.vae_levels[0].embed_dim).to(device)

        for i in range(self.n_level):
            # Sample from N(0, I) at each level
            z_i = torch.randn(batch_size, seq_len, self.vae_levels[i].embed_dim).to(device)
            accumulated_z = accumulated_z + z_i

        # Decode
        z = accumulated_z.permute(0, 2, 1)  # (B, D, T)
        dec = self.decode(z)
        samples = dec.transpose(1, 2)  # (B, T, pose_dim)

        return samples
