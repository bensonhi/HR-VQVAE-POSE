import sys
import math
import torch
from torch import nn
from torch.nn import functional as F
from m_smplx_layer import SMPLXLayer

sys.path.append('../')


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for arbitrary sequence lengths"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) tensor
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block with self-attention and FFN"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x: (B, T, D) tensor
        Returns:
            (B, T, D) tensor
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block with self-attention and FFN"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            x: (B, T, D) tensor
        Returns:
            (B, T, D) tensor
        """
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        return x


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for motion sequences"""
    def __init__(self, in_channel, d_model, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_channel, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (B, T, in_channel) - motion sequence
            src_key_padding_mask: (B, T) - True for padded positions
        Returns:
            (B, T, d_model) - encoded features
        """
        # Project input to model dimension
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Transformer-based decoder for motion sequences"""
    def __init__(self, out_channel, d_model, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, out_channel)

    def forward(self, x, tgt_key_padding_mask=None):
        """
        Args:
            x: (B, T, d_model) - latent sequence
            tgt_key_padding_mask: (B, T) - True for padded positions
        Returns:
            (B, T, out_channel) - reconstructed motion
        """
        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, tgt_key_padding_mask=tgt_key_padding_mask)

        x = self.norm(x)

        # Project to output dimension
        return self.output_proj(x)


class VAELevel(nn.Module):
    """Single VAE level with continuous latent encoding (mean + logvar)"""
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

        # Project to mean and logvar for VAE
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)

    def encode(self, x):
        """
        Encode to mean and logvar
        Args:
            x: (B, T, D) format
        Returns:
            mu: (B, T, D)
            logvar: (B, T, D)
        """
        mu = self.fc_mu(x)  # (B, T, D)
        logvar = self.fc_logvar(x)  # (B, T, D)
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
            input: (B, T, D) format from encoder
        Returns:
            z: sampled latent (B, T, D)
            mu: mean (B, T, D)
            logvar: log variance (B, T, D)
        """
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VAE_Pose_1(nn.Module):
    """Single-level Transformer VAE for pose sequences"""
    def __init__(
            self,
            in_channel=165,  # Pose dimension
            d_model=256,  # Transformer hidden dimension (replaces 'channel')
            embed_dim=64,  # Latent dimension
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            # Legacy parameters (ignored, kept for config compatibility)
            channel=None,
            n_res_block=None,
            n_res_channel=None,
            stride=None,
            decay=None,
            use_smplx=False,
            smplx_model_path='models_smplx_v1_1/models',
    ):
        super().__init__()

        # Use d_model if provided, otherwise fall back to channel for backward compatibility
        if d_model is None and channel is not None:
            d_model = channel

        self.embed_dim = embed_dim

        # Transformer encoder
        self.enc = TransformerEncoder(
            in_channel, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
        )

        # Project encoder output to latent dimension
        self.to_latent = nn.Linear(d_model, embed_dim)

        # VAE level
        self.vae_level = VAELevel(embed_dim)

        # Project latent back to decoder dimension
        self.from_latent = nn.Linear(embed_dim, d_model)

        # Transformer decoder
        self.dec = TransformerDecoder(
            in_channel, d_model, nhead, num_decoder_layers, dim_feedforward, dropout
        )

        # Optional SMPLX layer for geometry supervision
        self.use_smplx = use_smplx
        if use_smplx:
            self.smplx_layer = SMPLXLayer(model_path=smplx_model_path)
        else:
            self.smplx_layer = None

    def forward(self, input, padding_mask=None, compute_geometry=False):
        """
        Args:
            input: (B, T, pose_dim) - motion sequence
            padding_mask: (B, T) - True for padded positions (optional)
            compute_geometry: Whether to compute SMPLX geometry
        Returns:
            dec: reconstructed motion (B, T, pose_dim)
            kl_loss: KL divergence loss
        """
        z, mu, logvar = self.encode(input, padding_mask)
        dec = self.decode(z, padding_mask)

        # Compute KL divergence
        kl_loss = self.kl_divergence(mu, logvar)

        # Optionally compute geometry through SMPLX
        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(dec)
            return dec, kl_loss, vertices, joints

        return dec, kl_loss

    def encode(self, input, padding_mask=None):
        """
        Args:
            input: (B, T, pose_dim)
            padding_mask: (B, T) - True for padded positions
        Returns:
            z, mu, logvar: all (B, T, embed_dim)
        """
        enc = self.enc(input, src_key_padding_mask=padding_mask)  # (B, T, d_model)
        latent = self.to_latent(enc)  # (B, T, embed_dim)
        z, mu, logvar = self.vae_level(latent)
        return z, mu, logvar

    def decode(self, z, padding_mask=None):
        """
        Args:
            z: (B, T, embed_dim)
            padding_mask: (B, T) - True for padded positions
        Returns:
            dec: (B, T, pose_dim)
        """
        dec_input = self.from_latent(z)  # (B, T, d_model)
        dec = self.dec(dec_input, tgt_key_padding_mask=padding_mask)
        return dec

    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()


class VAE_Pose_ML(nn.Module):
    """Multi-level hierarchical Transformer VAE for pose sequences

    Supports arbitrary length motion sequences through transformer architecture
    with sinusoidal positional encoding.
    """
    def __init__(
            self,
            in_channel=165,  # Pose dimension
            d_model=256,  # Transformer hidden dimension (replaces 'channel')
            embed_dim=64,  # Latent dimension per level
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            n_level=3,  # Number of hierarchical levels
            dropout=0.1,
            # Legacy parameters (ignored, kept for config compatibility)
            channel=None,
            n_res_block=None,
            n_res_channel=None,
            stride=None,
            decay=None,
            use_smplx=False,
            smplx_model_path='models_smplx_v1_1/models',
    ):
        super().__init__()
        self.device = 'cpu'

        # Use d_model if provided, otherwise fall back to channel for backward compatibility
        if d_model is None and channel is not None:
            d_model = channel

        self.embed_dim = embed_dim
        self.n_level = n_level

        # Transformer encoder
        self.enc = TransformerEncoder(
            in_channel, d_model, nhead, num_encoder_layers, dim_feedforward, dropout
        )

        # Project encoder output to latent dimension
        self.to_latent = nn.Linear(d_model, embed_dim)

        # Multiple VAE levels (continuous latent space)
        self.vae_levels = nn.ModuleList()
        self.level_norms = nn.ModuleList()

        for i in range(n_level):
            self.vae_levels.append(VAELevel(embed_dim))
            self.level_norms.append(nn.LayerNorm(embed_dim))

        # Project latent back to decoder dimension
        self.from_latent = nn.Linear(embed_dim, d_model)

        # Transformer decoder
        self.dec = TransformerDecoder(
            in_channel, d_model, nhead, num_decoder_layers, dim_feedforward, dropout
        )

        # Optional SMPLX layer for geometry supervision
        self.use_smplx = use_smplx
        if use_smplx:
            self.smplx_layer = SMPLXLayer(model_path=smplx_model_path)
        else:
            self.smplx_layer = None

    def forward(self, input, padding_mask=None, compute_geometry=False, return_intermediate=False):
        """
        Args:
            input: (B, T, pose_dim) - motion sequence (arbitrary length T)
            padding_mask: (B, T) - True for padded positions (optional)
            compute_geometry: Whether to compute SMPLX geometry
            return_intermediate: Whether to return intermediate reconstructions
        Returns:
            dec: reconstructed motion (B, T, pose_dim)
            total_kl: combined KL divergence from all levels
        """
        if return_intermediate:
            return self.forward_with_intermediate_outputs(input, padding_mask, compute_geometry)

        z, kl_losses = self.encode(input, padding_mask)
        dec = self.decode(z, padding_mask)

        # Combine KL losses from all levels
        total_kl = sum(kl_losses)

        # Optionally compute geometry through SMPLX
        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(dec)
            return dec, total_kl, vertices, joints

        return dec, total_kl

    def forward_with_intermediate_outputs(self, input, padding_mask=None, compute_geometry=False):
        """
        Forward pass that returns intermediate reconstructions at each level.
        Useful for progressive training with level-specific losses.
        """
        # Encode input
        enc = self.enc(input, src_key_padding_mask=padding_mask)  # (B, T, d_model)
        latent = self.to_latent(enc)  # (B, T, embed_dim)

        intermediate_outputs = []
        kl_losses = []

        # Hierarchical residual learning in latent space
        residual_latent = latent  # (B, T, D)
        accumulated_z = torch.zeros_like(latent)  # (B, T, D)

        for i in range(self.n_level):
            # VAE encode at this level
            z_i, mu_i, logvar_i = self.vae_levels[i](residual_latent)

            # KL divergence for this level
            kl_i = self.kl_divergence(mu_i, logvar_i)
            kl_losses.append(kl_i)

            # Accumulate latent codes (hierarchical residual)
            accumulated_z = accumulated_z + z_i  # (B, T, D)

            # Decode from accumulated latent up to this level
            level_dec = self.decode(accumulated_z, padding_mask)
            intermediate_outputs.append(level_dec)

            # Update residual for next level
            if i < self.n_level - 1:
                residual_latent = residual_latent - z_i

        # Combine KL losses
        total_kl = sum(kl_losses)

        # Optionally compute geometry for final output
        pred_vertices, pred_joints = None, None
        if compute_geometry and self.use_smplx:
            pred_vertices, pred_joints = self.smplx_layer(intermediate_outputs[-1])

        if compute_geometry and self.use_smplx:
            return intermediate_outputs, total_kl, pred_vertices, pred_joints

        return intermediate_outputs, total_kl

    def encode(self, input, padding_mask=None):
        """
        Encode input motion to hierarchical latent space.

        Args:
            input: (B, T, pose_dim) - arbitrary length motion
            padding_mask: (B, T) - True for padded positions
        Returns:
            accumulated_z: (B, T, embed_dim) - final latent
            kl_losses: list of KL losses per level
        """
        enc = self.enc(input, src_key_padding_mask=padding_mask)  # (B, T, d_model)
        latent = self.to_latent(enc)  # (B, T, embed_dim)

        # Multi-level hierarchical VAE encoding
        kl_losses = []
        residual_latent = latent  # (B, T, D)
        accumulated_z = torch.zeros_like(latent)  # (B, T, D)

        for i in range(self.n_level):
            # Encode residual at this level
            z_i, mu_i, logvar_i = self.vae_levels[i](residual_latent)

            # KL divergence for this level
            kl_i = self.kl_divergence(mu_i, logvar_i)
            kl_losses.append(kl_i)

            # Accumulate latent codes
            accumulated_z = accumulated_z + z_i

            # Update residual for next level
            if i < self.n_level - 1:
                residual_latent = residual_latent - z_i

        return accumulated_z, kl_losses

    def decode(self, z, padding_mask=None):
        """
        Decode latent to motion sequence.

        Args:
            z: (B, T, embed_dim)
            padding_mask: (B, T) - True for padded positions
        Returns:
            dec: (B, T, pose_dim)
        """
        dec_input = self.from_latent(z)  # (B, T, d_model)
        dec = self.dec(dec_input, tgt_key_padding_mask=padding_mask)
        return dec

    def kl_divergence(self, mu, logvar):
        """Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)"""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def decode_partial_levels(self, input, num_levels=None, padding_mask=None):
        """
        Encode and decode using only the first num_levels.
        This allows visualizing what each level learns.
        """
        if num_levels is None:
            num_levels = self.n_level
        num_levels = min(num_levels, self.n_level)

        enc = self.enc(input, src_key_padding_mask=padding_mask)
        latent = self.to_latent(enc)

        kl_losses = []
        residual_latent = latent
        accumulated_z = torch.zeros_like(latent)

        for i in range(num_levels):
            z_i, mu_i, logvar_i = self.vae_levels[i](residual_latent)

            kl_i = self.kl_divergence(mu_i, logvar_i)
            kl_losses.append(kl_i)

            accumulated_z = accumulated_z + z_i

            if i < num_levels - 1:
                residual_latent = residual_latent - z_i

        dec = self.decode(accumulated_z, padding_mask)
        total_kl = sum(kl_losses)

        return dec, total_kl

    def sample(self, batch_size, seq_len, device='cuda'):
        """
        Sample from the prior p(z) = N(0, I) and decode.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length (can be any length!)
            device: Device to generate samples on

        Returns:
            samples: (batch_size, seq_len, pose_dim)
        """
        # Sample from standard normal for all levels and accumulate
        accumulated_z = torch.zeros(batch_size, seq_len, self.embed_dim).to(device)

        for i in range(self.n_level):
            z_i = torch.randn(batch_size, seq_len, self.embed_dim).to(device)
            accumulated_z = accumulated_z + z_i

        # Decode
        samples = self.decode(accumulated_z)

        return samples

    def encode_to_latents(self, input, padding_mask=None):
        """
        Encode input and return latent codes from each level separately.
        Useful for analysis and visualization.

        Args:
            input: (B, T, pose_dim)
            padding_mask: (B, T)
        Returns:
            latents: list of (z_i, mu_i, logvar_i) for each level
        """
        enc = self.enc(input, src_key_padding_mask=padding_mask)
        latent = self.to_latent(enc)

        latents = []
        residual_latent = latent

        for i in range(self.n_level):
            z_i, mu_i, logvar_i = self.vae_levels[i](residual_latent)
            latents.append((z_i, mu_i, logvar_i))

            if i < self.n_level - 1:
                residual_latent = residual_latent - z_i

        return latents


def create_padding_mask(lengths, max_len, device='cuda'):
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: (B,) tensor of actual sequence lengths
        max_len: maximum sequence length (padded length)
        device: device for the mask

    Returns:
        mask: (B, max_len) - True for padded positions
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    return mask


if __name__ == '__main__':
    # Test the model with different sequence lengths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VAE_Pose_ML(
        in_channel=165,
        d_model=256,
        embed_dim=64,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        n_level=3,
        dropout=0.1,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with different sequence lengths
    for seq_len in [32, 64, 128, 256]:
        x = torch.randn(2, seq_len, 165).to(device)

        # Forward pass
        dec, kl = model(x)
        print(f"Input: {x.shape} -> Output: {dec.shape}, KL: {kl.item():.4f}")

        # Test with intermediate outputs
        intermediates, kl = model(x, return_intermediate=True)
        print(f"  Intermediate outputs: {[i.shape for i in intermediates]}")

    # Test sampling at different lengths
    for seq_len in [50, 100, 200]:
        samples = model.sample(4, seq_len, device)
        print(f"Sampled: {samples.shape}")

    # Test with padding mask (variable length batching)
    print("\nTest with padding mask:")
    batch = torch.randn(3, 100, 165).to(device)  # Padded to max length 100
    lengths = torch.tensor([100, 75, 50]).to(device)  # Actual lengths
    mask = create_padding_mask(lengths, 100, device)

    dec, kl = model(batch, padding_mask=mask)
    print(f"Padded batch: {batch.shape} -> Output: {dec.shape}")
