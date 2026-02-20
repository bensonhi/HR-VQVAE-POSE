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


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding with length interpolation for arbitrary lengths"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) tensor
        Returns:
            x + positional encoding (interpolated if T > max_len)
        """
        T = x.size(1)
        if T <= self.max_len:
            pos_enc = self.pe[:, :T, :]
        else:
            # Interpolate for longer sequences
            pos_enc = F.interpolate(
                self.pe.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)
        return self.dropout(x + pos_enc)


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
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class TransformerDecoderBlockWithCrossAttn(nn.Module):
    """Transformer decoder block with self-attention, cross-attention to latent, and FFN"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            x: (B, T, D) - target sequence (positional queries)
            memory: (B, M, D) - latent memory to cross-attend to
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to latent memory
        cross_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + self.dropout(cross_out))

        # FFN
        x = self.norm3(x + self.ffn(x))
        return x


class DualCrossAttnDecoderBlock(nn.Module):
    """Decoder block with self-attn, dual cross-attn (z + audio), and FFN"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_z = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_audio = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, z_memory, audio_memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Args:
            x: (B, T, D) - target sequence
            z_memory: (B, M, D) - latent memory tokens
            audio_memory: (B, T_audio, D) - projected audio features, or None
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to z (always active — dedicated pathway)
        cross_z_out, _ = self.cross_attn_z(x, z_memory, z_memory)
        x = self.norm2(x + self.dropout(cross_z_out))

        # Cross-attention to audio (skipped when audio_memory is None, e.g. sampling)
        if audio_memory is not None:
            cross_audio_out, _ = self.cross_attn_audio(x, audio_memory, audio_memory)
            x = self.norm3(x + self.dropout(cross_audio_out))
        else:
            x = self.norm3(x)

        # FFN
        x = self.norm4(x + self.ffn(x))
        return x


class AdaLNCrossAttnDecoderBlock(nn.Module):
    """Decoder block with AdaLN z-conditioning + single cross-attn for audio/z memory.

    z conditions layer norms via learned scale/shift (DiT-style AdaLN).
    This avoids adding a time-independent constant to the residual stream,
    preserving temporal variation from audio cross-attention.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        # Non-affine LayerNorms (AdaLN provides scale/shift from z)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.dropout_module = nn.Dropout(dropout)
        # AdaLN modulation: z_embed -> (scale1, shift1, scale2, shift2, scale3, shift3)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init so AdaLN starts as identity (scale=0->1+0=1, shift=0)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, z_embed, memory, tgt_key_padding_mask=None):
        """
        Args:
            x: (B, T, D) - queries
            z_embed: (B, D) - latent conditioning (broadcast across time)
            memory: (B, M, D) - cross-attn memory (z_tokens+audio or z_tokens only)
        """
        mod = self.adaLN_modulation(z_embed)  # (B, 6*D)
        s1, b1, s2, b2, s3, b3 = [m.unsqueeze(1) for m in mod.chunk(6, dim=-1)]

        # Pre-norm self-attention with AdaLN
        x_mod = self.norm1(x) * (1 + s1) + b1
        attn_out, _ = self.self_attn(x_mod, x_mod, x_mod, key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout_module(attn_out)

        # Pre-norm cross-attention with AdaLN
        x_mod = self.norm2(x) * (1 + s2) + b2
        cross_out, _ = self.cross_attn(x_mod, memory, memory)
        x = x + self.dropout_module(cross_out)

        # Pre-norm FFN with AdaLN
        x_mod = self.norm3(x) * (1 + s3) + b3
        x = x + self.ffn(x_mod)
        return x


class LengthEmbedding(nn.Module):
    """Embedding for sequence length conditioning"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.max_len = max_len
        # Learnable length embeddings
        self.length_embed = nn.Embedding(max_len, d_model)
        # Also project continuous length for extrapolation
        self.length_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, length):
        """
        Args:
            length: int or (B,) tensor of lengths
        Returns:
            (B, D) length embedding
        """
        if isinstance(length, int):
            length = torch.tensor([length])

        device = self.length_embed.weight.device
        length = length.to(device)

        if length.dim() == 0:
            length = length.unsqueeze(0)

        B = length.size(0)

        # Use learned embedding for lengths within range
        clamped_length = length.clamp(0, self.max_len - 1)
        discrete_embed = self.length_embed(clamped_length)  # (B, D)

        # Also compute continuous embedding for extrapolation
        normalized_length = length.float().unsqueeze(-1) / self.max_len  # (B, 1)
        continuous_embed = self.length_proj(normalized_length)  # (B, D)

        # Combine both
        return discrete_embed + continuous_embed


class GestureTypeEmbedding(nn.Module):
    """Embedding for gesture type conditioning (beat=0, semantic=1)"""
    def __init__(self, d_model, num_types=2):
        super().__init__()
        self.embed = nn.Embedding(num_types, d_model)

    def forward(self, gesture_type):
        """
        Args:
            gesture_type: (B,) tensor of gesture type indices
        Returns:
            (B, d_model) gesture type embedding
        """
        return self.embed(gesture_type)


class AudioProjection(nn.Module):
    """Project audio features to model dimension with layer norm"""
    def __init__(self, audio_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(audio_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, audio_dim)
        Returns:
            (B, T, d_model)
        """
        return self.norm(self.proj(x))


class GlobalEncoder(nn.Module):
    """Transformer encoder that pools sequence into a global latent"""
    def __init__(self, in_channel, d_model, latent_dim, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, audio_dim=None):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_channel, d_model)

        # Audio projection for additive fusion at input (NEW)
        self.audio_input_proj = None
        if audio_dim is not None:
            self.audio_input_proj = AudioProjection(audio_dim, d_model)

        # Gesture type embedding (NEW)
        self.gesture_embed = GestureTypeEmbedding(d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Learnable [CLS] token for global pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Project to latent space (mu and logvar)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, x, padding_mask=None, gesture_type=None, audio_features=None):
        """
        Args:
            x: (B, T, in_channel) - motion sequence
            padding_mask: (B, T) - True for padded positions
            gesture_type: (B,) - gesture type indices (0=beat, 1=semantic), optional
            audio_features: (B, T, audio_dim) - audio features, optional
        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        B, T, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, d_model)

        # Additive audio fusion (NEW)
        if audio_features is not None and self.audio_input_proj is not None:
            x = x + self.audio_input_proj(audio_features)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, d_model)

        # Update padding mask for [CLS] token
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)

        x = self.norm(x)

        # Extract [CLS] token as global representation
        cls_output = x[:, 0, :]  # (B, d_model)

        # Add gesture type embedding (NEW)
        if gesture_type is not None:
            cls_output = cls_output + self.gesture_embed(gesture_type)

        # Project to mu and logvar
        mu = self.fc_mu(cls_output)  # (B, latent_dim)
        logvar = self.fc_logvar(cls_output)  # (B, latent_dim)

        return mu, logvar


class LengthConditionedDecoder(nn.Module):
    """Transformer decoder that generates motion conditioned on global latent and target length"""
    def __init__(self, out_channel, d_model, latent_dim, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, max_len=1024, audio_dim=None,
                 num_memory_tokens=4,
                 memory_dropout_prob=0.0):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.memory_dropout_prob = memory_dropout_prob

        # Length embedding
        self.length_embed = LengthEmbedding(d_model, max_len)

        # Gesture type embedding (NEW)
        self.gesture_embed = GestureTypeEmbedding(d_model)

        # Audio projection and positional encoding for decoder memory (NEW)
        self.audio_proj = None
        self.audio_pos_encoder = None
        if audio_dim is not None:
            self.audio_proj = AudioProjection(audio_dim, d_model)
            self.audio_pos_encoder = PositionalEncoding(d_model, max_len=max_len * 2, dropout=dropout)

        # Project latent to memory tokens
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # Create multiple memory tokens from single latent (richer conditioning)
        self.num_memory_tokens = num_memory_tokens
        self.memory_expand = nn.Linear(d_model, d_model * self.num_memory_tokens)

        # Learnable query tokens (will be expanded to target length)
        self.query_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding for queries
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len * 2, dropout=dropout)

        # Transformer decoder layers with AdaLN z-conditioning + unified cross-attention
        self.layers = nn.ModuleList([
            AdaLNCrossAttnDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, out_channel)

    def forward(self, z, target_length, padding_mask=None, gesture_type=None,
                lengths=None, audio_features=None):
        """
        Args:
            z: (B, latent_dim) - global latent
            target_length: int - desired output sequence length
            padding_mask: (B, T) - optional padding mask
            gesture_type: (B,) - gesture type indices, optional
            lengths: (B,) - per-sample true lengths, optional (overrides uniform target_length)
            audio_features: (B, T, audio_dim) - audio features, optional
        Returns:
            (B, target_length, out_channel)
        """
        B = z.size(0)
        device = z.device

        # z_embed for AdaLN: latent + length + gesture conditioning
        z_embed = self.latent_proj(z)  # (B, d_model)
        if lengths is not None:
            z_embed = z_embed + self.length_embed(lengths)
        else:
            z_embed = z_embed + self.length_embed(torch.tensor([target_length] * B, device=device))
        if gesture_type is not None:
            z_embed = z_embed + self.gesture_embed(gesture_type)

        # z memory tokens for cross-attention
        z_memory = self.memory_expand(z_embed)  # (B, d_model * M)
        z_memory = z_memory.view(B, self.num_memory_tokens, self.d_model)

        # Memory token dropout (inverted dropout, classifier-free guidance style)
        if self.training and self.memory_dropout_prob > 0:
            keep_prob = 1.0 - self.memory_dropout_prob
            mask = torch.bernoulli(torch.full(
                (B, self.num_memory_tokens, 1), keep_prob, device=device))
            z_memory = z_memory * mask / keep_prob

        # Build cross-attention memory: [z_tokens, audio_tokens] or [z_tokens]
        if audio_features is not None and self.audio_proj is not None:
            audio_memory = self.audio_pos_encoder(self.audio_proj(audio_features))
            memory = torch.cat([z_memory, audio_memory], dim=1)  # (B, M+T, D)
        else:
            memory = z_memory  # (B, M, D) — sampling mode (ACTOR-style)

        # Queries with positional encoding
        queries = self.pos_encoder(self.query_embed.expand(B, target_length, -1))

        # Decoder layers with AdaLN + unified cross-attention
        x = queries
        for layer in self.layers:
            x = layer(x, z_embed=z_embed, memory=memory, tgt_key_padding_mask=padding_mask)

        x = self.norm(x)
        return self.output_proj(x)


class VAELevel(nn.Module):
    """Single VAE level for hierarchical VAE (kept for compatibility)"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)

    def forward(self, input):
        mu = self.fc_mu(input)
        logvar = self.fc_logvar(input)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class LengthConditionedVAE(nn.Module):
    """
    Length-Conditioned Hierarchical VAE for Motion Generation

    Key features:
    - Global latent z captures motion content/style
    - Length conditioning controls output duration
    - Gesture type conditioning (beat vs semantic)
    - Audio conditioning (Wav2Vec 2.0 features)
    - Hierarchical levels for coarse-to-fine generation

    Sampling:
        z = torch.randn(batch_size, latent_dim)
        motion = model.decode(z, length=200)  # Generate 200 frames
    """
    def __init__(
            self,
            in_channel=165,
            d_model=256,
            latent_dim=256,
            embed_dim=64,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            n_level=3,
            dropout=0.1,
            max_len=1024,
            audio_dim=None,
            target_kl=50.0,
            num_memory_tokens=4,
            condition_dropout_prob=0.0,
            memory_dropout_prob=0.0,
            # Legacy parameters (ignored)
            free_bits=None,
            channel=None,
            n_res_block=None,
            n_res_channel=None,
            stride=None,
            decay=None,
            use_smplx=False,
            smplx_model_path='models_smplx_v1_1/models',
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.n_level = n_level
        self.max_len = max_len
        self.target_kl = target_kl
        self.condition_dropout_prob = condition_dropout_prob

        # Global encoder: sequence -> single latent
        self.encoder = GlobalEncoder(
            in_channel, d_model, latent_dim,
            nhead, num_encoder_layers, dim_feedforward, dropout,
            audio_dim=audio_dim,
        )

        # Hierarchical VAE levels on the global latent
        self.vae_levels = nn.ModuleList()
        for i in range(n_level):
            level_dim = latent_dim // n_level
            self.vae_levels.append(VAELevel(level_dim))

        # Length-conditioned decoder: latent + length -> sequence
        self.decoder = LengthConditionedDecoder(
            in_channel, d_model, latent_dim,
            nhead, num_decoder_layers, dim_feedforward, dropout, max_len,
            audio_dim=audio_dim,
            num_memory_tokens=num_memory_tokens,
            memory_dropout_prob=memory_dropout_prob,
        )

        # Optional SMPLX layer
        self.use_smplx = use_smplx
        if use_smplx:
            self.smplx_layer = SMPLXLayer(model_path=smplx_model_path)
        else:
            self.smplx_layer = None

    def encode(self, x, padding_mask=None, gesture_type=None, audio_features=None):
        """
        Encode motion sequence to global latent.

        Args:
            x: (B, T, in_channel) - motion sequence
            padding_mask: (B, T) - optional padding mask
            gesture_type: (B,) - gesture type indices, optional
            audio_features: (B, T, audio_dim) - audio features, optional

        Returns:
            z: (B, latent_dim) - sampled latent
            mu: (B, latent_dim) - mean
            logvar: (B, latent_dim) - log variance
            kl_loss: scalar KL loss (target-KL-adjusted)
            raw_kl: scalar raw KL (for logging, no target clamping)
        """
        # Get global mu and logvar
        mu, logvar = self.encoder(x, padding_mask, gesture_type=gesture_type,
                                  audio_features=audio_features)

        # Reparameterize the full latent
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (B, latent_dim)

        # Compute raw KL per level (no per-dimension clamping)
        level_dim = self.latent_dim // self.n_level
        kl_losses = []

        for i in range(self.n_level):
            start_idx = i * level_dim
            end_idx = start_idx + level_dim
            if i == self.n_level - 1:
                end_idx = self.latent_dim

            mu_i = mu[:, start_idx:end_idx]
            logvar_i = logvar[:, start_idx:end_idx]

            # Per-dimension KL: (B, D_level)
            kl_per_dim = -0.5 * (1 + logvar_i - mu_i.pow(2) - logvar_i.exp())

            # Sum over dimensions, mean over batch (raw, no clamping)
            kl_i = kl_per_dim.sum(dim=-1).mean()
            kl_losses.append(kl_i)

        # Total raw KL across all levels
        total_raw_kl = sum(kl_losses)

        # Apply target KL: max(target, actual)
        # When actual < target: loss = target (constant, no gradient) → only recon gradient
        # When actual > target: loss = actual (normal KL gradient pushes toward prior)
        if self.target_kl > 0:
            kl_for_loss = torch.max(torch.tensor(self.target_kl, device=mu.device), total_raw_kl)
        else:
            kl_for_loss = total_raw_kl

        return z, mu, logvar, kl_for_loss, total_raw_kl, kl_losses

    def decode(self, z, length, padding_mask=None, gesture_type=None, lengths=None,
               audio_features=None):
        """
        Decode global latent to motion sequence of specified length.

        Args:
            z: (B, latent_dim) - global latent
            length: int - desired output length
            padding_mask: (B, T) - optional padding mask
            gesture_type: (B,) - gesture type indices, optional
            lengths: (B,) - per-sample true lengths, optional
            audio_features: (B, T, audio_dim) - audio features, optional

        Returns:
            (B, length, out_channel) - generated motion
        """
        return self.decoder(z, length, padding_mask, gesture_type=gesture_type,
                           lengths=lengths, audio_features=audio_features)

    def forward(self, x, padding_mask=None, gesture_type=None, lengths=None,
                audio_features=None, compute_geometry=False, return_intermediate=False):
        """
        Forward pass: encode input, decode to same length.

        Args:
            x: (B, T, in_channel) - input motion
            padding_mask: (B, T) - optional padding mask
            gesture_type: (B,) - gesture type indices, optional
            lengths: (B,) - per-sample true lengths, optional
            audio_features: (B, T, audio_dim) - audio features, optional
            compute_geometry: whether to compute SMPLX geometry
            return_intermediate: whether to return intermediate outputs per level

        Returns:
            recon: (B, T, in_channel) - reconstructed motion
            kl_loss: target-KL-adjusted KL divergence (for loss computation)
            raw_kl: raw KL divergence (for logging)
        """
        B, T, _ = x.shape

        # Condition dropout: drop audio + gesture_type to force reliance on z
        if self.training and self.condition_dropout_prob > 0:
            if torch.rand(1).item() < self.condition_dropout_prob:
                audio_features = None
                gesture_type = None

        # Encode
        z, mu, logvar, kl_loss, raw_kl, kl_per_level = self.encode(
            x, padding_mask, gesture_type=gesture_type, audio_features=audio_features)

        if return_intermediate:
            return self._forward_with_intermediate(z, T, kl_loss, raw_kl, x, padding_mask,
                                                   compute_geometry, gesture_type=gesture_type,
                                                   lengths=lengths, audio_features=audio_features,
                                                   kl_per_level=kl_per_level)

        # Decode to same length as input
        recon = self.decode(z, T, padding_mask, gesture_type=gesture_type,
                           lengths=lengths, audio_features=audio_features)

        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(recon)
            return recon, kl_loss, raw_kl, vertices, joints

        return recon, kl_loss, raw_kl

    def _forward_with_intermediate(self, z, length, kl_loss, raw_kl, x, padding_mask,
                                   compute_geometry, gesture_type=None, lengths=None,
                                   audio_features=None, kl_per_level=None):
        """Forward with intermediate outputs for progressive training."""
        intermediate_outputs = []
        level_dim = self.latent_dim // self.n_level

        for i in range(self.n_level):
            end_idx = (i + 1) * level_dim
            if i == self.n_level - 1:
                end_idx = self.latent_dim

            z_partial = z.clone()
            if end_idx < self.latent_dim:
                z_partial[:, end_idx:] = 0

            level_output = self.decode(z_partial, length, padding_mask,
                                       gesture_type=gesture_type, lengths=lengths,
                                       audio_features=audio_features)
            intermediate_outputs.append(level_output)

        if compute_geometry and self.use_smplx:
            vertices, joints = self.smplx_layer(intermediate_outputs[-1])
            return intermediate_outputs, kl_loss, raw_kl, vertices, joints, kl_per_level

        return intermediate_outputs, kl_loss, raw_kl, kl_per_level

    def sample(self, batch_size, length, device='cuda', temperature=1.0,
               gesture_type=None, audio_features=None):
        """
        Sample motion sequences of specified length.

        Args:
            batch_size: number of samples
            length: desired sequence length
            device: device to generate on
            temperature: sampling temperature
            gesture_type: (B,) - gesture type indices, optional
            audio_features: (B, T, audio_dim) - audio features, optional

        Returns:
            (batch_size, length, out_channel) - generated motion
        """
        z = torch.randn(batch_size, self.latent_dim, device=device) * temperature
        return self.decode(z, length, gesture_type=gesture_type,
                          audio_features=audio_features)

    def interpolate(self, x1, x2, num_steps=10, length=None, gesture_type=None,
                    audio_features=None):
        """
        Interpolate between two motion sequences in latent space.

        Args:
            x1: (1, T1, D) - first motion
            x2: (1, T2, D) - second motion
            num_steps: number of interpolation steps
            length: output length (default: average of T1 and T2)
            gesture_type: (B,) - gesture type indices, optional
            audio_features: (B, T, audio_dim) - audio features, optional

        Returns:
            (num_steps, length, D) - interpolated motions
        """
        if length is None:
            length = (x1.size(1) + x2.size(1)) // 2

        z1, _, _, _, _ = self.encode(x1)
        z2, _, _, _, _ = self.encode(x2)

        alphas = torch.linspace(0, 1, num_steps, device=z1.device)
        interpolated = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            motion = self.decode(z_interp, length, gesture_type=gesture_type,
                                audio_features=audio_features)
            interpolated.append(motion)

        return torch.cat(interpolated, dim=0)

    def kl_divergence(self, mu, logvar):
        """Compute KL divergence for full latent."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def decode_partial_levels(self, x, num_levels=None, padding_mask=None,
                              gesture_type=None, lengths=None, audio_features=None):
        """Decode using only first num_levels (for visualization)."""
        if num_levels is None:
            num_levels = self.n_level
        num_levels = min(num_levels, self.n_level)

        B, T, _ = x.shape
        z, mu, logvar, kl_loss, raw_kl = self.encode(x, padding_mask, gesture_type=gesture_type,
                                                      audio_features=audio_features)

        level_dim = self.latent_dim // self.n_level
        end_idx = num_levels * level_dim
        if num_levels == self.n_level:
            end_idx = self.latent_dim

        z_partial = z.clone()
        if end_idx < self.latent_dim:
            z_partial[:, end_idx:] = 0

        recon = self.decode(z_partial, T, padding_mask, gesture_type=gesture_type,
                           lengths=lengths, audio_features=audio_features)

        return recon, kl_loss


# Aliases for backward compatibility
VAE_Pose_ML = LengthConditionedVAE
VAE_Pose_1 = LengthConditionedVAE


def create_padding_mask(lengths, max_len, device='cuda'):
    """Create padding mask from sequence lengths."""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    return mask


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test with new model size and audio/gesture conditioning
    model = LengthConditionedVAE(
        in_channel=165,
        d_model=512,
        latent_dim=512,
        embed_dim=64,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        n_level=3,
        dropout=0.1,
        audio_dim=768,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test reconstruction with audio + gesture_type + padding_mask
    print("\n=== Reconstruction Test with Audio + Gesture Type ===")
    for seq_len in [32, 64, 128]:
        x = torch.randn(2, seq_len, 165).to(device)
        audio = torch.randn(2, seq_len, 768).to(device)
        gesture_type = torch.tensor([0, 1], dtype=torch.long, device=device)
        lengths = torch.tensor([seq_len, seq_len], dtype=torch.long, device=device)

        recon, kl = model(x, gesture_type=gesture_type, lengths=lengths, audio_features=audio)
        print(f"Input: {x.shape} -> Recon: {recon.shape}, KL: {kl.item():.4f}")

    # Test with padding mask (variable length)
    print("\n=== Variable Length with Padding Mask ===")
    B, T_max = 4, 100
    actual_lengths = torch.tensor([30, 50, 80, 100], dtype=torch.long, device=device)
    padding_mask = create_padding_mask(actual_lengths, T_max, device)
    x = torch.randn(B, T_max, 165).to(device)
    audio = torch.randn(B, T_max, 768).to(device)
    gesture_type = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device)

    recon, kl = model(x, padding_mask=padding_mask, gesture_type=gesture_type,
                      lengths=actual_lengths, audio_features=audio)
    print(f"Variable-length input: {x.shape} -> Recon: {recon.shape}, KL: {kl.item():.4f}")

    # Test without audio (backward compat)
    print("\n=== Backward Compatibility (no audio) ===")
    model_compat = LengthConditionedVAE(
        in_channel=165, d_model=256, latent_dim=256, embed_dim=64,
        nhead=8, num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=1024, n_level=3, dropout=0.1,
    ).to(device)
    print(f"Compat model parameters: {sum(p.numel() for p in model_compat.parameters()):,}")

    x = torch.randn(2, 64, 165).to(device)
    recon, kl = model_compat(x)
    print(f"No audio: Input: {x.shape} -> Recon: {recon.shape}, KL: {kl.item():.4f}")

    # Test generation
    print("\n=== Generation Test ===")
    for length in [50, 100, 200]:
        samples = model.sample(batch_size=4, length=length, device=device,
                              gesture_type=torch.zeros(4, dtype=torch.long, device=device))
        print(f"Generated: {samples.shape}")

    # Test intermediate outputs
    print("\n=== Intermediate Outputs Test ===")
    x = torch.randn(2, 64, 165).to(device)
    audio = torch.randn(2, 64, 768).to(device)
    gesture_type = torch.tensor([0, 1], dtype=torch.long, device=device)
    intermediates, kl = model(x, gesture_type=gesture_type, audio_features=audio,
                              return_intermediate=True)
    print(f"Intermediate outputs: {[i.shape for i in intermediates]}")
