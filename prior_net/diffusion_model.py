"""
Flow-matching prior for VAE latents.

Instead of regressing mu from audio, models the full conditional distribution
p(z | audio, speaker, gesture) using conditional flow matching (rectified flow).

Two variants:
  - FlowMatchingPrior: MLP denoiser for single-z (B, latent_dim)
  - TemporalFlowMatchingPrior: Transformer denoiser for temporal z (B, T', latent_dim)
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


# ==========================================================================
# Components
# ==========================================================================

class SinusoidalTimestepEmbedding(nn.Module):
    """Maps scalar t in [0,1] to a d_model-dim embedding."""

    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t):
        """t: (B,) or (B,1) -> (B, d_model)"""
        t = t.view(-1)  # (B,)
        args = t[:, None] * self.freqs[None, :]  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        return self.mlp(emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AudioEncoder(nn.Module):
    """Encodes variable-length audio into a fixed-size context vector via CLS token."""

    def __init__(self, audio_dim=768, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, num_speakers=31,
                 num_gesture_types=2, max_len=1024):
        super().__init__()
        self.d_model = d_model

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.speaker_embed = nn.Embedding(num_speakers, d_model)
        self.gesture_embed = nn.Embedding(num_gesture_types, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio_features, speaker_id, gesture_type, padding_mask=None):
        """Returns: context (B, d_model)"""
        B = audio_features.shape[0]

        x = self.audio_proj(audio_features)
        cond = self.speaker_embed(speaker_id) + self.gesture_embed(gesture_type)
        x = x + cond.unsqueeze(1)
        x = self.pos_encoder(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x[:, 0, :]  # CLS output


class AdaLNMLPBlock(nn.Module):
    """MLP block with AdaLN conditioning from timestep + audio context."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN = nn.Linear(cond_dim, 2 * hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Zero-init AdaLN so initial output = identity
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        # Set shift to 0, scale to 1 at init
        with torch.no_grad():
            self.adaLN.bias[:hidden_dim] = 1.0  # scale init = 1

    def forward(self, x, cond):
        """x: (B, hidden_dim), cond: (B, cond_dim) -> (B, hidden_dim)"""
        scale_shift = self.adaLN(cond)  # (B, 2*hidden)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.norm(x)
        h = scale * h + shift
        h = self.fc(h)
        return x + h  # residual


# ==========================================================================
# Main model
# ==========================================================================

class FlowMatchingPrior(nn.Module):
    """Conditional flow matching model for p(z | audio, speaker, gesture).

    Training: learns velocity field v(z_t, t, cond) for the ODE dz/dt = v.
    Inference: integrates from z_0 ~ N(0,I) to z_1 ~ p(z|audio).
    """

    def __init__(
        self,
        latent_dim=128,
        audio_dim=768,
        d_model=256,
        nhead=4,
        num_audio_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        num_speakers=31,
        num_gesture_types=2,
        hidden_dim=512,
        num_mlp_blocks=6,
        max_len=1024,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        # Audio encoder (processes audio once, cached during sampling)
        self.audio_encoder = AudioEncoder(
            audio_dim=audio_dim, d_model=d_model, nhead=nhead,
            num_layers=num_audio_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, num_speakers=num_speakers,
            num_gesture_types=num_gesture_types, max_len=max_len,
        )

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(d_model)

        # Null conditioning for classifier-free guidance
        self.null_context = nn.Parameter(torch.zeros(1, d_model))

        # z input projection
        self.z_proj = nn.Linear(latent_dim, hidden_dim)

        # Conditioning projection (t_embed + audio_context -> cond_dim)
        cond_dim = d_model * 2  # concat of time and audio
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        # AdaLN MLP blocks
        self.blocks = nn.ModuleList([
            AdaLNMLPBlock(hidden_dim, cond_dim) for _ in range(num_mlp_blocks)
        ])

        # Output projection
        self.out_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.out_adaLN = nn.Linear(cond_dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

        # Zero-init output so initial prediction = 0
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_adaLN.weight)
        nn.init.zeros_(self.out_adaLN.bias)
        with torch.no_grad():
            self.out_adaLN.bias[:hidden_dim] = 1.0

        # z normalization stats (set after data extraction)
        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))

    def encode_audio(self, audio_features, speaker_id, gesture_type, padding_mask=None):
        """Pre-compute audio context (call once, reuse across ODE steps)."""
        return self.audio_encoder(audio_features, speaker_id, gesture_type, padding_mask)

    def predict_velocity(self, z_t, t, audio_context, cond=None):
        """Predict velocity field v(z_t, t, audio_context).

        Args:
            z_t: (B, latent_dim) - current noisy latent
            t: (B,) - timestep in [0, 1]
            audio_context: (B, d_model) - pre-computed audio encoding
            cond: (B, cond_dim) - optional precomputed conditioning

        Returns:
            v: (B, latent_dim) - predicted velocity
        """
        # Build conditioning
        if cond is None:
            t_emb = self.time_embed(t)  # (B, d_model)
            cond = self.cond_proj(torch.cat([t_emb, audio_context], dim=-1))

        # Process z through AdaLN MLP
        h = self.z_proj(z_t)  # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, cond)

        # Output
        scale_shift = self.out_adaLN(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = self.out_norm(h)
        h = scale * h + shift
        v = self.out_proj(h)
        return v

    def compute_loss(self, z_1, audio_features, speaker_id, gesture_type,
                     padding_mask=None, cond_drop_prob=0.1):
        """Flow matching training loss.

        Args:
            z_1: (B, latent_dim) - target latent (VAE encoder mu)
            audio_features: (B, T, audio_dim)
            speaker_id, gesture_type: (B,)
            padding_mask: (B, T)
            cond_drop_prob: probability of dropping conditioning (for CFG)

        Returns:
            loss: scalar MSE on velocity prediction
        """
        B = z_1.shape[0]
        device = z_1.device

        # Normalize z
        z_1_norm = (z_1 - self.z_mean) / self.z_std

        # Sample noise and time
        z_0 = torch.randn_like(z_1_norm)
        # Logit-normal time distribution (more weight near 0 and 1)
        t = torch.sigmoid(torch.randn(B, device=device) * 0.5)
        t = t.clamp(1e-5, 1 - 1e-5)

        # Interpolate
        t_expand = t[:, None]  # (B, 1)
        z_t = (1 - t_expand) * z_0 + t_expand * z_1_norm
        v_target = z_1_norm - z_0

        # Audio context with optional CFG dropout
        audio_context = self.encode_audio(
            audio_features, speaker_id, gesture_type, padding_mask)

        # Random conditioning dropout for CFG
        if cond_drop_prob > 0 and self.training:
            drop_mask = torch.rand(B, device=device) < cond_drop_prob
            if drop_mask.any():
                null = self.null_context.expand(B, -1)
                audio_context = torch.where(
                    drop_mask[:, None], null, audio_context)

        # Predict velocity
        v_pred = self.predict_velocity(z_t, t, audio_context)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def sample(self, audio_features, speaker_id, gesture_type,
               padding_mask=None, num_steps=50, guidance_scale=1.0):
        """Sample z by integrating the learned ODE.

        Args:
            audio_features: (B, T, audio_dim)
            speaker_id, gesture_type: (B,)
            num_steps: number of Euler steps
            guidance_scale: CFG scale (1.0 = no guidance)

        Returns:
            z: (B, latent_dim) - sampled latent
        """
        B = audio_features.shape[0]
        device = audio_features.device

        # Encode audio once
        audio_context = self.encode_audio(
            audio_features, speaker_id, gesture_type, padding_mask)

        # For CFG: also compute unconditional context
        if guidance_scale != 1.0:
            null_context = self.null_context.expand(B, -1)

        # Start from noise
        z = torch.randn(B, self.latent_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)

            if guidance_scale != 1.0:
                # Classifier-free guidance
                v_cond = self.predict_velocity(z, t, audio_context)
                v_uncond = self.predict_velocity(z, t, null_context)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = self.predict_velocity(z, t, audio_context)

            # Midpoint method for better accuracy
            z_mid = z + v * (dt / 2)
            t_mid = torch.full((B,), (i + 0.5) * dt, device=device)

            if guidance_scale != 1.0:
                v_cond = self.predict_velocity(z_mid, t_mid, audio_context)
                v_uncond = self.predict_velocity(z_mid, t_mid, null_context)
                v_mid = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_mid = self.predict_velocity(z_mid, t_mid, audio_context)

            z = z + v_mid * dt

        # Denormalize
        z = z * self.z_std + self.z_mean
        return z


# ==========================================================================
# Temporal Flow Matching Prior
# ==========================================================================

class TemporalAudioEncoder(nn.Module):
    """Encodes audio into temporal features (B, T', d_model) via transformer + Conv1d."""

    def __init__(self, audio_dim=768, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, num_speakers=31,
                 num_gesture_types=2, max_len=1024, temporal_downsample=8,
                 pose_dim=165, text_dim=0):
        super().__init__()
        self.d_model = d_model
        self.temporal_downsample = temporal_downsample
        self.text_dim = text_dim

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Optional text conditioning (pre-computed BERT embeddings)
        self.text_proj = None
        if text_dim > 0:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, d_model),
                nn.LayerNorm(d_model),
            )
        self.speaker_embed = nn.Embedding(num_speakers, d_model)
        self.gesture_embed = nn.Embedding(num_gesture_types, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Anchor conditioning
        self.anchor_pose_proj = nn.Sequential(
            nn.Linear(pose_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.anchor_audio_proj = nn.Sequential(
            nn.Linear(audio_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.anchor_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Strided conv to match z temporal resolution
        self.downsample = nn.Conv1d(
            d_model, d_model,
            kernel_size=temporal_downsample,
            stride=temporal_downsample,
        )
        self.downsample_post = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def _encode_anchor(self, anchor_frames, anchor_audio):
        if anchor_frames is None:
            return None
        anchor_emb = self.anchor_pose_proj(anchor_frames)
        if anchor_audio is not None:
            anchor_emb = anchor_emb + self.anchor_audio_proj(anchor_audio)
        pooled = anchor_emb.mean(dim=1)
        return self.anchor_gate(pooled) * pooled

    def forward(self, audio_features, speaker_id, gesture_type, padding_mask=None,
                anchor_frames=None, anchor_audio=None, text_features=None):
        """Returns: (B, T', d_model) temporal audio context."""
        B, T, _ = audio_features.shape

        x = self.audio_proj(audio_features)

        # Additive text fusion (pre-computed BERT embeddings)
        if self.text_proj is not None and text_features is not None:
            # text_features: (B, T, text_dim) — same temporal resolution as audio
            text_emb = self.text_proj(text_features[:, :T])
            x = x + text_emb

        cond = self.speaker_embed(speaker_id) + self.gesture_embed(gesture_type)
        anchor_cond = self._encode_anchor(anchor_frames, anchor_audio)
        if anchor_cond is not None:
            cond = cond + anchor_cond
        x = x + cond.unsqueeze(1)
        x = self.pos_encoder(x)

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        # Strided conv downsampling
        k = self.temporal_downsample
        T_prime = math.ceil(T / k)
        T_padded = T_prime * k

        x_conv = x.transpose(1, 2)
        if padding_mask is not None:
            x_conv = x_conv * (~padding_mask).unsqueeze(1).float()
        if T_padded > T:
            x_conv = F.pad(x_conv, (0, T_padded - T), value=0)

        x_down = self.downsample(x_conv).transpose(1, 2)
        x_down = self.downsample_post(x_down)
        return x_down  # (B, T', d_model)


class AdaLNDenoiserBlock(nn.Module):
    """Transformer block: self-attn on z + cross-attn to audio, AdaLN from timestep."""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN1 = nn.Linear(d_model, 2 * d_model)

        # Cross-attention to audio
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN3 = nn.Linear(d_model, 2 * d_model)

        self.dropout = nn.Dropout(dropout)

        # Zero-init AdaLN for identity at start
        for adaLN in [self.adaLN1, self.adaLN3]:
            nn.init.zeros_(adaLN.weight)
            nn.init.zeros_(adaLN.bias)
            with torch.no_grad():
                adaLN.bias[:d_model] = 1.0  # scale=1

    def forward(self, x, audio_ctx, t_cond, z_padding_mask=None):
        """
        x: (B, T', d_model) - noisy z tokens
        audio_ctx: (B, T', d_model) - audio context
        t_cond: (B, d_model) - timestep conditioning
        """
        # AdaLN self-attention
        scale1, shift1 = self.adaLN1(t_cond).chunk(2, dim=-1)
        h = self.norm1(x) * scale1.unsqueeze(1) + shift1.unsqueeze(1)
        h, _ = self.self_attn(h, h, h, key_padding_mask=z_padding_mask)
        x = x + self.dropout(h)

        # Cross-attention to audio (no AdaLN — audio provides its own structure)
        h = self.norm2(x)
        h, _ = self.cross_attn(h, audio_ctx, audio_ctx)
        x = x + self.dropout(h)

        # AdaLN FFN
        scale3, shift3 = self.adaLN3(t_cond).chunk(2, dim=-1)
        h = self.norm3(x) * scale3.unsqueeze(1) + shift3.unsqueeze(1)
        x = x + self.ffn(h)
        return x


class TemporalFlowMatchingPrior(nn.Module):
    """Flow matching for temporal latent sequences p(z_seq | audio, speaker, gesture).

    Denoiser is a transformer with self-attention on z tokens + cross-attention
    to downsampled audio features, conditioned on timestep via AdaLN.
    """

    def __init__(
        self,
        latent_dim=16,
        audio_dim=768,
        d_model=256,
        nhead=4,
        num_audio_layers=4,
        num_denoiser_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        num_speakers=31,
        num_gesture_types=2,
        max_len=1024,
        temporal_downsample=8,
        pose_dim=165,
        cond_drop_prob=0.1,
        text_dim=0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.temporal_downsample = temporal_downsample
        self.cond_drop_prob = cond_drop_prob

        # Audio encoder → temporal features (B, T', d_model)
        self.audio_encoder = TemporalAudioEncoder(
            audio_dim=audio_dim, d_model=d_model, nhead=nhead,
            num_layers=num_audio_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, num_speakers=num_speakers,
            num_gesture_types=num_gesture_types, max_len=max_len,
            temporal_downsample=temporal_downsample, pose_dim=pose_dim,
            text_dim=text_dim,
        )

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(d_model)

        # Null audio context for CFG (per-token)
        self.null_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # z input projection
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.z_pos_encoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        # Denoiser transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNDenoiserBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_denoiser_layers)
        ])

        # Output
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_adaLN = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)

        # Zero-init output
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.out_adaLN.weight)
        nn.init.zeros_(self.out_adaLN.bias)
        with torch.no_grad():
            self.out_adaLN.bias[:d_model] = 1.0

        # z normalization stats (per-dim, set from training data)
        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))

    def encode_audio(self, audio_features, speaker_id, gesture_type, padding_mask=None,
                     anchor_frames=None, anchor_audio=None, text_features=None):
        """Pre-compute audio context (call once, reuse across ODE steps)."""
        return self.audio_encoder(
            audio_features, speaker_id, gesture_type, padding_mask,
            anchor_frames=anchor_frames, anchor_audio=anchor_audio,
            text_features=text_features)

    def predict_velocity(self, z_t, t, audio_ctx, z_padding_mask=None):
        """Predict velocity field v(z_t, t, audio_ctx).

        Args:
            z_t: (B, T', latent_dim) - noisy temporal latent sequence
            t: (B,) - timestep in [0, 1]
            audio_ctx: (B, T', d_model) - pre-computed temporal audio context
            z_padding_mask: (B, T') - True for padded z tokens

        Returns:
            v: (B, T', latent_dim) - predicted velocity
        """
        t_cond = self.time_embed(t)  # (B, d_model)

        h = self.z_proj(z_t)  # (B, T', d_model)
        h = self.z_pos_encoder(h)

        for block in self.blocks:
            h = block(h, audio_ctx, t_cond, z_padding_mask=z_padding_mask)

        # Output with AdaLN
        scale, shift = self.out_adaLN(t_cond).chunk(2, dim=-1)
        h = self.out_norm(h) * scale.unsqueeze(1) + shift.unsqueeze(1)
        v = self.out_proj(h)  # (B, T', latent_dim)
        return v

    def compute_loss(self, z_1, audio_features, speaker_id, gesture_type,
                     padding_mask=None, z_padding_mask=None,
                     cond_drop_prob=None, anchor_frames=None, anchor_audio=None,
                     text_features=None):
        """Flow matching training loss on temporal z sequences.

        Args:
            z_1: (B, T', latent_dim) - target temporal latent sequence
            audio_features: (B, T, audio_dim)
            z_padding_mask: (B, T') - True for padded z tokens
            text_features: (B, T, text_dim) - optional pre-computed text embeddings
        """
        if cond_drop_prob is None:
            cond_drop_prob = self.cond_drop_prob
        B = z_1.shape[0]
        device = z_1.device

        # Normalize z (broadcast over time dim)
        z_1_norm = (z_1 - self.z_mean) / self.z_std

        # Mask padded z tokens
        if z_padding_mask is not None:
            z_1_norm = z_1_norm * (~z_padding_mask).unsqueeze(-1).float()

        # Sample noise and time
        z_0 = torch.randn_like(z_1_norm)
        if z_padding_mask is not None:
            z_0 = z_0 * (~z_padding_mask).unsqueeze(-1).float()

        t = torch.sigmoid(torch.randn(B, device=device) * 0.5).clamp(1e-5, 1 - 1e-5)

        # Interpolate: (B, T', latent_dim)
        t_expand = t[:, None, None]
        z_t = (1 - t_expand) * z_0 + t_expand * z_1_norm
        v_target = z_1_norm - z_0

        # Audio context (with optional text)
        audio_ctx = self.encode_audio(
            audio_features, speaker_id, gesture_type, padding_mask,
            anchor_frames=anchor_frames, anchor_audio=anchor_audio,
            text_features=text_features)

        # CFG dropout: replace audio context with null tokens
        if cond_drop_prob > 0 and self.training:
            drop_mask = torch.rand(B, device=device) < cond_drop_prob
            if drop_mask.any():
                null = self.null_token.expand(B, audio_ctx.shape[1], -1)
                audio_ctx = torch.where(
                    drop_mask[:, None, None], null, audio_ctx)

        # Predict velocity
        v_pred = self.predict_velocity(z_t, t, audio_ctx, z_padding_mask=z_padding_mask)

        # MSE loss (masked for padded tokens)
        if z_padding_mask is not None:
            valid_mask = (~z_padding_mask).unsqueeze(-1).float()
            loss = ((v_pred - v_target) ** 2 * valid_mask).sum() / valid_mask.sum() / self.latent_dim
        else:
            loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def sample(self, audio_features, speaker_id, gesture_type,
               padding_mask=None, num_steps=50, guidance_scale=1.0,
               anchor_frames=None, anchor_audio=None, z_seq_len=None,
               temperature=1.0, num_samples=1, truncation=1.0,
               refine_strength=0.0, use_euler=False, text_features=None):
        """Sample temporal z sequence by integrating the learned ODE.

        Args:
            z_seq_len: int, number of z tokens (T'). If None, inferred from audio length.
            num_samples: int, if > 1, sample multiple z and average (reduces variance).
            truncation: float, scale z toward mean after ODE (< 1 reduces diversity).
            refine_strength: float, if > 0, add noise at this t and re-denoise (SDEdit).
            use_euler: bool, if True use Euler method instead of midpoint.
            text_features: (B, T, text_dim) - optional pre-computed text embeddings

        Returns:
            z: (B, T', latent_dim) - sampled temporal latent sequence
        """
        B = audio_features.shape[0]
        T_audio = audio_features.shape[1]
        device = audio_features.device

        if z_seq_len is None:
            z_seq_len = math.ceil(T_audio / self.temporal_downsample)

        # Encode audio once (with optional text)
        audio_ctx = self.encode_audio(
            audio_features, speaker_id, gesture_type, padding_mask,
            anchor_frames=anchor_frames, anchor_audio=anchor_audio,
            text_features=text_features)

        # Pad/trim audio_ctx to match z_seq_len
        if audio_ctx.shape[1] != z_seq_len:
            if audio_ctx.shape[1] < z_seq_len:
                pad = audio_ctx[:, -1:, :].expand(B, z_seq_len - audio_ctx.shape[1], -1)
                audio_ctx = torch.cat([audio_ctx, pad], dim=1)
            else:
                audio_ctx = audio_ctx[:, :z_seq_len, :]

        # CFG: null context
        if guidance_scale != 1.0:
            null_ctx = self.null_token.expand(B, z_seq_len, -1)

        def _run_ode(audio_ctx_in, null_ctx_in):
            z = torch.randn(B, z_seq_len, self.latent_dim, device=device) * temperature
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B,), i * dt, device=device)
                if guidance_scale != 1.0:
                    v_cond = self.predict_velocity(z, t, audio_ctx_in)
                    v_uncond = self.predict_velocity(z, t, null_ctx_in)
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = self.predict_velocity(z, t, audio_ctx_in)
                if use_euler:
                    z = z + v * dt
                else:
                    # Midpoint method
                    z_mid = z + v * (dt / 2)
                    t_mid = torch.full((B,), (i + 0.5) * dt, device=device)
                    if guidance_scale != 1.0:
                        v_cond = self.predict_velocity(z_mid, t_mid, audio_ctx_in)
                        v_uncond = self.predict_velocity(z_mid, t_mid, null_ctx_in)
                        v_mid = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        v_mid = self.predict_velocity(z_mid, t_mid, audio_ctx_in)
                    z = z + v_mid * dt
            return z

        if num_samples > 1:
            z_sum = _run_ode(audio_ctx, null_ctx if guidance_scale != 1.0 else None)
            for _ in range(num_samples - 1):
                z_sum = z_sum + _run_ode(audio_ctx, null_ctx if guidance_scale != 1.0 else None)
            z = z_sum / num_samples
        else:
            z = _run_ode(audio_ctx, null_ctx if guidance_scale != 1.0 else None)

        # SDEdit refinement: add noise at t=refine_strength, re-denoise
        if refine_strength > 0:
            t_start = refine_strength
            noise = torch.randn_like(z)
            z_noisy = (1 - t_start) * noise + t_start * z
            # Run ODE from t_start to 1
            refine_steps = max(1, int(num_steps * (1 - t_start)))
            dt_r = (1.0 - t_start) / refine_steps
            for i in range(refine_steps):
                t_val = t_start + i * dt_r
                t = torch.full((B,), t_val, device=device)
                if guidance_scale != 1.0:
                    v_cond = self.predict_velocity(z_noisy, t, audio_ctx)
                    v_uncond = self.predict_velocity(z_noisy, t, null_ctx)
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = self.predict_velocity(z_noisy, t, audio_ctx)
                z_noisy = z_noisy + v * dt_r
            z = z_noisy

        # Truncation trick: scale z toward 0 in normalized space
        if truncation != 1.0:
            z = z * truncation

        # Denormalize
        z = z * self.z_std + self.z_mean
        return z
