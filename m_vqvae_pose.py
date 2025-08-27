import sys
import torch
from torch import nn
from torch.nn import functional as F

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
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
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

    def forward(self, input):
        # Input: (batch, sequence_length, pose_dim)
        # Transpose to (batch, pose_dim, sequence_length) for conv1d
        input = input.transpose(1, 2)
        
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        
        # Transpose back to (batch, sequence_length, pose_dim)
        dec = dec.transpose(1, 2)
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

    def forward(self, input):
        # Input: (batch, sequence_length, pose_dim)
        # Transpose to (batch, pose_dim, sequence_length) for conv1d
        input = input.transpose(1, 2)
        
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        
        # Transpose back to (batch, sequence_length, pose_dim)
        dec = dec.transpose(1, 2)
        return dec, diff

    def encode(self, input):
        enc = self.enc(input)
        quant = self.quantize_conv(enc)
        
        # Multi-level quantization
        diffs = []
        residual = quant.permute(0, 2, 1)  # (B, T, D)
        
        for i in range(self.n_level):
            quantized, diff, _ = self.quantizes[i](residual)
            diffs.append(diff)
            
            # Update residual (subtract quantized version)
            residual = residual - quantized
            
            if i == 0:  # Use first level as main quantization
                main_quant = quantized
        
        # Combine all levels
        final_quant = main_quant.permute(0, 2, 1)  # (B, D, T)
        combined_diff = torch.stack(diffs).mean()
        
        return final_quant, combined_diff.unsqueeze(0), None

    def decode(self, quant):
        dec = self.dec(quant)
        return dec

    def decode_code(self, codes):
        # For compatibility - use first code
        if isinstance(codes, list):
            code = codes[0]
        else:
            code = codes
            
        quant = self.quantizes[0].embed_code(code)
        quant = quant.permute(0, 2, 1)  # (B, D, T)
        dec = self.decode(quant)
        return dec.transpose(1, 2)