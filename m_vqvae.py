import sys
import torch
from torch import nn

sys.path.append('../')

from image.vqvae import Quantize, Decoder, Encoder


class VQVAE_1(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=32,
            n_res_block=2,
            n_res_channel=16,
            embed_dim=16,
            n_embed=32,
            decay=0.99,
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.quantize = Quantize(embed_dim, n_embed)
        self.dec = Decoder(
            embed_dim ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant )

        return dec, diff

    def encode(self, input):
        enc = self.enc(input)

        quant = self.quantize_conv(enc).permute(0, 2, 3, 1)
        quant, diff, id = self.quantize(quant)
        quant = quant.permute(0, 3, 1, 2)
        diff = diff.unsqueeze(0)

        return quant,diff, id

    def decode(self, quant):
        dec = self.dec(quant)
        return dec

    def decode_code(self, code):
        quant = self.quantize.embed_code(code)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec
