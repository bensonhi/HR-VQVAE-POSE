import sys
import torch
from torch import nn
from torch.nn import functional as F
# sys.path.append('../')
#
# from image.vqvae import Quantize


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, device = 'cuda'):
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
        # print('flatten input {}'.format(flatten.shape))
        # print('quantize embed {}'.format(self.embed))
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        # print('quantize dist {}'.format(dist.shape))
        _, embed_ind = (-dist).max(1)
        # print('quantize embed_ind {}'.format(embed_ind.shape))
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
        # print('quantize quantize {}'.format(quantize.shape))
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                    nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1),
                ]
            )

        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE_ML(nn.Module):
    def __init__(
            self,
            in_channel=1,
            channel=32,
            n_res_block=2,
            n_res_channel=16,
            embed_dim=16,
            n_level=4,
            n_embed = 16,
            decay=0.80,
            stride=4,
    ):
        super().__init__()
        self.device = 'cuda'
        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=stride)
        # self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.n_level = n_level
        self.quantizes = nn.ModuleList()
        self.quantizes_conv = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_level):
            self.quantizes.append(Quantize(embed_dim, n_embed))
            self.quantizes_conv.append(nn.Conv2d(embed_dim, embed_dim, 1))
            self.bns.append(nn.BatchNorm2d(embed_dim))
        # self.quantizes = Quantize(embed_dim, n_embed,decay=decay)
        # self.dec_t = Decoder(
        #     embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        # )
        # self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        # self.quantize_b = Quantize(embed_dim, n_embed)
        # self.upsample_t = nn.ConvTranspose2d(
        #     embed_dim, embed_dim, 4, stride=2, padding=1
        # )
        self.dec = Decoder(
            embed_dim ,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=stride,
        )

    def forward(self, input):
        quant, diff, _,_ = self.encode(input)
        # print('quant shape {}'.format(quant.shape))
        dec = self.decode(quant)

        return dec, diff

    def encode(self, input):
        enc = self.enc(input)

        bottleneck = self.quantize_conv(enc)
        ids = None
        quants = None
        diffs = None
        quant_sum = None
        for i,quantize in enumerate(self.quantizes):
            # print('bottleneck shape'.format(bottleneck.shape))
            quant, diff, id = quantize(bottleneck.permute(0, 2, 3, 1))
            # print(bottleneck.shape)
            # print(quant.shape)
            quant = quant.permute(0, 3, 1, 2)
            diff = diff.unsqueeze(0)

            if diffs is None:
                diffs = diff
                quant_sum = quant
                quants = quant.unsqueeze(1)
                ids = id.unsqueeze(1)
            else:
                diffs += diff
                quant_sum += quant
                quants = torch.cat((quants,quant.unsqueeze(1)),dim=1)
                ids = torch.cat((ids, id.unsqueeze(1)), dim=1)
            bottleneck -= quant
            # bottleneck = F.relu(self.bns[i](self.quantizes_conv[i](bottleneck)))
        return quant_sum, diffs, quants, ids

    def decode(self, quant):
        dec = self.dec(quant)
        return dec

    def decode_code(self, codes):
        quants = None
        for i, code in enumerate(codes):
            quant = self.quantizes.embed_code(code)
            quant = quant.permute(0, 3, 1, 2)
            quants += quant
        dec = self.decode(quants)

        return dec
