import argparse
import sys
import numpy as np
sys.path.append('../../image/modified')
sys.path.append('../../video')

from dataset import lmdb_ffhq
from dataloader import video_mnist_dataloader
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from m_vqvae_multi_level8 import VQVAE_ML
from torch import optim, nn
import matplotlib.pyplot as plt


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)

run_num = "test_4*16"
batch_size = 12
lr = 0.0001
device = 'cuda'
epoch_num = 50
image_samples = 4
channel = 512
n_res_block = 12
n_res_channel =512
embed_dim=256
n_level = 4
n_embed= 16

def train(epoch_num, loader, model, optimizer, scheduler,run_num ,image_samples, device):
    loader = tqdm(loader)

    latent_loss_weight = 0.10
    mse_sum = 0
    mse_n = 0
    
    for iter_, img in enumerate(loader):
        model.zero_grad()
        img = img.to(device)
        n_outs, latent_loss = model(img)
        recon_loss_n = np.empty_like(n_outs, dtype=np.float64)
        recon_loss_n[0] = np.power((n_outs[0] - img) / (np.abs(n_outs[0] + img)),2)
        recon_loss_0 = recon_loss_n[0].sum() / (img.size)
        recon_loss_0.backward()
        for j in range(np.shape(n_outs)[0]-1):
            recon_loss_n[j+1] = np.power((np.sqrt(recon_loss_n[j])-n_outs[j+1]) / (np.linalg.norm(np.sqrt(recon_loss_n[j])) + n_outs[j+1])), 2)
            recon_loss = recon_loss_n[j+1]
            exec(f'recon_loss_{j+1} = np.sqrt(recon_loss.sum()) / (img.size))
            locals()['recon_loss_%s' % str(j+1)].backward()
            
        recon_loss = recon_loss.sum() / (img.size)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        if iter_ % 20 is 0:
            loader.set_description(
                (
                    f'epoch_num: {epoch_num}; mse: {recon_loss.item():.5f}; '
                    f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                    f'lr: {lr:.5f}'
                )
            )

        if iter_ % 200 == 0:
            #print('eval')
            model.eval()
            #print('sample')
            sample = img[:image_samples]
            #print('no grad')
            with torch.no_grad():
                #print('predict')
                out, _ = model(sample)
                #print('save')
                utils.save_image(
                    torch.cat([sample, out], 0),
                    '{}/vqvae_dimensional_{}_{}.png'.format(*[run_num, epoch,iter_]),
                    nrow=len(sample),
                    normalize=True,
                    range=(-1, 1),
                )
            #print('train')
            #print(mse_sum / mse_n)
            #print(recon_loss.item())
            model.train()

path = "path to the lmdb folder"
lmdb_path = path + 'name of the lmdb file'
dataset = lmdb_ffhq(lmdb_path)
in_channel = 3
loader = video_mnist_dataloader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)

model = VQVAE_ML(
            in_channel=in_channel,
            channel=channel,
            n_res_block=n_res_block,
            n_res_channel=n_res_channel,
            embed_dim=embed_dim,
            n_level=n_level,
            n_embed=n_embed,
            decay=0.99).to(device)

model = model.to(device)

optimizer = get_optimizer(model, lr)

for epoch in range(epoch_num):
    train(epoch, loader, model, optimizer, scheduler = None, run_num=run_num, image_samples=image_samples, device=device)
