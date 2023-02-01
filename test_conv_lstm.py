import sys
sys.path.append('..')
from video.dataset import MnistVideoCodeLMDBDataset
from video.dataloader import video_mnist_dataloader
from torchvision import utils
import numpy as np
from video.LSTM import LSTM
from video.ConvLstmCell import ConvLstmCell
from video.train_vqvae_lstm import train
from image.modified.m_vqvae import VQVAE_1
from torch import nn
import torch

lambda_name = 'lamda name'
vqvae_ckpt_path = 'path to the chekpoint'

input_channel = 8
epoch_num = 10
batch_size = 64
device = 'cuda'
lr = 0.0001
run_num = 1
image_samples = 10

dataset = MnistVideoCodeLMDBDataset(lambda_name, 2)
loader = video_mnist_dataloader(dataset, batch_size, shuffle=True)
model = ConvLstmCell( input_channel= 8,hidden_channel= 32,kernel_size= 3)

vqvae_model = VQVAE_1(in_channel=1,
            channel=32,
            n_res_block=2,
            n_res_channel=16,
            embed_dim=16,
            n_embed=16,
            decay=0.99, )
vqvae_model = nn.DataParallel(vqvae_model)
vqvae_model.load_state_dict(torch.load(vqvae_ckpt_path))
vqvae_model = vqvae_model.to(device)

videomnist_path = 'path to the test set'
orginal_frames = np.load(videomnist_path)
orginal_frames = orginal_frames.swapaxes(0, 1).astype(np.float32)
orginal_frames[orginal_frames > 0] = 1.

def callback(sample):
    o_frames = orginal_frames[video_inds[0], :image_samples, :, :]
    with torch.no_grad():
        sample = vqvae_model.module.decode_code(sample)
        utils.save_image(
            torch.cat([sample, o_frames], 0),
            dir + 'samples/videomnist/vqvae/{}/{}.png'.format(*[run_num, epoch]),
            nrow=image_samples,
            normalize=True,
            range=(-1, 1),
        )

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)

def train(model, input_channel, loader, callback, epoch_num, device, lr, run_num, ):
    image_samples = 1
    writer_path = 'vqvae_videomnist_1_00099_lstm'
    optimizer = get_optimizer(model, lr)
    model = model.to(device)
    model = nn.DataParallel(model)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir='logs/{}_{}'.format(*[writer_path, run_num]))

    for epoch in range(epoch_num):
        loader = tqdm(loader)
        mse_sum = 0
        mse_n = 0
        for iter, (frames, video_inds, frame_inds) in enumerate(loader):
            model.zero_grad()
            for i in range(frames.shape[1] - 1):
                input_ = _to_one_hot(frames[:, i, :, :], input_channel)
                output = _to_one_hot(frames[:, i + 1, :, :], input_channel)
                input_ = input_.to(device)
                output = output.to(device)
                pred = model(input_)
                loss = criterion(pred, output)
                loss.backward()
                optimizer.step()
                mse_sum += loss.item() * input_.shape[0]
                mse_n += input_.shape[0]
            lr = optimizer.param_groups[0]['lr']
            if iter % 200 is 0:
                loader.set_description(
                    (
                        'iter: {iter + 1}; mse: {loss.item():.5f}; '
                        f'avg mse: {mse_sum / mse_n:.5f}; '
                        f'lr: {lr:.5f}'
                    )
                )
            if iter is 0 and epoch > 0:
                writer.add_scalar('Loss/train', mse_sum / mse_n, epoch_num)
                vqvae_model.eval()
                sample = pred[0, :image_samples, :, :]
                callback(sample)
                vqvae_model.train()

            torch.save(model.state_dict(),
                       dir + 'checkpoints/videomnist/vqvae-lstm/{}/{}.pt'.format(*[run_num, str(epoch).zfill(5)]))

model = ConvLstmCell(input_channel, 32, 3)
train(model,4, loader, callback, epoch_num, device, lr, run_num, )
