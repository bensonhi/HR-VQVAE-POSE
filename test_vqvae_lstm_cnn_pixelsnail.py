import sys
sys.path.append('..')
from video.dataset import MnistVideoCodeLMDBDataset
from video.dataloader import video_mnist_dataloader
from torchvision import utils
import numpy as np
from video.LSTM3 import LSTM3
from video.train_vqvae_lstm_cnn_pixelsnail import train
from image.modified.m_vqvae import VQVAE_1
from torch import nn
import torch
from matplotlib import pyplot as plt
from image.modified.m_pixelsnail import PixelSNAIL
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

vqvae_ckpt_path = '../video/checkpoints/videomnist/vqvae/1/00099.pt'
input_channel = 16
device = 'cuda'
run_num = 3
vqvae_model = VQVAE_1(in_channel=1,
            channel=32,
            n_res_block=4,
            n_res_channel=16,
            embed_dim=16,
            n_embed=input_channel,
            decay=0.99, )
vqvae_model = nn.DataParallel(vqvae_model)
vqvae_model.load_state_dict(torch.load(vqvae_ckpt_path))
vqvae_model = vqvae_model.to(device)

def callback(sample ,frame , epoch, _iter):
    global vqvae_model, input_channel, device, run_num
    with torch.no_grad():
        vqvae_model.eval()

        sample = vqvae_model.module.decode_code(sample)
        sample = sample.cpu().detach()
        sample = sample.squeeze()
        sample = (sample > 0.5).float()
        sample = sample.unsqueeze(1)
        frame = vqvae_model.module.decode_code(frame)
        frame = frame.cpu().detach()
        frame = frame.squeeze()
        frame = frame.unsqueeze(1)
        frame = (frame > 0.5).float()
        path = 'vqvae_lstm_pixelsnail_run{}_epoch{}_iter{}.png'.format(*[run_num,epoch,_iter])
        merge = torch.cat([sample,frame], 0)
        utils.save_image(
            merge,
            path,
            nrow=len(frame),
            normalize=True,
            range=(-1, 1),
        )
        img = plt.imread(path)
        plt.imshow(img)
        plt.show()

def train_model(checkpoint):
    global vqvae_model, input_channel, device, run_num
    
    lambda_name = 'vqvae_videomnist_2_00099'
    hidden_channel = input_channel
    epoch_num = 100
    batch_size = 128
    lr = 0.0001
    
    image_samples = 10
    frame_len = 8

    dataset = MnistVideoCodeLMDBDataset(lambda_name, frame_len + 1)
    loader = video_mnist_dataloader(dataset, batch_size, shuffle=True)


    videomnist_path = '../video/datasets/mnist/moving_mnist/mnist_test_seq.npy'
    orginal_frames = np.load(videomnist_path)
    orginal_frames = orginal_frames.swapaxes(0, 1).astype(np.float32)
    orginal_frames[orginal_frames > 0] = 1.

    input_size = (16,16)
    input_channel = 16
    device = 'cuda'


    hidden_channel = 32
    cnn_channel = 32
    channel = 32
    cnn_kernel_size = 5
    kernel_size = 5
    n_block = 1
    n_res_block = 1
    n_res_channel = 32
    dropout = 0.1
    n_out_res_block = 1
    n_cond_res_block = 1
    cond_res_channel = 32
    lr = 0.0001
    epoch_num = 100


    torch.backends.cudnn.enabled = False
    lstm_model = LSTM3( input_channel= input_channel,hidden_channel= hidden_channel, device=device)
    cnn_model = nn.Conv2d(hidden_channel,
                cnn_channel,
                cnn_kernel_size,
                stride=1,
                padding=cnn_kernel_size // 2,)

    pixel_model = PixelSNAIL(
                shape = [input_size[0], input_size[1]],
                n_class = input_channel,
                cond_channel = cnn_channel,
                channel = channel,
                kernel_size = kernel_size,
                n_block = n_block,
                n_res_block = n_res_block,
                res_channel = n_res_channel,
                dropout=dropout,
                n_out_res_block=n_out_res_block,
                n_cond_res_block=n_cond_res_block,
                cond_res_channel=cond_res_channel,
            )
    train(checkpoint, lstm_model,cnn_model, pixel_model,input_channel, loader, callback, epoch_num, device, lr, run_num, image_samples=image_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=int, default=-1)
    args = parser.parse_args()
    print(args.ckpt)
    train_model(args.ckpt)
