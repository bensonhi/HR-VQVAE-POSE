import sys
sys.path.append('../video')
sys.path.append('../image/modified')
from m_vqvae import VQVAE_1
from torchvision import utils
from dataset import VideoMnistDataset, VideoMnistLMDBDataset
from dataloader import video_mnist_dataloader
import torch
from extract_videomnist import extract
from matplotlib import pyplot as plt
import numpy as np
from torch import nn


batch_size = 100
device = 'cuda'
dataset_video = VideoMnistDataset('../video/datasets/mnist/moving_mnist/mnist_test_seq.npy', 1, 0, 20000)
loader = video_mnist_dataloader(dataset_video, batch_size, shuffle=True, num_workers=4, drop_last=True)
lamda_name = 'vqvae_videomnist_1_00099'
ckpt_path = '../video/checkpoints/videomnist/vqvae/1/00002.pt'

model = VQVAE_1(in_channel=1,
            channel=32,
            n_res_block=2,
            n_res_channel=16,
            embed_dim=8,
            n_embed=4,
            decay=0.99, )

ckpt = torch.load(ckpt_path)
if 'model' in ckpt:
    ckpt = ckpt['model']
    print('model')

if ckpt is None:
    print('none') 
model = nn.DataParallel(model)

model.load_state_dict(ckpt)
model = model.to(device)
sample_size  = 10
subset = torch.utils.data.Subset(dataset_video, list(range(sample_size)))

testloader_subset  = torch.utils.data.DataLoader(subset, batch_size=sample_size, num_workers=4, shuffle=False)


with torch.no_grad():
    for frame in testloader_subset:
        frame = frame[0]
        model.eval()
        frame = frame.to(device)

        out,_ = model(frame)
        out = (out > 0.5).float()
        utils.save_image(
                    torch.cat([frame, out], 0),
                    'sample_extract.png',
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
