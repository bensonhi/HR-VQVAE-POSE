import sys
sys.path.append('../video')
sys.path.append('../image/modified')
from m_vqvae import VQVAE_1
from torchvision import utils
from dataset import MnistVideoDataset, MnistVideoCodeLMDBDataset
from dataloader import video_mnist_dataloader
import torch
from extract_videomnist import extract
from matplotlib import pyplot as plt
import numpy as np
from torch import nn





batch_size = 100
device = 'cuda'
lamda_name = 'vqvae_videomnist_1_00099'
ckpt_path = '../video/checkpoints/videomnist/vqvae/1/00099.pt'





model = VQVAE_1(in_channel=1,
            channel=32,
            n_res_block=4,
            n_res_channel=16,
            embed_dim=16,
            n_embed=16,
            decay=0.99, )
model = nn.DataParallel(model)
model.load_state_dict(torch.load(ckpt_path))
model = model.to(device)


extract(model, lamda_name, device, video_batch=10)
plt.imshow(frame.cpu().detach().numpy()[0,0,:,:])



sample_size = 1
dataset_video = MnistVideoDataset(path='../video/datasets/mnist/moving_mnist/mnist_test_seq.npy', frame_len=20)
loader = video_mnist_dataloader(dataset_video, 1, shuffle=False, num_workers=4, drop_last=True)

subset = torch.utils.data.Subset(dataset_video, list(range(sample_size)))
testloader_subset  = torch.utils.data.DataLoader(subset, batch_size=sample_size, num_workers=4, shuffle=False)


with torch.no_grad():
    for frames,vind,find in testloader_subset:
        model.eval()
        frames= frames.squeeze(0)
        frames = frames.unsqueeze(1)
        frames = frames.to(device)

        outs,_ = model(frames)
        outs = (outs > 0.5).float()
        utils.save_image(
                    torch.cat([frames, outs], 0),
                    'sample_encode.png',
                    nrow=len(outs),
                    normalize=True,
                    range=(-1, 1),
                )

img = plt.imread('sample_encode.png')
plt.imshow(img)
plt.show()



sample_size = 20
dataset_lmdb = MnistVideoCodeLMDBDataset(lamda_name,sample_size)
subset = torch.utils.data.Subset(dataset_lmdb, [0])
testloader_subset  = torch.utils.data.DataLoader(subset, batch_size=sample_size, num_workers=0, shuffle=False)

torch.backends.cudnn.enabled = False

with torch.no_grad():
    for data in testloader_subset:
        _ids, vis, fis = data
        model.eval()
        _ids = _ids.to(device)
        _ids = _ids.squeeze()
        outs = model.module.decode_code(_ids)
        outs = (outs > 0.5).float()
        frames = (frames > 0.5).float()
        merge = torch.cat([frames, outs], 0)        
        utils.save_image(
                        merge,
                    'sample_extract.png',
                    nrow=len(frames),
                    normalize=True,
                    range=(-1, 1),

                )

img = plt.imread('sample_extract.png')
plt.imshow(img)
plt.show()
