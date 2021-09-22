from m_sample import make_sample
from m_util import load_model

dataset = 'fashion_mnist'
n_run = 0
vqvae_epoch = 378
top_epoch = 89
bottom_epoch = 19
file_name = '0.png'

vqvae, top, bottom, middle, sample_dir = load_model(device='cuda', dataset=dataset, n_run=n_run,
                                                    vqvae_epoch=vqvae_epoch, top_epoch=top_epoch,
                                                    bottom_epoch=bottom_epoch, middle_epoch=-1)

make_sample(vqvae, top, middle, bottom, '{}/sample/{}'.format(*[sample_dir, file_name]), temp=1.0)
