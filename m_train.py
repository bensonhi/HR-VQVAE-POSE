from m_trainer import train
from m_data_loader import get_lmdb_pixel_loader, get_image_loader
from m_preprocessing import get_mnist_transform, get_imagenet_transform
from m_sample import vqvae_sampler

dataset_name = 'fashion_mnist'
folder_name = 'vqvae_1'

n_run = 0
start_epoch = -1
end_epoch = -1
batch_size = -1
lr = -1
sched=None
device='cuda'
size=28
amp=None
# amp='O0'

sample_period = 1
sampler = vqvae_sampler

if folder_name == 'top':
    loader = get_lmdb_pixel_loader(dataset_name, n_run, batch_size,
                                   x_name='top', cond=None, shuffle=True, num_workers=4)
elif folder_name == 'bottom':
    loader = get_lmdb_pixel_loader(dataset_name, n_run, batch_size,
                                   x_name='bottom', cond='top', shuffle=True, num_workers=4)
elif folder_name == 'vqvae_1':
    loader = get_image_loader(dataset_name, batch_size, transform=get_mnist_transform(), shuffle=True, num_workers=2)

elif folder_name == 'vqvae':
    loader = get_image_loader(dataset_name, batch_size, transform= get_imagenet_transform(size), shuffle=True, num_workers=4)

train(
       folder_name,
       loader,
       dataset_name,
       n_run,
       sample_period,
       sampler,
       start_epoch=start_epoch,
       end_epoch=end_epoch,
       batch_size=batch_size,
       sched=sched,
       device=device,
       size=size,
       lr=lr,
       amp=amp)

