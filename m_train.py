from m_trainer import train
from m_data_loader import get_lmdb_pixel_loader, get_image_loader
from m_preprocessing import get_mnist_transform, get_imagenet_transform, get_ffhq_transform
from m_sample import vqvae_sampler
from m_util import conf_parser
import os

# FFHQ Dataset Configuration
dataset_name = 'ffhq'
folder_name = 'vqvae'

n_run = 0
start_epoch = -1
end_epoch = -1
batch_size = -1
lr = -1
sched=None
device='cuda'
size=256  # FFHQ image size
amp=None

sample_period = 1
sampler = vqvae_sampler

# FFHQ dataset path - UPDATE THIS PATH to your FFHQ dataset location
ffhq_dataset_path = 'data/ffhq'  # Change this to your actual FFHQ dataset path

if folder_name == 'top':
    loader = get_lmdb_pixel_loader(dataset_name, n_run, batch_size,
                                   x_name='top', cond=None, shuffle=True, num_workers=4)
elif folder_name == 'bottom':
    loader = get_lmdb_pixel_loader(dataset_name, n_run, batch_size,
                                   x_name='bottom', cond='top', shuffle=True, num_workers=4)
elif folder_name == 'vqvae_1':
    from torchvision import datasets
    from torch.utils.data import DataLoader
    dataset = datasets.FashionMNIST('data', train=True, transform=get_mnist_transform(), download=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

elif folder_name == 'vqvae':
    # Load configuration parameters first
    model_options, train_params = conf_parser(dataset_name, n_run, folder_name)
    
    # Create FFHQ data loader using config batch size
    transform = get_ffhq_transform(size=size)
    loader = get_image_loader(
        dataset_path=ffhq_dataset_path,
        batch_size=train_params['batch'],  # Use batch=32 from conf.ini
        transform=transform,
        shuffle=True,
        num_workers=8   # Optimized for 13-core machine
    )
    

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

