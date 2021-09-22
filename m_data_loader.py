from torch.utils.data import DataLoader
from m_dataset import LMDBDataset, LMDB_VQVAE_PIXEL_Dataset
from torchvision import datasets


def get_lmdb_pixel_loader(dataset_path, n_run, batch_size, x_name, cond_name, shuffle=True, num_workers=4,
                          drop_last=True):
    lampda_name = '{}_{}'.format(*[dataset_path, n_run])
    dataset = LMDB_VQVAE_PIXEL_Dataset(lampda_name, x_name, cond_name)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
    )
    return loader


def get_lmdb_loader(dataset_path, n_run, batch_size, shuffle=True, num_workers=4, drop_last=True):
    lampda_name = '{}_{}'.format(*[dataset_path, n_run])
    dataset = LMDBDataset(lampda_name)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
    )
    return loader


def get_image_loader(dataset_path, batch_size, transform, shuffle=True, num_workers=4):
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

