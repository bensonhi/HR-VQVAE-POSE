import sys

sys.path.append('../')

import torch
from torchvision.utils import save_image
from m_util import get_runtime_sampler_path
from torchvision import utils
from sample import sample_model


@torch.no_grad()
def vqvae_sampler(folder_name, model, imgs, dataset_name, run_num, epoch, batch_size):
    path = get_runtime_sampler_path(folder_name, dataset_name, run_num, epoch)

    with torch.no_grad():
        out, _ = model(imgs)

    utils.save_image(
        torch.cat([imgs, out], 0),
        path,
        nrow=batch_size,
        normalize=True,
        range=(-1, 1),
    )


def runtime_vqvae_sampler(model, imgs, dataset_name, run_num, epoch, batch_size):
    model.eval()
    vqvae_sampler(model, imgs, dataset_name, run_num, epoch, batch_size)
    model.train()


def runtime_pixelsnail_sampler(folder_name, model,
                               dataset_name, run_num, epoch, batch_size=16, condition=None, image_size=[32, 32],
                               device='cuda', temperature=1.0):
    model.eval()
    row = sample_model(model, image_size=image_size, condition=condition,
                             batch_size=batch_size, device=device, temperature=temperature)
    path = get_runtime_sampler_path(folder_name, dataset_name, run_num, epoch)
    utils.save_image(
        torch.cat(row, 0),
        path,
        nrow=batch_size,
        normalize=True,
        range=(-1, 1),
    )


def make_sample(model_vqvae, model_top, model_middle, model_bottom, file_path, batch=16, device='cuda', temp=1.0):
    top_sample = sample_model(model_top, device, batch, [32, 32], temp)

    if model_middle is not None:
        middle_sample = sample_model(
            model_middle, device, batch, [64, 64], temp, condition=top_sample
        )
        bottom_sample = sample_model(
            model_bottom, device, batch, [128, 128], temp, condition=middle_sample
        )
    else:
        bottom_sample = sample_model(
            model_bottom, device, batch, [64, 64], temp, condition=top_sample
        )

    if model_middle is not None:
        decoded_sample = model_vqvae.decode_code(top_sample, middle_sample, bottom_sample)
    else:
        decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)

    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, file_path,
               normalize=True, range=(-1, 1))
