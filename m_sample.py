import sys

sys.path.append('../')

import torch
from torchvision.utils import save_image
from m_util import get_runtime_sampler_path
from torchvision import utils
# from sample import sample_model


@torch.no_grad()
def vqvae_sampler(folder_name, model, data, dataset_name, run_num, epoch, batch_size):
    import os
    import numpy as np
    path = get_runtime_sampler_path(folder_name, dataset_name, run_num, epoch)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with torch.no_grad():
        out, _ = model(data)

    if 'beat2' in dataset_name.lower() or 'pose' in dataset_name.lower():
        # For pose data, save as numpy arrays instead of images
        save_dict = {
            'original': data.cpu().numpy(),
            'reconstructed': out.cpu().numpy()
        }

        # Check if this is a multi-level model and generate samples for each level
        if hasattr(model, 'n_level') or (hasattr(model, 'module') and hasattr(model.module, 'n_level')):
            # Get the actual model (unwrap DataParallel if needed)
            actual_model = model.module if hasattr(model, 'module') else model

            if hasattr(actual_model, 'decode_partial_levels'):
                n_levels = actual_model.n_level
                print(f"  Generating samples for {n_levels} levels...")

                # Generate reconstruction for each level (1, 2, 3, ...)
                for level in range(1, n_levels + 1):
                    # Transpose data to conv1d format
                    data_conv = data.transpose(1, 2)
                    level_out, _ = actual_model.decode_partial_levels(data_conv, num_levels=level)
                    save_dict[f'reconstructed_level_{level}'] = level_out.cpu().numpy()
                    print(f"    - Level {level} (using {level} quantization level(s))")

        np.savez(path + '.npz', **save_dict)
        print(f"  Saved to: {path}.npz")
    else:
        # For image data, save as images
        utils.save_image(
            torch.cat([data, out], 0),
            path + '.png',
            nrow=batch_size,
            normalize=True,
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
