import sys

sys.path.append('../')

import torch
import numpy as np
from torchvision.utils import save_image
from m_util import get_runtime_sampler_path
from torchvision import utils


@torch.no_grad()
def vae_sampler(folder_name, model, data, dataset_name, run_num, epoch, batch_size,
                padding_mask=None, gesture_type=None, lengths=None, audio_features=None):
    import os
    import numpy as np
    path = get_runtime_sampler_path(folder_name, dataset_name, run_num, epoch)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model.eval()
    with torch.no_grad():
        out, _, _ = model(data, padding_mask=padding_mask, gesture_type=gesture_type,
                         lengths=lengths, audio_features=audio_features)

    if 'beat2' in dataset_name.lower() or 'pose' in dataset_name.lower():
        # For pose data, save as numpy arrays instead of images
        save_dict = {
            'original': data.cpu().numpy(),
            'reconstructed': out.cpu().numpy()
        }

        # Save conditioning info
        if lengths is not None:
            save_dict['lengths'] = lengths.cpu().numpy()
        if gesture_type is not None:
            save_dict['gesture_type'] = gesture_type.cpu().numpy()

        # Check if this is a multi-level model and generate samples for each level
        if hasattr(model, 'n_level') or (hasattr(model, 'module') and hasattr(model.module, 'n_level')):
            actual_model = model.module if hasattr(model, 'module') else model

            if hasattr(actual_model, 'decode_partial_levels'):
                n_levels = actual_model.n_level
                print(f"  Generating samples for {n_levels} levels...")

                for level in range(1, n_levels + 1):
                    level_out, _ = actual_model.decode_partial_levels(
                        data, num_levels=level, padding_mask=padding_mask,
                        gesture_type=gesture_type, lengths=lengths,
                        audio_features=audio_features
                    )
                    save_dict[f'reconstructed_level_{level}'] = level_out.cpu().numpy()
                    print(f"    - Level {level} (using {level} VAE level(s))")

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
    model.train()


def runtime_vae_sampler(model, imgs, dataset_name, run_num, epoch, batch_size):
    model.eval()
    vae_sampler(model, imgs, dataset_name, run_num, epoch, batch_size)
    model.train()
