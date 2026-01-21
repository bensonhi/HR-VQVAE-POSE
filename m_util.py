import sys
sys.path.append('../')
import torch
import os
import glob
from m_vae_pose import VAE_Pose_1, VAE_Pose_ML
from m_conf_parser import model_option_parser, training_params_parser


def get_sample_dir(dataset, n_run):
    return 'checkpoint/{}/{}/sample'.format(*[dataset, n_run])


def load_part(model, checkpoint, device):
    ckpt = torch.load(checkpoint)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    return model


def create_model_object(model_type, options):
    if model_type == 'vae':
        return VAE_Pose_ML(
            in_channel=options['in_channel'],
            channel=options['channel'],
            n_res_block=options['n_res_block'],
            n_res_channel=options['n_res_channel'],
            embed_dim=options['embed_dim'],
            n_level=options['n_level'],
            decay=options['decay'],
            stride=options['stride'],
            use_smplx=options.get('use_smplx', False),
            smplx_model_path=options.get('smplx_model_path', 'models_smplx_v1_1/models')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_type(folder_name):
    if folder_name == 'vae':
        return 'vae'
    else:
        return 'vae'  # Default to vae


def get_path(dataset_name, run_num, folder_name, file_type, checkpoint=0):
    checkpoint = '{}'.format(str(checkpoint).zfill(3))
    model_type = get_model_type(folder_name)

    file_path = 'checkpoint/{}/{}/{}/'.format(*[dataset_name, run_num, model_type])

    if file_type == 'conf':
        file_path += 'conf.ini'
    else:
        file_path += '{}.pt'.format(*[checkpoint])
    return file_path


def get_runtime_sampler_path(folder_name, dataset_name, run_num, epoch):
    model_type = get_model_type(folder_name)
    file_path = 'checkpoint/{}/{}/{}/'.format(*[dataset_name, run_num, model_type])
    file_path += 'runtime_samples/{}'.format(*[str(epoch + 1).zfill(5)])
    return file_path


def find_latest_checkpoint(dataset_name='beat2_poses', run_num=0, folder_name='vae'):
    """
    Find the latest checkpoint file for a given dataset/run/folder.

    Args:
        dataset_name: Dataset name (default: 'beat2_poses')
        run_num: Run number (default: 0)
        folder_name: Folder name (default: 'vae')

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    model_type = get_model_type(folder_name)
    checkpoint_dir = 'checkpoint/{}/{}/{}/'.format(dataset_name, run_num, model_type)

    # Look for .pt files
    pattern = os.path.join(checkpoint_dir, '*.pt')
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None

    # Sort by modification time (most recent first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)

    return checkpoints[0]


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load a checkpoint and remove DataParallel 'module.' prefix if present.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        state_dict: Clean state dict without 'module.' prefix
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    if state_dict and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    return state_dict


def conf_parser(dataset, n_run, folder_name):
    conf_path = get_path(dataset, n_run, folder_name, 'conf')
    model_type = get_model_type(folder_name)
    options = model_option_parser(model_type, conf_path)
    train_params = training_params_parser(conf_path)
    return options, train_params


def model_object_parser(dataset, n_run, folder_name):
    model_type = get_model_type(folder_name)
    options, _ = conf_parser(dataset, n_run, folder_name)
    return create_model_object(model_type, options)


def load_vae_model(device, dataset, n_run, vae_epoch):
    """Load a trained VAE model."""
    vae_checkpoint_path = get_path(dataset, n_run, 'vae', 'ckpt', checkpoint=vae_epoch)
    vae_obj = model_object_parser(dataset, n_run, 'vae')
    model_vae = load_part(vae_obj, vae_checkpoint_path, device)
    sample_dir = get_sample_dir(dataset, n_run)
    return model_vae, sample_dir
