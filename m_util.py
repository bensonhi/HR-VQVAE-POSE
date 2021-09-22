import sys
sys.path.append('../')
import torch
from m_vqvae import VQVAE_1
from image.vqvae import VQVAE
from image.pixelsnail import PixelSNAIL
from m_conf_parser import model_option_parser, training_params_parser


def get_sample_dir(dataset, n_run):
    return '../checkpoint/{}/{}/sample'.format(*[dataset, n_run])


def load_part(model, checkpoint, device):
    ckpt = torch.load(checkpoint)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    return model


def create_model_object(model_type, options):
    if model_type == 'pixelsnail':
        return PixelSNAIL(
            shape=options['shape'],
            n_class=options['n_class'],
            channel=options['channel'],
            kernel_size=options['kernel_size'],
            n_block=options['n_block'],
            n_res_block=options['n_res_block'],
            res_channel=options['res_channel'],
            dropout=options['dropout'],
            n_cond_res_block=options['n_cond_res_block'],
            cond_res_channel=options['cond_res_channel'],
            cond_res_kernel=options['cond_res_kernel'],
            n_out_res_block=options['n_out_res_block'],
            attention=options['attention']
        )
    elif model_type == 'vqvae_1':
        return VQVAE_1(
            in_channel=options['in_channel'],
            channel=options['channel'],
            n_res_block=options['n_res_block'],
            n_res_channel=options['n_res_channel'],
            embed_dim=options['embed_dim'],
            n_embed=options['n_embed'],
            decay=options['decay']
        )
    elif model_type == 'vqvae':
        return VQVAE(
            in_channel=options['in_channel'],
            channel=options['channel'],
            n_res_block=options['n_res_block'],
            n_res_channel=options['n_res_channel'],
            embed_dim=options['embed_dim'],
            n_embed=options['n_embed'],
            decay=options['decay']
        )


def get_model_type(folder_name):
    if folder_name in ['top', 'bottom', 'middle']:
        return 'pixelsnail'
    elif folder_name == 'vqvae_1':
        return 'vqvae_1'
    elif folder_name == 'vqvae':
        return 'vqvae'


def get_path(dataset_name, run_num, folder_name, file_type, checkpoint=0):
    checkpoint = '{}'.format(str(checkpoint).zfill(3))
    model_type = get_model_type(folder_name)

    file_path = '../checkpoint/{}/{}/{}/'.format(*[dataset_name, run_num, model_type])
    if model_type == 'pixelsnail':
        file_path += '{}/'.format(folder_name)

    if file_type == 'conf':
        file_path += 'conf.ini'
    else:
        file_path += '{}.pt'.format(*[checkpoint])
    return file_path


def get_runtime_sampler_path(folder_name, dataset_name, run_num, epoch):
    model_type = get_model_type(folder_name)
    file_path = '../checkpoint/{}/{}/{}/'.format(*[dataset_name, run_num, model_type])
    if model_type == 'pixelsnail':
        file_path += '{}/'.format(folder_name)
    file_path += 'runtime_samples/{}'.format(*[str(epoch + 1).zfill(5)])
    return file_path


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


def load_model(device, dataset, n_run, vqvae_epoch, top_epoch, bottom_epoch, middle_epoch=-1):
    top_checkpoint_path = get_path(dataset, n_run, 'top', 'ckpt', checkpoint=top_epoch)
    middle_checkpoint_path = get_path(dataset, n_run, 'middle', 'ckpt', checkpoint=middle_epoch)
    bottom_checkpoint_path = get_path(dataset, n_run, 'bottom', 'ckpt', checkpoint=bottom_epoch)
    vqvae_checkpoint_path = get_path(dataset, n_run, 'vqvae', 'ckpt', checkpoint=vqvae_epoch)

    vqvae_obj = model_object_parser(dataset, n_run, 'vqvae')
    top_obj = model_object_parser(dataset, n_run, 'top')
    bottom_obj = model_object_parser(dataset, n_run, 'bottom')
    if middle_epoch > 0:
        middle_obj = model_object_parser(dataset, n_run, 'middle')

    model_vqvae = load_part(vqvae_obj, vqvae_checkpoint_path, device)
    model_top = load_part(top_obj, top_checkpoint_path, device)
    model_bottom = load_part(bottom_obj, bottom_checkpoint_path, device)
    model_middle = None
    if middle_epoch > 0:
        model_middle = load_part(middle_obj, middle_checkpoint_path, device)

    sample_dir = get_sample_dir(dataset, n_run)
    return model_vqvae, model_top, model_bottom, model_middle, sample_dir
