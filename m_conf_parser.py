import configparser
import sys

sys.path.append('../')


def model_option_parser(model_type, conf_path):
    config = configparser.ConfigParser()
    config.read(conf_path)
    model = config['Model']
    if model_type == 'vae':
        options = {
            'in_channel': model.getint('in_channel'),
            'channel': model.getint('channel'),
            'n_res_block': model.getint('n_res_block'),
            'n_res_channel': model.getint('n_res_channel'),
            'embed_dim': model.getint('embed_dim'),
            'n_level': model.getint('n_level'),
            'decay': model.getfloat('decay'),
            'stride': model.getint('stride'),
        }

        # Check for SMPLX parameters
        if 'use_smplx' in model:
            options['use_smplx'] = model.getboolean('use_smplx')
            if 'smplx_model_path' in model:
                options['smplx_model_path'] = model.get('smplx_model_path')

        return options
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def training_params_parser(conf_path):
    config = configparser.ConfigParser()
    config.read(conf_path)
    train_conf = config['Train']
    return {
        'batch': train_conf.getint('batch'),
        'epoch': train_conf.getint('epoch'),
        'lr': train_conf.getfloat('lr'),
        'amp': train_conf.get('amp'),
    }
