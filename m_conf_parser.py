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
            'embed_dim': model.getint('embed_dim'),
            'n_level': model.getint('n_level'),
        }

        # Transformer parameters (new)
        if 'd_model' in model:
            options['d_model'] = model.getint('d_model')
        elif 'channel' in model:
            # Backward compatibility: use channel as d_model
            options['d_model'] = model.getint('channel')
        else:
            options['d_model'] = 256  # default

        if 'nhead' in model:
            options['nhead'] = model.getint('nhead')
        if 'num_encoder_layers' in model:
            options['num_encoder_layers'] = model.getint('num_encoder_layers')
        if 'num_decoder_layers' in model:
            options['num_decoder_layers'] = model.getint('num_decoder_layers')
        if 'dim_feedforward' in model:
            options['dim_feedforward'] = model.getint('dim_feedforward')
        if 'dropout' in model:
            options['dropout'] = model.getfloat('dropout')

        # Legacy parameters (kept for backward compatibility, ignored by model)
        if 'n_res_block' in model:
            options['n_res_block'] = model.getint('n_res_block')
        if 'n_res_channel' in model:
            options['n_res_channel'] = model.getint('n_res_channel')
        if 'decay' in model:
            options['decay'] = model.getfloat('decay')
        if 'stride' in model:
            options['stride'] = model.getint('stride')

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
