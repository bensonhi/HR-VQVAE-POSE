import configparser
import sys

sys.path.append('../')


def model_option_parser(model_type, conf_path):
    config = configparser.ConfigParser()
    config.read(conf_path)
    model = config['Model']
    if model_type == 'pixelsnail':
        return {
            'shape': [model.getint('shape_height'), model.getint('shape_width')],
            'n_class': model.getint('n_class'),
            'channel': model.getint('channel'),
            'kernel_size': model.getint('kernel_size'),
            'n_block': model.getint('n_block'),
            'n_res_block': model.getint('n_res_block'),
            'res_channel': model.getint('res_channel'),
            'dropout': model.getfloat('dropout'),
            'n_cond_res_block': model.getint('n_cond_res_block'),
            'cond_res_channel': model.getint('cond_res_channel'),
            'cond_res_kernel': model.getint('cond_res_kernel'),
            'n_out_res_block': model.getint('n_out_res_block'),
            'attention': model.getboolean('attention')
        }
    elif model_type in ['vqvae','vqvae_1']:
        return {
            'in_channel': model.getint('in_channel'),
            'channel': model.getint('channel'),
            'n_res_block': model.getint('n_res_block'),
            'n_res_channel': model.getint('n_res_channel'),
            'embed_dim': model.getint('embed_dim'),
            'n_embed': model.getint('n_embed'),
            'decay': model.getfloat('decay'),
        }


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
