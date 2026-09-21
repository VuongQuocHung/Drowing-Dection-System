import importlib
# from .default_config import default_config
from .yowo_v2_config import yowo_v2_config


def get_d_cfg(args):
    module_name = f"config.{args.config}"
    module = importlib.import_module(module_name)
    d_cfg = getattr(module, 'config')
    return d_cfg


def build_model_config(args):
    d_cfg = get_d_cfg(args)
    print('==============================')
    print('Model Config: {} '.format(d_cfg['version'].upper()))
    
    if 'yowo_v2_' in d_cfg['version']:
        m_cfg = yowo_v2_config[d_cfg['version']]

    return m_cfg


def build_default_config(args):
    print('==============================')
    d_cfg = get_d_cfg(args)
    d_cfg['config'] = args.config
    if args.fold is not None:
        d_cfg['fold'] = args.fold

    d_cfg['train_splitlist'] = f"{d_cfg['fold']}_train.csv"
    d_cfg['test_splitlist'] = f"{d_cfg['fold']}_test.csv"
    d_cfg['test_save_path'] = f"./temp/{args.config}/{d_cfg['fold']}/"
    d_cfg['eval_save_path'] = f"./temp/{args.config}/{d_cfg['fold']}/"
    d_cfg['save_folder'] = f"./temp/{args.config}/{d_cfg['fold']}/"
    if args.weight is None:
        d_cfg['weight'] = f"{d_cfg['version']}_epoch_{d_cfg['epoch_weight']}.pth"
    else:
        d_cfg['weight'] = args.weight

    # d_cfg = default_config
    if args.cuda is not None:
        d_cfg['cuda']=args.cuda
    return d_cfg
