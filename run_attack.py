import torch
import wandb
from utils.utils import *
from utils.load_utils import *
import trainer_normal
import trainer_dshift
import trainer_ft
import os
import copy


def auto_save_name(config):
    if config.task_name in ['SL', 'sNTL', 'sCUTI']:
        save_path = f'saved_models/{config.task_name}_{config.domain_src}_{config.teacher_network}.pth'
    else:
        save_path = f'saved_models/{config.task_name}_{config.domain_src}_{config.domain_tgt}_{config.teacher_network}.pth'
    return save_path


if __name__ == '__main__':
    wandb.init(project='TransNTL', config='config/cifarstl/attack.yml')
    config = wandb.config
    setup_seed(config.seed)

    dataloader_train, dataloader_val, dataloader_test, datasets_name = load_data_tntl(config)
    model_ntl = load_model(config)
    model_ntl.eval()

    # pretrain
    if config.train_teacher_scratch:
        cprint('train model from scratch', 'magenta')
        cprint(f'method: {config.task_name}', 'yellow')
        if config.task_name in ['tNTL', 'sNTL']:
            trainer_func = trainer_normal.train_tntl
        elif config.task_name in ['tCUTI', 'sCUTI']:
            trainer_func = trainer_normal.train_tCUTI
        else:
            raise NotImplementedError
        trainer_func(config, dataloader_train, dataloader_val, 
                     model_ntl, datasets_name=datasets_name)
        # save
        if config.save_train_teacher:
            save_path = auto_save_name(config)
            cprint(f'save path: {save_path}')
            torch.save(model_ntl.state_dict(), save_path)
    else: 
        cprint('load saved parameters', 'magenta')
        # load trained model and evualate on test data
        if config.pretrained_teacher == 'auto':
            save_path = auto_save_name(config)
        else: 
            save_path = config.pretrained_teacher
        cprint(save_path)
        model_ntl.load_state_dict(torch.load(save_path))
        trainer_normal.eval_src(config, dataloader_test,
                             model_ntl, datasets_name=datasets_name)
    
    # attack 
    dataloader_train_srgt = load_surrogate_data(config, dataloader_train)
    model_surrogate = load_surrogate_model(config)
    model_surrogate.eval()

    if config.train_surrogate_scratch:
        cprint('train surrogate model from scratch', 'magenta')
        cprint(f'method: {config.how_to_train_surrogate}', 'yellow')
        # Fine-tuning-based attack methods
        model_surrogate = copy.deepcopy(model_ntl)
        # attack by TransNTL
        if config.how_to_train_surrogate == 'TransNTL':
            cprint('Fine-Tuning by TransNTL on Ds', 'yellow')
            ft_func = trainer_dshift.TransNTL
        # attack by FTAL
        elif config.how_to_train_surrogate == 'FT_FTAL':
            cprint('Fine-Tuning by FTAL on Ds', 'yellow')
            ft_func = trainer_ft.FTAL
        # attack by RTAL
        elif config.how_to_train_surrogate == 'FT_RTAL':
            cprint('Fine-Tuning by FTAL on Ds', 'yellow')
            ft_func = trainer_ft.RTAL
        # run attack
        ft_func(config, dataloader_train_srgt, dataloader_val, dataloader_test,
                model_ntl, model_surrogate, datasets_name)
        