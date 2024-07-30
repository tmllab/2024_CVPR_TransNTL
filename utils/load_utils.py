from .utils import *
from data_split import Cus_Dataset
from torch.utils.data import DataLoader
import copy
import cv2, os


def load_model(config):
    cprint('load model', 'magenta')
    model = model_dict[config.teacher_network](
                pretrained=config.teacher_pretrain, 
                num_classes=config.num_classes, 
                img_size=config.image_size).to(config.device)
    return model


def load_surrogate_model(config):
    cprint('load surrogate model', 'magenta')
    model = model_dict[config.surrogate_network](
                pretrained=config.surrogate_pretrain, 
                num_classes=config.num_classes, 
                img_size=config.image_size).to(config.device)
    return model


def load_surrogate_data(config, dataloader_train):
    dataloader_train_srgt = []
    for loader in dataloader_train:
        part_dataset = copy.deepcopy(loader.dataset)
        num = len(part_dataset)
        ind = np.arange(num)
        ind = np.random.permutation(
            ind)[:int(num*config.surrogate_data_percen)]
        part_dataset.list_img1 = part_dataset.list_img1[ind]
        part_dataset.list_label1 = part_dataset.list_label1[ind]
        part_dataset.data_size = len(ind)
        dataloader_train_srgt.append(
            DataLoader(part_dataset, batch_size=config.batch_size,
                       shuffle=True, num_workers=config.num_workers,
                       drop_last=True))
    return dataloader_train_srgt


def load_data_tntl(config):
    cprint('load data', 'magenta')
    assert config.pre_split

    # source domain
    loaded_src = torch.load(
        f'data_presplit/{config.domain_src}_{config.image_size}.pth')
    datafile_src_train = loaded_src['train']
    datafile_src_val = loaded_src['val']
    datafile_src_test = loaded_src['test']
    
    dataloader_train = DataLoader(datafile_src_train, batch_size=config.batch_size,
                                    shuffle=True, num_workers=config.num_workers,
                                    drop_last=True)
    dataloader_val = DataLoader(datafile_src_val, batch_size=config.batch_size,
                                shuffle=False, num_workers=config.num_workers,
                                drop_last=False)
    dataloader_test = DataLoader(datafile_src_test, batch_size=config.batch_size,
                                shuffle=False, num_workers=config.num_workers,
                                drop_last=False)
    
    # target domain
    dataset_tgts_name = [config.domain_tgt]
    datasets_name = [config.domain_src] + dataset_tgts_name
    dataloader_tgt_train = []
    dataloader_tgt_val = []
    dataloader_tgt_test = []
    for tgt in dataset_tgts_name:
        loaded_tgt = torch.load(f'data_presplit/{tgt}_{config.image_size}.pth')
        datafile_tgt_train = loaded_tgt['train']
        datafile_tgt_val = loaded_tgt['val']
        datafile_tgt_test = loaded_tgt['test']
        dataloader_tgt_train.append(DataLoader(datafile_tgt_train,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=config.num_workers,
                                             drop_last=True))
        dataloader_tgt_val.append(DataLoader(datafile_tgt_val,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False))
        dataloader_tgt_test.append(DataLoader(datafile_tgt_test,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers,
                                             drop_last=False))
    dataloader_train = [dataloader_train] + dataloader_tgt_train
    dataloader_val = [dataloader_val] + dataloader_tgt_val
    dataloader_test = [dataloader_test] + dataloader_tgt_test

    return dataloader_train, dataloader_val, dataloader_test, datasets_name


# domain_digits = ['mt', 'us', 'sn', 'mm', 'sd']
domain_digits = ['mt', 'sn', 'mm', 'sd']
domain_cifar_stl = ['cifar', 'stl']
domain_visda = ['visda_v', 'visda_t']

