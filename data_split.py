from utils.utils import *
# from ntl_utils.getdata import Cus_Dataset
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import os


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Cus_Dataset(data.Dataset):
    def __init__(self, mode = None, \
                            dataset_1 = None, begin_ind1 = 0, size1 = 0,\
                            dataset_2 = None, begin_ind2 = 0, size2 = 0,\
                            dataset_3 = None, begin_ind3 = 0, size3 = 0,\
                            dataset_4 = None, begin_ind4 = 0, size4 = 0,\
                            new_model = None,
                            is_img_path = False, is_img_path_aug = [None, None, None, None],
                            spec_dataTransform = None, config=None):

        self.mode = mode
        self.list_img = []
        self.list_img1 = []
        self.list_img2 = []
        
        self.list_label = []
        self.list_label1 = []
        self.list_label2 = []

        self.data_size = 0
        if spec_dataTransform is None: 
            dataTransform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            self.transform = dataTransform
        else: 
            self.transform = spec_dataTransform
        # self.transform11 = dataTransform_ori
        self.is_img_path = is_img_path
        self.is_img_path_aug = is_img_path_aug

        if self.mode == 'authorized_training_src': 
            
            self.data_size = size1

            path_list1 = dataset_1[0][begin_ind1: begin_ind1+size1]
            label_list1 = dataset_1[1][begin_ind1: begin_ind1+size1]

            # list_img1 and list_label1
            for i in range(size1):
                self.list_img1.append(path_list1[i])
                self.list_label1.append(label_list1[i])

            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img1 = np.asarray(self.list_img1)
            self.list_img1 = self.list_img1[ind]

            self.list_label1 = np.asarray(self.list_label1)
            self.list_label1 = self.list_label1[ind]


        elif self.mode == 'val': #val data

            self.data_size = size1
            path_list = dataset_1[0][begin_ind1: begin_ind1+size1]
            if self.is_img_path:
                for file_path in path_list:
                    img = np.array(rgb_loader(file_path))
                    self.list_img.append(img)
            else:
                for file_path in path_list:
                    self.list_img.append(file_path)

            self.list_label = dataset_1[1][begin_ind1: begin_ind1+size1]


    def __getitem__(self, item):
        
        if self.mode == 'authorized_training_src':
            img1 = self.list_img1[item]
            # img1 = np.array(self.list_img1[item])
            # label1 = self.list_label1[item]
            label1 = self.list_label1[item]
            out1 = self.transform(img1)
            out2 = torch.LongTensor(label1)
            return out1, out2, out2, out2
        elif self.mode == 'val':
            img = self.list_img[item]
            # img = np.array(self.list_img[item])
            label = self.list_label[item]
            # label = np.array(self.list_label[item])
            # return self.transform(img), torch.LongTensor([label])
            return self.transform(img), torch.LongTensor(label).unsqueeze(0)


    def __len__(self):
        return self.data_size


def split():
    image_size = 64
    sample_num = 8000
    data_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

    domain_dict = {
        # 'mt': get_mnist_data,
        # # 'us': get_usps_data,
        # 'sn': get_svhn_data,
        # 'mm': get_mnist_m_data,
        # 'sd': get_syn_data,
        'cifar': get_cifar_data,
        'stl': get_stl_data,
        # 'visda_t': get_visda_data_src,
        # 'visda_v': get_visda_data_tgt
    }

    dataset_names = list(domain_dict.keys())
    dataset_funcs = list(domain_dict.values())

    dataset_split_seed = 2021
    setup_seed(dataset_split_seed)

    if not os.path.exists('./data_presplit'):
        os.makedirs('./data_presplit')

    for name, dataset_funcs in zip(dataset_names, dataset_funcs):
        data = dataset_funcs()
        datafile_train = Cus_Dataset(mode='authorized_training_src',
                            # source domain
                            dataset_1=data,
                            begin_ind1=2000,
                            size1=sample_num,
                            # others
                            is_img_path_aug=[False, False, False, False],
                            spec_dataTransform=data_transforms)
        datafile_val = Cus_Dataset(mode='val',
                            # source domain
                            dataset_1=data,
                            begin_ind1=0,
                            size1=1000,
                            # others
                            is_img_path_aug=[False, False, False, False],
                            spec_dataTransform=data_transforms)
        datafile_test = Cus_Dataset(mode='val',
                            # source domain
                            dataset_1=data,
                            begin_ind1=1000,
                            size1=1000,
                            # others
                            is_img_path_aug=[False, False, False, False],
                            spec_dataTransform=data_transforms)

        torch.save({'train': datafile_train, 'val': datafile_val, 'test': datafile_test}, 
                   f'data_presplit/{name}_{image_size}.pth')

        pass


if __name__ == '__main__':

    split()
    exit()



