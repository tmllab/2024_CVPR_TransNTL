from .ntl_utils.utils_digit import get_mnist_data, get_syn_data, get_mnist_m_data, get_svhn_data, get_usps_data
from .ntl_utils.getdata import Cus_Dataset
from .ntl_utils.utils_cifar_stl import get_cifar_data, get_stl_data
from .ntl_utils.utils_visda import get_visda_data_tgt, get_visda_data_src
import models.ntl_vggnet as ntl_vggnet
import torch
import numpy as np
import random 
from termcolor import cprint
import torchvision


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

domain_dict = {'mt': get_mnist_data,
               'us': get_usps_data,
               'sn': get_svhn_data,
               'mm': get_mnist_m_data,
               'sd': get_syn_data,
               'cifar': get_cifar_data,
               'stl': get_stl_data,
               'visda_t': get_visda_data_src,
               'visda_v': get_visda_data_tgt}

model_dict = {'vgg11': ntl_vggnet.vgg11,
              'vgg13': ntl_vggnet.vgg13,
              'vgg19': ntl_vggnet.vgg19}


# domain_digits = ['mt', 'us', 'sn', 'mm', 'sd']
domain_digits = ['mt', 'sn', 'mm', 'sd']
domain_cifar_stl = ['cifar', 'stl']
domain_visda = ['visda_v', 'visda_t']

