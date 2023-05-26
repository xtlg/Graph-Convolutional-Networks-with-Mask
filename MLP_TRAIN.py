import torch

import models
import utils.util
from utils import train_tool

torch.manual_seed(1)
torch.cuda.manual_seed(1)

MLP_train = train_tool.MLP_train(data_path='./data/39Bus.csv',
                                 pos_path='./data/39Bus_rand_missing20.csv',
                                 config_path='./config/config_mask.yaml')

MLP_train = train_tool.MLP_M_train(data_path='./data/39Bus.csv',
                                   pos_path='./data/39Bus_rand_missing20.csv',
                                   config_path='./config/config_mask.yaml')

train_tool.MLP_hy_train(data_path='./data/39Bus.csv',
                        pos_path='./data/39Bus_rand_missing20.csv',
                        config_path='./config/config_mask.yaml')
