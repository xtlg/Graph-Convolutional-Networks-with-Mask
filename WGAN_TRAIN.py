import torch
from utils import train_tool

torch.manual_seed(1)
torch.cuda.manual_seed(1)

GAN_train = train_tool.WGAN_train(data_path='./data/39Bus.csv',
                                  pos_path='./data/39Bus_rand_missing20.csv',
                                  config_path='./config/config_WGAN.yaml')
