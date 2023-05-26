import torch
from utils import train_tool

torch.manual_seed(1)
torch.cuda.manual_seed(1)

GCNM_train = train_tool.GCN_train(data_path='./data/39Bus.csv',
                                  pos_path='./data/39Bus_rand_missing20.csv',
                                  config_path='./config/config_mask.yaml',
                                  adj_path='./data/adj.xls')

GCN_train = train_tool.GCN_train(data_path='./data/39Bus.csv',
                                 pos_path='./data/39Bus_rand_missing20.csv',
                                 config_path='./config/config_no_mask.yaml',
                                 adj_path='./data/adj.xls')

