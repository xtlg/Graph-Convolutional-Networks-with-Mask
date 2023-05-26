import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
import yaml


class Dataset(Dataset):
    def __init__(self, dataset, pos, config):
        self.targets = dataset.astype('float32')
        self.pos = pos.astype('float32')
        self.features = self.targets * self.pos

    def __getitem__(self, index):
        return self.features[index], self.targets[index], self.pos[index]

    def __len__(self):
        return len(self.features)


def load_config(yaml_file_path):
    with open(yaml_file_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_data(data_path):
    # don't consider the flag of power flow
    data = pd.read_csv(data_path, header=None, index_col=None)
    data = data.values
    return data[:, :]


def get_adj(path):
    adj = pd.read_excel(path)
    adj = adj.dropna(axis=1)
    adj = adj.values
    adj = torch.FloatTensor(adj)
    # adj = adj[:, :]
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + torch.eye(adj.shape[0]))
    adj = torch.FloatTensor(adj)
    return adj

