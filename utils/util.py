import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
import yaml


class Dataset(Dataset):
    def __init__(self, dataset, pos):
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


def create_lstm_data(data, window):
    lstm_data = []
    L = len(data)
    for i in range(L - window):
        lstm_data.append(data[i: i + window + 1, :])
    return np.array(lstm_data)


def create_lstm_pos(pos, window):
    lstm_data = []
    L = len(pos)
    for i in range(L - window):
        lstm_data.append(pos[i: i + window + 1, :])
    return np.array(lstm_data)


class LSTM_Dataset(Dataset):
    def __init__(self, dataset):
        self.features = dataset[:, : -1, :].astype('float32')
        self.targets = dataset[:, -1, :].astype('float32')

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)


def load_model(model, model_path):
    model_load = torch.load(model_path)
    model.load_state_dict(model_load['state_dict'])
    return model