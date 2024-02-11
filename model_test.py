import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from utils import tool, util
from models.model import MLP, MLPM, GCN, GCNM, GCNR, MLPR


data_path = './data/39_bus_load.csv'
config_path = './config/config.yaml'
pos_path = './data/39Bus_rand_missing20.csv'
result_path = './result'
adj_path = './data/adj.xls'
test_file_path = './data/test_data'
config = util.load_config(config_path)
test_list = os.listdir(test_file_path)
config = util.load_config(config_path)
dataset = util.load_data(data_path)
pos = util.load_data(pos_path)
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
test = dataset[:2000, : config['n_features'] * config['n_nodes']]

MLP = MLP(config)
MLP = util.load_model(MLP, './result/model/MLP.pth.tar')
MLPM = MLPM(config)
MLPM = util.load_model(MLPM, './result/model/MLPM.pth.tar')
MLPR = MLPR(config)
MLPR = util.load_model(MLPR, './result/model/MLPR.pth.tar')
GCN = GCN(config)
GCN = util.load_model(GCN, './result/model/GCN.pth.tar')
GCNM = GCNM(config)
GCNM = util.load_model(GCNM, './result/model/GCNM.pth.tar')
GCNR = GCNR(config)
GCNR = util.load_model(GCNR, './result/model/GCNR.pth.tar')

models = [MLP, MLPM, MLPR, GCN, GCNM, GCNR]
models_names = ['MLP', 'MLPM', 'MLPR','GCN', 'GCNM', 'GCNR']

loss_mat = []
for model in models:
    loss_list = []
    for i in test_list:
        pos_path = test_file_path + '/' + i
        pos = util.load_data(pos_path)
        pos = pos[:2000, :config['n_features'] * config['n_nodes']]
        test_set = util.Dataset(test, pos)
        loss = tool.model_test(model, test_set, adj_path)
        loss_list.append(loss)
        torch.cuda.empty_cache()
    loss_mat.append(loss_list)

data = np.array(loss_mat)
df = pd.DataFrame(data.T, columns=models_names)
# need to rectify the f_name for different model type
f_name = result_path + '/test/Linear_test_loss_set.csv'
df.to_csv(f_name)
