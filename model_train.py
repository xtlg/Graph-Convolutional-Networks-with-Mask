import torch
from sklearn.preprocessing import MinMaxScaler

from utils import tool, util
from models.model import MLP, MLPM, GCN, GCNM, GCNR, MLPR

torch.manual_seed(1)
torch.cuda.manual_seed(1)

data_path = './data/39_bus_load.csv'
config_path = './config/config.yaml'
pos_path = './data/39Bus_rand_missing20.csv'

config = util.load_config(config_path)
dataset = util.load_data(data_path)
pos = util.load_data(pos_path)
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
train_set = dataset[:6000, : config['n_features'] * config['n_nodes']]
pos_train_set = pos[:6000, : config['n_features'] * config['n_nodes']]
val_set = dataset[6000: 8000, :config['n_features'] * config['n_nodes']]
pos_val_set = pos[6000: 8000, :config['n_features'] * config['n_nodes']]
train_set = util.Dataset(train_set, pos_train_set)
val_set = util.Dataset(val_set, pos_val_set)

MLP = MLP(config)
MLPM = MLPM(config)
MLPR = MLPR(config)
GCN = GCN(config)
GCNM = GCNM(config)
GCNR = GCNR(config)

models = [MLP, MLPM, MLPR, GCN, GCNM, GCNR]
models_names = ['MLP', 'MLPM', 'MLPR','GCN', 'GCNM', 'GCNR']

num = 0
for model in models:
    models_name = models_names[num]
    num += 1
    tool.model_train(model, models_name, train_set, val_set,
                     adj_path='./data/adj.xls',
                     result_path='./result')
    torch.cuda.empty_cache()
